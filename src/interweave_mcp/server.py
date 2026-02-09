"""
Interweave MCP Server — voice I/O tools for Claude Code.

Tools: speak_and_listen, speak, listen
Models (kept warm via lifespan): Parakeet MLX (STT), Kokoro MLX (TTS), Silero VAD
Transport: stdio

NOTE: An AEC (echo cancellation) based interruption detection approach was tried
using speexdsp to subtract TTS output from mic input during playback, then running
VAD on the cleaned signal. It didn't work reliably — the AEC couldn't fully remove
speaker bleed, causing the VAD to hear the TTS voice as user speech and triggering
false interruptions. We now use a simpler turn-based approach: speak fully, then listen.
"""

from __future__ import annotations

import atexit
import logging
import queue
import re
import signal
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from mcp.server.fastmcp import FastMCP, Context
from silero_vad import VADIterator, load_silero_vad

# ── Shutdown coordination ────────────────────────────────────────────────────
# Set by lifespan cleanup or signal handlers so blocking loops exit promptly.

_shutdown = threading.Event()

# ── Config ───────────────────────────────────────────────────────────────────

PARAKEET_HF_REPO = "animaslabs/parakeet-tdt-0.6b-v3-mlx"
KOKORO_MLX_REPO = "mlx-community/Kokoro-82M-bf16"
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.1
KOKORO_SAMPLE_RATE = 24000

SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512             # 32ms at 16kHz (Silero requirement)
VAD_THRESHOLD = 0.4                 # speech probability threshold
VAD_MIN_SILENCE_MS = 1500           # ms of silence after speech to end turn
MAX_SPEECH_SEC = 600.0  # 10 min — no practical limit on how long the user speaks
START_TIMEOUT_SEC = 30.0

# ── Logging (stderr only — stdout is the MCP protocol channel) ──────────────

log = logging.getLogger("interweave")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False
logging.getLogger("phonemizer").setLevel(logging.ERROR)


# ── Helpers ──────────────────────────────────────────────────────────────────


def sanitize_no_dashes(text: str) -> str:
    text = re.sub(r"[-\u2013\u2014]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Recorder (Silero VAD) ───────────────────────────────────────────────────


class Recorder:
    def __init__(self):
        log.info("Loading Silero VAD...")
        self.vad_model = load_silero_vad()
        log.info("Silero VAD ready.")

    def record_utterance(self) -> Optional[np.ndarray]:
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        vad = VADIterator(
            self.vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )

        def on_audio(indata, frames, time_info, status):
            audio_queue.put(indata.copy().reshape(-1))

        chunks: list[np.ndarray] = []
        speech_active = False
        t0 = time.monotonic()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=on_audio,
        ):
            while not _shutdown.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if not speech_active and time.monotonic() - t0 > START_TIMEOUT_SEC:
                        vad.reset_states()
                        return None
                    continue

                chunks.append(chunk)
                event = vad(torch.from_numpy(chunk))

                if event is not None:
                    if "start" in event:
                        speech_active = True
                        log.info("VAD: speech started")
                    if "end" in event and speech_active:
                        log.info("VAD: speech ended")
                        break

                if speech_active and time.monotonic() - t0 > MAX_SPEECH_SEC:
                    log.info("VAD: max duration reached")
                    break

        vad.reset_states()
        if not chunks or not speech_active:
            return None
        return np.concatenate(chunks, axis=0)


# ── STT (Parakeet MLX) ──────────────────────────────────────────────────────


class STT:
    def __init__(self):
        import parakeet_mlx

        log.info("Loading Parakeet MLX: %s", PARAKEET_HF_REPO)
        t0 = time.monotonic()
        self.model = parakeet_mlx.from_pretrained(PARAKEET_HF_REPO)
        log.info("Parakeet ready (%.1fs)", time.monotonic() - t0)

    def transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            sf.write(tmp, audio, SAMPLE_RATE, subtype="PCM_16")
            t0 = time.monotonic()
            result = self.model.transcribe(str(tmp))
            text = sanitize_no_dashes(result.text)
            log.info("STT (%.1fs): %s", time.monotonic() - t0, text)
            return text
        finally:
            tmp.unlink(missing_ok=True)


# ── Streaming audio playback buffer ─────────────────────────────────────────


class _AudioBuffer:
    """Feeds audio to sd.OutputStream as it arrives, signals when done."""

    def __init__(self):
        self._buf = np.empty(0, dtype=np.float32)
        self._pos = 0
        self._done = False
        self._event = threading.Event()
        self._lock = threading.Lock()

    def put(self, audio: np.ndarray) -> None:
        with self._lock:
            self._buf = np.concatenate([self._buf, audio.astype(np.float32)])

    def finish(self) -> None:
        self._done = True

    def stop(self) -> None:
        self._done = True

    def wait(self) -> None:
        self._event.wait()

    def on_finished(self) -> None:
        self._event.set()

    def callback(self, outdata, frames, time_info, status):
        with self._lock:
            available = len(self._buf) - self._pos
        if available >= frames:
            outdata[:, 0] = self._buf[self._pos:self._pos + frames]
            self._pos += frames
        elif available > 0:
            outdata[:available, 0] = self._buf[self._pos:self._pos + available]
            outdata[available:] = 0
            self._pos += available
        else:
            outdata[:] = 0
            if self._done:
                raise sd.CallbackStop


# ── TTS (Kokoro via mlx-audio) ───────────────────────────────────────────────


class TTS:
    def __init__(self):
        from mlx_audio.tts.utils import load_model

        log.info("Loading Kokoro MLX: %s (voice=%s)", KOKORO_MLX_REPO, KOKORO_VOICE)
        t0 = time.monotonic()
        self.model = load_model(KOKORO_MLX_REPO)
        for _ in self.model.generate(text="ok", voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang_code="a"):
            pass
        log.info("Kokoro MLX ready (%.1fs)", time.monotonic() - t0)

    def speak(self, text: str) -> None:
        clean = sanitize_no_dashes(text.replace("\n", " ").strip())
        if not clean:
            return
        t0 = time.monotonic()
        buf = _AudioBuffer()

        stream = sd.OutputStream(
            samplerate=KOKORO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=buf.callback,
            blocksize=2048,
            finished_callback=buf.on_finished,
        )
        stream.start()

        first_chunk = True
        for result in self.model.generate(text=clean, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang_code="a"):
            if _shutdown.is_set():
                break
            audio = np.array(result.audio, dtype=np.float32)
            if first_chunk:
                log.info("TTS first chunk in %.2fs", time.monotonic() - t0)
                first_chunk = False
            buf.put(audio)

        buf.finish()
        # Wait for playback with timeout so shutdown isn't blocked
        while not buf._event.is_set():
            if _shutdown.is_set():
                buf.stop()
                break
            buf._event.wait(timeout=0.1)
        stream.stop()
        stream.close()
        log.info("TTS done (%.2fs total): %s", time.monotonic() - t0, clean[:80])


# ── MCP Server ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(server):
    """Load all models once at startup, keep warm for the session."""
    log.info("Interweave MCP starting — loading models...")
    stt = STT()
    tts = TTS()
    recorder = Recorder()
    log.info("All models loaded. Ready.")
    try:
        yield {"stt": stt, "tts": tts, "recorder": recorder}
    finally:
        log.info("Shutting down — signalling blocking loops to exit...")
        _shutdown.set()


mcp_server = FastMCP("interweave_mcp", lifespan=lifespan)


@mcp_server.tool(
    name="interweave_speak_and_listen",
    annotations={
        "title": "Speak text and listen for response",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def interweave_speak_and_listen(text: str, ctx: Context) -> str:
    """Speak text aloud via TTS, then listen for the user's spoken response.

    ONLY use this when you need the user's input to continue. Examples:
    - Asking a question: "Should I skip this one or send a request?"
    - Asking for confirmation: "I found 3 matches. Want me to go through them?"
    - Asking for a decision you can't make yourself

    Do NOT use this for status updates, progress reports, or informational
    messages where you plan to keep working. Use interweave_speak instead
    for those, so you don't block waiting for a reply the user doesn't
    need to give.

    Think of it this way: if you're going to do more work after speaking,
    use interweave_speak. If you need the user to answer before you can
    continue, use this.

    IMPORTANT — the text goes through a TTS voice engine. Write text that
    sounds natural when spoken aloud:
    - No asterisks, markdown, or formatting characters
    - No bullet points, numbered lists, or headings
    - No parenthetical asides or footnotes
    - No emojis or special unicode characters
    - No dashes (hyphens, en dashes, em dashes) — use commas or periods instead
    - Write contractions naturally (don't, can't, I'm, you're)
    - Spell out abbreviations and acronyms on first use
    - Keep sentences short and conversational

    Args:
        text: Plain spoken text. Must be clean prose with no formatting.
        ctx: MCP context (injected automatically).

    Returns:
        The transcribed user speech, or an error/status message.
    """
    state = ctx.request_context.lifespan_context
    tts: TTS = state["tts"]
    stt: STT = state["stt"]
    recorder: Recorder = state["recorder"]

    # Speak fully, then listen (turn-based)
    tts.speak(text)
    audio = recorder.record_utterance()
    if audio is None:
        return "[no speech detected within timeout]"

    transcript = stt.transcribe(audio)
    if not transcript:
        return "[could not transcribe audio]"
    return transcript


@mcp_server.tool(
    name="interweave_speak",
    annotations={
        "title": "Speak text aloud",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def interweave_speak(text: str, ctx: Context) -> str:
    """Speak text aloud via TTS without listening for a response.

    USE THIS when you're giving a status update and plan to keep working.
    This is the right choice most of the time. Examples:
    - "I'll go through the listings now." (then start browsing)
    - "Skipping this one, no pets allowed." (then move to next)
    - "Done, I sent 3 viewing requests." (final summary)
    - "Let me check that for you." (then go do the work)

    After speaking, you continue with your next action immediately.
    The user hears you but doesn't need to reply.

    Only use interweave_speak_and_listen instead when you genuinely
    need the user to answer a question before you can proceed.

    IMPORTANT — the text goes through a TTS voice engine. Write text that
    sounds natural when spoken aloud:
    - No asterisks, markdown, formatting, emojis, or special characters
    - No dashes of any kind — use commas or periods instead
    - No bullet points, numbered lists, or headings
    - Write clean, conversational prose only

    Args:
        text: Plain spoken text. Must be clean prose with no formatting.
        ctx: MCP context (injected automatically).

    Returns:
        Confirmation that the text was spoken.
    """
    state = ctx.request_context.lifespan_context
    tts: TTS = state["tts"]
    tts.speak(text)
    return f"Spoke: {text[:100]}"


@mcp_server.tool(
    name="interweave_listen",
    annotations={
        "title": "Listen for user speech",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def interweave_listen(ctx: Context) -> str:
    """Listen for the user's spoken input and transcribe it.

    Records audio from the microphone using VAD (voice activity detection)
    to determine when the user starts and stops speaking, then transcribes
    using Parakeet MLX.

    Args:
        ctx: MCP context (injected automatically).

    Returns:
        The transcribed user speech, or a status message if no speech detected.
    """
    state = ctx.request_context.lifespan_context
    stt: STT = state["stt"]
    recorder: Recorder = state["recorder"]

    audio = recorder.record_utterance()
    if audio is None:
        return "[no speech detected within timeout]"

    transcript = stt.transcribe(audio)
    if not transcript:
        return "[could not transcribe audio]"
    return transcript


def _on_shutdown_signal(signum, frame):
    log.info("Received signal %s, shutting down...", signum)
    _shutdown.set()


def main():
    signal.signal(signal.SIGTERM, _on_shutdown_signal)
    signal.signal(signal.SIGINT, _on_shutdown_signal)
    atexit.register(_shutdown.set)
    mcp_server.run()


if __name__ == "__main__":
    main()
