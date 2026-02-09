# Interweave

Voice I/O MCP server for Claude Code. Speak and listen through your mic and speakers.

Uses Kokoro MLX (TTS), Parakeet MLX (STT), and Silero VAD. Runs locally on Apple Silicon.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/EndlessHoper/interweave/main/install.sh | bash
```

This will:
1. Install Python dependencies (~2GB, includes torch and ML libraries)
2. Download ML models (Kokoro TTS, Parakeet STT, Silero VAD)
3. Register interweave with Claude Code

After install, start a new Claude Code session and talk.

### Manual install

If you prefer to install step by step:

```bash
uv pip install --system interweave
interweave --warmup
claude mcp add interweave -- interweave
```

### Codex

```bash
uv pip install --system interweave
interweave --warmup
codex mcp add interweave -- interweave
```

### OpenCode

```bash
uv pip install --system interweave
interweave --warmup
```

Then add to `opencode.json`:

```json
{
  "mcp": {
    "interweave": {
      "type": "local",
      "command": ["interweave"],
      "enabled": true
    }
  }
}
```

## Requirements

- macOS with Apple Silicon
- Python 3.11+
- Microphone and speakers
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
