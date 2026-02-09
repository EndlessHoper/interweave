# Interweave

Voice I/O MCP server for Claude Code. Speak and listen through your mic and speakers.

Uses Kokoro MLX (TTS), Parakeet MLX (STT), and Silero VAD. Runs locally on Apple Silicon.

## Install

```
claude mcp add interweave -- uvx interweave
```

### Other tools

**Codex:**
```
codex mcp add interweave -- uvx interweave
```

**OpenCode** (`opencode.json`):
```json
{
  "mcp": {
    "interweave": {
      "type": "local",
      "command": ["uvx", "interweave"],
      "enabled": true
    }
  }
}
```

## Requirements

- macOS with Apple Silicon
- Python 3.11+
- Microphone and speakers
