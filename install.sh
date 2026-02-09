#!/bin/bash
set -e

echo ""
echo "Installing Interweave -- voice I/O for Claude Code"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is required but not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

# Install the package and dependencies
echo "[1/3] Installing package and dependencies..."
echo "      This may take a few minutes on first install."
echo ""
uv tool install --python ">=3.11" interweave 2>&1 || uv tool upgrade --python ">=3.11" interweave 2>&1
echo ""

# Download ML models
echo "[2/3] Downloading ML models..."
echo ""

PYTHON="$(dirname "$(which interweave)")"/python

echo "      Silero VAD (voice activity detection)..."
$PYTHON -c "from silero_vad import load_silero_vad; load_silero_vad(); print('      Done.')"

echo "      Parakeet MLX (speech-to-text)..."
$PYTHON -c "import parakeet_mlx; parakeet_mlx.from_pretrained('animaslabs/parakeet-tdt-0.6b-v3-mlx'); print('      Done.')"

echo "      Kokoro MLX (text-to-speech)..."
$PYTHON -c "from mlx_audio.tts import load_model; m = load_model('mlx-community/Kokoro-82M-bf16'); print('      Done.')"

echo ""

# Register with Claude Code
echo "[3/3] Registering with Claude Code..."
if command -v claude &> /dev/null; then
    claude mcp add interweave -- interweave 2>&1 || true
    echo ""
    echo "Done! Start a new Claude Code session and talk."
else
    echo "      Claude Code CLI not found. Register manually:"
    echo "      claude mcp add interweave -- interweave"
fi

echo ""
