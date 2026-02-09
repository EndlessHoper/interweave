#!/bin/bash
set -e

echo ""
echo "🎙️  Installing Interweave — voice I/O for Claude Code"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv is required but not installed."
    echo "   Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

# Install the package and dependencies
echo "📦 Installing dependencies (torch, mlx-audio, etc.)..."
echo "   This may take a few minutes on first install."
echo ""
uv pip install interweave

# Download ML models
echo ""
echo "🧠 Downloading ML models..."
echo ""
interweave --warmup

# Register with Claude Code
if command -v claude &> /dev/null; then
    echo "🔧 Registering with Claude Code..."
    claude mcp add interweave -- interweave
    echo ""
    echo "✅ Done! Start a new Claude Code session and try:"
    echo '   "hey, can you hear me?" (interweave will speak and listen)'
else
    echo "⚠️  Claude Code CLI not found. Register manually:"
    echo "   claude mcp add interweave -- interweave"
fi

echo ""
