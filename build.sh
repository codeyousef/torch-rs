#!/bin/bash
# Simple build script for torch-rs project

set -e  # Exit on any error

echo "🚀 Building torch-rs project..."

# Activate virtual environment if it exists
if [ -d "torch_test_env" ]; then
    echo "📦 Activating PyTorch environment..."
    source torch_test_env/bin/activate
fi

# Build the project
echo "🔨 Building base library..."
cargo build

# Run tests
echo "🧪 Running tests..."
cargo test

# Build documentation
echo "📚 Building documentation..."
cargo doc --no-deps

echo "✅ Build completed successfully!"
echo ""
echo "📖 View documentation: target/doc/tch/index.html"
echo "🏃 Run examples: cargo run --example <example_name>"