#!/bin/bash
# Track compilation errors for torch-rs features

echo "=== Compilation Error Tracker ==="
echo "Date: $(date)"
echo "Feature: torch-rs"
echo ""

# Activate virtual environment
source torch_test_env/bin/activate
export LIBTORCH_USE_PYTORCH=1

# Count errors
echo "Counting compilation errors..."
ERROR_COUNT=$(cargo check --features torch-rs 2>&1 | grep -c "^error")

echo "Total compilation errors: $ERROR_COUNT"

# Show error categories
echo ""
echo "Error categories:"
cargo check --features torch-rs 2>&1 | grep "^error\[E" | sed 's/error\[E\([0-9]*\)\].*/E\1/' | sort | uniq -c | sort -rn

# Save to log
LOG_FILE="compilation_errors_$(date +%Y%m%d_%H%M%S).log"
cargo check --features torch-rs 2>&1 > "$LOG_FILE"
echo ""
echo "Full error log saved to: $LOG_FILE"