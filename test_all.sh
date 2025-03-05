#!/bin/bash

CPU=false
FOLDS=()

usage() {
    echo "Usage: $0 --cpu [bool] --folds [list of numbers]"
    echo ""
    echo "Arguments:"
    echo "  --cpu [bool]       If test should run on CPU (true) or on GPU (false) (default: false)"
    echo "  --folds [list]     A space-separated list of fold numbers (1-10) (default: all)"
    exit 1
}

validate_number() {
    if [[ "$1" -ge 1 && "$1" -le 10 ]]; then
        return 0
    else
        return 1
    fi
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --folds)
            shift
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                FOLDS+=("$1")
                shift
            done
            ;;
    esac
done

# Default: all folds
if [ ${#FOLDS[@]} -eq 0 ]; then
    FOLDS=(1 2 3 4 5 6 7 8 9 10)
fi

for fold in "${FOLDS[@]}"; do
    if ! validate_number "$fold"; then
        echo "Error: $fold is out of range (1-10)."
        exit 1
    fi
done

echo "Running on CPU: $CPU"
echo "Folds: ${FOLDS[@]}"

for fold in "${FOLDS[@]}"; do
    echo "Runing test on fold $fold"
    
    # Run for variation 1
    python test_model.py --variation 1 --fold "$fold" --cpu "$CPU" 2>&1 | tee "test_var1_fold${fold}.log"

    # Run for variation 2
    python test_model.py --variation 2 --fold "$fold" --cpu "$CPU" 2>&1 | tee "test_var2_fold${fold}.log"
    
done
