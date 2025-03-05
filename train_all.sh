#!/bin/bash

FOLDS=()

usage() {
    echo "Usage: $0 --folds [list of numbers]"
    echo ""
    echo "Arguments:"
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

echo "Folds: ${FOLDS[@]}"

for fold in "${FOLDS[@]}"; do

    echo ""
    echo "-----------------------------------------"
    echo "Runing Training on fold $fold"  
    echo ""
    echo ""
    
    # Task 1
    bash train.sh --model base --variation 1 --fold "$fold"
    bash train.sh --model instruct --variation 1 --fold "$fold"
    

    # Task 2
    bash train.sh --model base --variation 2 --fold "$fold"
    bash train.sh --model instruct --variation 2 --fold "$fold"

done

echo "Finshed training folds: ${FOLDS[@]}"
