#!/bin/bash

FOLDS=()
VARIATIONS=()

usage() {
    echo "Usage: $0 --folds [list of numbers]"
    echo ""
    echo "Arguments:"
    echo "  --folds [list]      A space-separated list of fold numbers (1-10) (default: all)"
    echo "  --variations [list] A space-separated list of variations numbers (1-2) (default: all)"
    exit 1
}

validate_fold_number() {
    if [[ "$1" -ge 1 && "$1" -le 10 ]]; then
        return 0
    else
        return 1
    fi
}

validate_variation_number() {
    if [[ "$1" -ge 1 && "$1" -le 2 ]]; then
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

        --variations)
            shift
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                VARIATIONS+=("$1")
                shift
            done
            ;;
    esac
done

# Default: all folds
if [ ${#FOLDS[@]} -eq 0 ]; then
    FOLDS=(1 2 3 4 5 6 7 8 9 10)
fi

# Default: all variations
if [ ${#FOLDS[@]} -eq 0 ]; then
    VARIATIONS=(1 2)
fi

for fold in "${FOLDS[@]}"; do
    if ! validate_fold_number "$fold"; then
        echo "Error: $fold is out of range (1-10)."
        exit 1
    fi
done

for variation in "${VARIATIONS[@]}"; do
    if ! validate_variation_number "$variation"; then
        echo "Error: $variation is out of range (1-2)."
        exit 1
    fi
done

echo "Folds: ${FOLDS[@]}"
echo "Variations: ${FOLDS[@]}"

for fold in "${FOLDS[@]}"; do

    echo ""
    echo "-----------------------------------------"
    echo "Runing Training on fold $fold"  
    echo ""
    echo ""

    for variation in "${VARIATIONS[@]}"; do

        echo ""
        echo "Starting task $variation"
        echo ""

        bash train.sh --model base --variation "$variation" --fold "$fold"
        bash train.sh --model instruct --variation "$variation" --fold "$fold"
        

    done

done

echo "Finshed training folds: ${FOLDS[@]}, variations: ${VARIATIONS[@]}"
