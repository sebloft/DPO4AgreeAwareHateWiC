#!/bin/bash

MODEL="base"
VARIATION=""
FOLD=""

usage() {
    echo "Usage: $0 --model MODEL --variation VARIATION --fold FOLD"
    echo ""
    echo "Arguments:"
    echo "  --model MODEL       Model type (default: 'base')"
    echo "  --variation VARIATION  Variation number of the model"
    echo "  --fold FOLD         Fold number for cross-validation"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --variation)
            VARIATION="$2"
            shift 2
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$FOLD" ]] || [[ -z "$VARIATION" ]]; then
    usage
fi

if [[ "$MODEL" == "base" ]]; then
    MODEL_NAME="OpenRLHF/Llama-3-8b-sft-mixture"
    EPOCHS=15
elif [[ "$MODEL" == "instruct" ]]; then
    MODEL_NAME="OpenRLHF/Llama-3-8b-rlhf-100k"
    EPOCHS=10
else
    echo "Error: Invalid model type specified. Use 'base' or 'instruct'."
    exit 1
fi

cd ${MODEL}

python ../trl_dpo.py \
    --dataset_name sebloft/dpo_task${VARIATION} \
    --dataset_config fold${FOLD} \
    --model_name_or_path ${MODEL_NAME} \
    --learning_rate 5.0e-7 \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --dataset_train_split train \
    --dataset_test_split dev \
    --eval_strategy epoch \
    --output_dir ./${MODEL}_var${VARIATION}_fold${FOLD} \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \  \
    2>&1 | tee train_var${VARIATION}_fold${FOLD}.log

cd ..

