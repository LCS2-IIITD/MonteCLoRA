# !/bin/bash


LEARNING_RATES=(5e-5 1e-4 5e-4 1e-3)


METHOD_NAME="dora" 
export CUDA_VISIBLE_DEVICES=0
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    echo "Starting fine-tuning with learning rate: $LEARNING_RATE"
    echo "Method: $METHOD_NAME"
    echo "=================================================="
    
    python finetune_gsm8k.py \
        --base_model 'meta-llama/Llama-3.2-3B-Instruct' \
        --data_path 'gsm8k' \
        --output_dir "./trained-models/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}" \
        --method "${METHOD_NAME}" \
        --subsample_split 1.0 \
        --batch_size 8 \
        --kl_loss_weight 1e-5 \
        --use_entropy True \
        --micro_batch_size 4 \
        --num_epochs 3 \
        --num_epochs_coop 0 \
        --lora_r 32 \
        --learning_rate ${LEARNING_RATE} \
        --learning_rate_coop ${LEARNING_RATE} \
        --cutoff_len 256 \
        --val_set_size 120 \
        --apply_peft \
        --eval_step 233 \
        --save_step 233 \
        --num_virtual_tokens 10 \
        --lora_alpha 64 \
        --posthoc_app 0 \
        --lora_target_modules '["q_proj", "k_proj", "v_proj"]' \
        --target_modules '["q_proj", "k_proj", "v_proj"]' \
        --use_monteclora False \
        --sample_scaler 2.5e-5 \
        --monteclora_at '' \
        --monteclora_targets '["q_proj", "k_proj", "v_proj"]' \
        --wandb_project 'MonteCLoRA-gsm' \
        --wandb_run_name "llama-3.2-3B-${METHOD_NAME}-${LEARNING_RATE}-qkv"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed training with learning rate: $LEARNING_RATE"
    else
        echo "Training failed with learning rate: $LEARNING_RATE"
        echo "Continuing with next learning rate..."
    fi
    
    echo "=================================================="
    echo
done

echo "All fine-tuning runs completed!"

METHOD_NAME="monteclora" 

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    echo "Starting fine-tuning with learning rate: $LEARNING_RATE"
    echo "Method: $METHOD_NAME"
    echo "=================================================="
    
    python finetune_gsm8k.py \
        --base_model 'meta-llama/Llama-3.2-3B-Instruct' \
        --data_path 'gsm8k' \
        --output_dir "./trained-models/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}" \
        --method "${METHOD_NAME}" \
        --subsample_split 1.0 \
        --batch_size 8 \
        --kl_loss_weight 1e-5 \
        --use_entropy True \
        --micro_batch_size 4 \
        --num_epochs 3 \
        --num_epochs_coop 0 \
        --lora_r 32 \
        --learning_rate ${LEARNING_RATE} \
        --learning_rate_coop ${LEARNING_RATE} \
        --cutoff_len 256 \
        --val_set_size 120 \
        --apply_peft \
        --eval_step 233 \
        --save_step 233 \
        --num_virtual_tokens 10 \
        --lora_alpha 64 \
        --posthoc_app 0 \
        --lora_target_modules '["q_proj", "k_proj", "v_proj"]' \
        --target_modules '["q_proj", "k_proj", "v_proj"]' \
        --use_monteclora True \
        --sample_scaler 2.5e-5 \
        --monteclora_at 'lora_A' \
        --monteclora_targets '["q_proj", "k_proj", "v_proj"]' \
        --wandb_project 'MonteCLoRA-gsm' \
        --wandb_run_name "llama-3.2-3B-${METHOD_NAME}-${LEARNING_RATE}-qkv"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed training with learning rate: $LEARNING_RATE"
    else
        echo "Training failed with learning rate: $LEARNING_RATE"
        echo "Continuing with next learning rate..."
    fi
    
    echo "=================================================="
    echo
done

echo "All fine-tuning runs completed!"


# # !/bin/bash

METHOD_NAME="monteclora" 

LEARNING_RATES=(1e-5 5e-5 1e-4 5e-4 1e-3)

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    echo "Starting fine-tuning with learning rate: $LEARNING_RATE"
    echo "Method: $METHOD_NAME"
    echo "=================================================="
    
    lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,peft="./trained-models/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}/checkpoint-2802" \
    --tasks gsm8k_cot_llama \
    --log_samples \
    --output_path ./result/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}/checkpoint-2802  \
    --batch_size 32   --trust_remote_code --confirm_run_unsafe_code \
    --fewshot_as_multiturn --apply_chat_template
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed training with learning rate: $LEARNING_RATE"
    else
        echo "Training failed with learning rate: $LEARNING_RATE"
        echo "Continuing with next learning rate..."
    fi
    
    echo "=================================================="
    echo
done

METHOD_NAME="lora" 

LEARNING_RATES=(1e-5 5e-5 1e-4 5e-4 1e-3)

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    echo "Starting fine-tuning with learning rate: $LEARNING_RATE"
    echo "Method: $METHOD_NAME"
    echo "=================================================="
    
    lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,peft="./trained-models/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}/checkpoint-2802" \
    --tasks gsm8k_cot_llama \
    --log_samples \
    --output_path ./result/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}/checkpoint-2802  \
    --batch_size 32   --trust_remote_code --confirm_run_unsafe_code \
    --fewshot_as_multiturn --apply_chat_template
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed training with learning rate: $LEARNING_RATE"
    else
        echo "Training failed with learning rate: $LEARNING_RATE"
        echo "Continuing with next learning rate..."
    fi
    
    echo "=================================================="
    echo
done
METHOD_NAME="dora" 

LEARNING_RATES=(1e-5 5e-5 1e-4 5e-4 1e-3)

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    echo "Starting fine-tuning with learning rate: $LEARNING_RATE"
    echo "Method: $METHOD_NAME"
    echo "=================================================="
    
    lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,peft="./trained-models/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}/checkpoint-2802" \
    --tasks gsm8k_cot_llama \
    --log_samples \
    --output_path ./result/math_gsm8k_rev/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}/checkpoint-2802  \
    --batch_size 32   --trust_remote_code --confirm_run_unsafe_code \
    --fewshot_as_multiturn --apply_chat_template
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed training with learning rate: $LEARNING_RATE"
    else
        echo "Training failed with learning rate: $LEARNING_RATE"
        echo "Continuing with next learning rate..."
    fi
    
    echo "=================================================="
    echo
done
echo "All fine-tuning runs completed!"

