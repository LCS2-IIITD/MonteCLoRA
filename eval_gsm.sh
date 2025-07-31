METHOD_NAME="monteclora" 

LEARNING_RATES=(1e-5 5e-4 3e-3 5e-3 7e-3)

export CUDA_VISIBLE_DEVICES=0
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    echo "Starting fine-tuning with learning rate: $LEARNING_RATE"
    echo "Method: $METHOD_NAME"
    echo "=================================================="
    
    lm_eval \
    --model hf \
    --model_args pretrained=/home/models/Meta-Llama-3.2-3B-Instruct,peft="./trained-models/math/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}/checkpoint-2802" \
    --tasks gsm8k_cot_llama \
    --log_samples \
    --output_path ./result/math/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}  \
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