  CUDA_VISIBLE_DEVICES=3 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'ft-training_set/math_10k.json' \
  --output_dir './final_trained_models/meth' \
  --batch_size 16 \
  --var_loss_scale 5e-3 \
  --kl_loss_weight 1e-5 \
  --use_entropy True \
  --micro_batch_size 16 \
  --num_epochs 3 \
  --num_epochs_coop 3 \
  --lora_r 32 \
  --learning_rate 1e-4 \
  --learning_rate_coop 1e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --eval_step 1000 \
  --save_step 1000 \
  --lora_alpha 64 \
  --posthoc_app 0 \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --lora_cooperative_at 'lora_A' \
  --cooperative_targets '["q_proj", "k_proj", "v_proj"]' \
  --wandb_project 'llama-coop-test' \
  --wandb_run_name 'cs/llama/qkv/lora_A/1e-4' \


CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
--model LLaMA-7B \
--adapter LoRA \
--dataset openbookqa \
--base_model 'yahma/llama-7b-hf' \
--lora_weights './final_trained_models/cs/llama/qkv/lora_A/1e-4/3_epochs/True/checkpoint-53260' \
--batch_size 1