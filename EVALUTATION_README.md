##Installing lm_eval

```bash
git clonee https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```
## Running evaluation on gsm8k 

```bash
CUDA_VISIBLE_DEVICES=2 lm_eval --model hf --model_args pretrained=/home/models/Meta-Llama-3.2-3B-Instruct,peft=/home/vaibhav/MonteCLoRA/llama32_gsm8k_finetuned_template_ia3/checkpoint-934 --tasks gsm8k_cot_llama --log_samples --output_path ./llama_lora_gsm8k  --batch_size 64   --trust_remote_code --confirm_run_unsafe_code --limit 64 --fewshot_as_multiturn --apply_chat_template
```