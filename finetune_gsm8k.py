import os
import sys
from typing import List, Optional, Union

import fire
import torch
import transformers
from datasets import load_dataset
import torch.nn as nn
import numpy as np
import re

"""
Unused imports:
import bitsandbytes as bnb
"""
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.getcwd()+"/peft/src/")
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
print(sys.path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"

from peft import (  # noqa: E402
    LoraConfig,
    AdaLoraConfig,
    IA3Config,
    PromptTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    EvalPrediction,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)  # noqa: F402

import transformers
from peft.optimizers import create_loraplus_optimizer
from trl import SFTTrainer


    
def train(
        # model/data params
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        data_path: str = "gsm8k",
        output_dir: str = "./debug/test",
        method: str = 'ia3',  # Options: 'lora', 'adalora', 'ia3', 'prompt', 'loraplus'
        apply_peft: bool = True,  # Changed from apply_lora to apply_peft for more general naming
        subsample_split: float = 1,
        use_qloRA: bool = False,
        
        # monteclora hyperparams
        use_monteclora: bool = False,
        monteclora_at: str = "",
        monteclora_targets: List[str] = ["q_proj", "v_proj", "k_proj"],
        monteclora_n: int = 4,
        monteclora_m: int = 4,
        sample_scaler: float = 3e-3,
        use_entropy: bool = True,
        dirichlet_prior: int = 1,
        kl_loss_weight: float = 1e-3,
        mc_training: bool = True,
        posthoc_app: int = 0,
        use_dora: bool = False,
        
        # training hyperparams
        load_8bit: bool = False,
        batch_size: int = 8,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        num_epochs_coop: int = 0,
        learning_rate: float = 1e-5,
        learning_rate_coop: float = -1,
        cutoff_len: int = 256,
        val_set_size: int =120,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 233,
        save_step: int = 233,
        warmup_steps: int = 0,
        
        # lora hyperparams
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj"],
        target_modules: List[str] = ["q_proj", "v_proj", "k_proj"],
        
        # loraplus hyperparams
        loraplus_lr_ratio: float = 20,
        
        # adalora hyperparams
        target_r: int = 8,
        init_r: int = 12,
        tinit: int = 0,
        tfinal: int = 0,
        deltaT: int = 1,
        beta1: float = 0.85,
        beta2: float = 0.85,
        orth_reg_weight: float = 0.5,
        total_step: Optional[int] = None,
        rank_pattern: Optional[str] = None,
        # prompt tuning hyperparams
        num_virtual_tokens: int = 30,  # Added parameter for prompt tuning
        prompt_tuning_init_text: str = "",  # Added parameter for prompt tuning initialization text
        
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    num_epochs_std = num_epochs - num_epochs_coop
    
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"apply_peft: {apply_peft}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"num_epochs_coop: {num_epochs_coop}\n"
        f"num_epochs_std: {num_epochs_std}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"use_qloRA: {use_qloRA}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"monteclora_at: {monteclora_at}\n"
        f"posthoc_app: {posthoc_app}\n"
        f"number of experts: {monteclora_n}\n"
        f"use entropy loss: {use_entropy}\n"
        f"method: {method}\n"
        f"target_modules: {target_modules}\n"
        f"num_virtual_tokens: {num_virtual_tokens}\n"  # Added this line
    )
    
    if num_epochs_coop > 0:
        assert num_epochs > num_epochs_coop, "num_epochs must be greater than num_epochs_coop"
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='meta-llama/Llama-2-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    # tokenizer.padding_side = "left"  # Allow batched inference/

    print("Loading model...")
    if use_qloRA:
        # Use QLoRA for memory-efficient fine-tuning
        import bitsandbytes as bnb
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Prepare model for QLoRA training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    elif load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    # Load GSM8K dataset
    if "Instruct" in base_model :
        use_template = True
    else:
        use_template = False
    print("Loading GSM8K dataset...")
    if data_path == "gsm8k":
        data = load_dataset("gsm8k", "main")
    elif data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # Format data according to the model's chat template
    def format_gsm8k_with_apply_chat_template(example):
        question = example["question"].strip()
        answer = example["answer"].strip()

        q_template ='Given the following problem, reason and give a final answer to the problem.\nProblem: {quest}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
        cleanup_pattern = re.compile(r'\$?<<.*?>>')
        parts = answer.split("####")
        raw_reasoning = parts[0].strip()
        final_ans = parts[1].strip() if len(parts) > 1 else ""

        reasoning = cleanup_pattern.sub("", raw_reasoning)
        reasoning = reasoning.replace("\n", " ").strip()

        usr = (
            "Given the following problem, reason and give a final answer to the problem.\n"
            f"Problem: {question}\n"
            'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n\n'
            
        )
        asst = (f"{reasoning}"
                f"The final answer is {final_ans}\n")
        
        conversation = [
            # {"role": "system", "content": "You are a helpful assistant that's good at solving math problems step by step."},
            # {"role":"user", "content":""},
            {"role": "user", "content": usr},
            {"role": "assistant", "content": asst}
        ]
        
        # Apply the model's chat template
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": formatted_text}
    
    def format_gsm8k_without_template(example):
        question = example["question"].strip()
        answer = example["answer"].strip()

        q_template ='Given the following problem, reason and give a final answer to the problem.\nProblem: {quest}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
        cleanup_pattern = re.compile(r'\$?<<.*?>>')
        parts = answer.split("####")
        raw_reasoning = parts[0].strip()
        final_ans = parts[1].strip() if len(parts) > 1 else ""

        reasoning = cleanup_pattern.sub("", raw_reasoning)
        reasoning = reasoning.replace("\n", " ").strip()

        usr = (
            "Given the following problem, reason and give a final answer to the problem.\n"
            f"Problem: {question}\n"
            'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n\n'
            f"{reasoning}\n"
            f"The final answer is {final_ans}\n"
            
        )
        
        return {"text": usr}

    # Apply formatting to the dataset
    print("Formatting dataset with chat template...")
    if data_path == "gsm8k":
        if use_template:
            train_data = data["train"].map(format_gsm8k_with_apply_chat_template)
            if val_set_size > 0:
                val_data = data["test"].map(format_gsm8k_with_apply_chat_template)
            else:
                val_data = None
        else:
            train_data = data["train"].map(format_gsm8k_without_template)
            if val_set_size > 0:
                val_data = data["test"].map(format_gsm8k_without_template)
            else:
                val_data = None
    else:
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=6
            )
            # Subsample if requested
            sample_fraction = subsample_split
            train_data = train_val["train"].shuffle(seed=6).select(range(int(len(train_val["train"]) * sample_fraction)))
            val_data = train_val["test"].shuffle()
        else:
            # Subsample if requested
            sample_fraction = subsample_split
            train_data = data["train"].shuffle(seed=6).select(range(int(len(data["train"]) * sample_fraction)))
            val_data = None

    # Tokenize dataset
    print(train_data[0]['text'])
    print(tokenizer.padding_side)
    # tokenizer.padding_side = 'left'
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=cutoff_len,
        )

    print("Tokenizing dataset...")
    tokenized_train_data = train_data.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    if val_data:
        tokenized_val_data = val_data.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )
    else:
        tokenized_val_data = None

    if apply_peft:
        if method in ['adalora', 'AdaLoRA']: 
            config = AdaLoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                use_monteclora=use_monteclora,
                monteclora_at=monteclora_at,
                monteclora_targets=monteclora_targets,
                monteclora_n=monteclora_n,
                monteclora_m=monteclora_m,
                sample_scaler=sample_scaler,
                use_entropy=use_entropy,
                dirichlet_prior=dirichlet_prior,
                bias="none",
                task_type="CAUSAL_LM",
                kl_loss_weight=kl_loss_weight,
                mc_training=mc_training,
                use_dora=use_dora,
                # adalora specific
                target_r=target_r,
                init_r=init_r,
                tinit=tinit,
                tfinal=tfinal,
                deltaT=deltaT,
                beta1=beta1,
                beta2=beta2,
                orth_reg_weight=orth_reg_weight,
                total_step=total_step,
                rank_pattern=rank_pattern
            )
        elif method == 'ia3':
            config = IA3Config(
                target_modules=target_modules,
                feedforward_modules=target_modules,
                task_type="CAUSAL_LM",
            )
        elif method == 'prompt':  # Changed from 'prefix' to 'prompt'
            config = PromptTuningConfig(  # Changed from PrefixTuningConfig to PromptTuningConfig
                task_type="CAUSAL_LM",
                num_virtual_tokens=num_virtual_tokens,
                tokenizer_name_or_path=base_model,
                prompt_tuning_init=None if not prompt_tuning_init_text else "TEXT",
                prompt_tuning_init_text=prompt_tuning_init_text if prompt_tuning_init_text else None,
            )
           
        else:  # Default to LoRA
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                use_monteclora=use_monteclora,
                monteclora_at=monteclora_at,
                monteclora_targets=monteclora_targets,
                monteclora_n=monteclora_n,
                monteclora_m=monteclora_m,
                sample_scaler=sample_scaler,
                use_entropy=use_entropy,
                dirichlet_prior=dirichlet_prior,
                mc_training=mc_training,
                kl_loss_weight=kl_loss_weight,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=use_dora
            )
        
        print(config)
        model = get_peft_model(model, config)
        print(model)

    # Create loraplus optimizer if method is loraplus
    optimizer = None
    scheduler = None
    if method == 'loraplus':
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=learning_rate,
            loraplus_lr_ratio=loraplus_lr_ratio,
        )

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Run standard fine-tuning
    if num_epochs_std > 0:

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs_std,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=micro_batch_size,
            eval_strategy="steps" if val_set_size > 0 else "no",
            eval_steps=eval_step if val_set_size > 0 else None,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            save_steps=save_step,
            save_strategy="steps",
            fp16=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if use_wandb else "tensorboard",
            run_name=wandb_run_name if use_wandb else None,
            save_total_limit=30,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            seed=6,
            # optim="adamw_torch" if method != 'loraplus' else None,
        )

        # Setup the trainer with TRL's SFTTrainer for better handling of chat templates
        print(tokenizer.decode(tokenized_train_data[2]['input_ids']))
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_val_data,
            tokenizer=tokenizer,
            packing=False,
            dataset_text_field="text",
            max_seq_length=cutoff_len,
            optimizers=(optimizer, scheduler) if method == 'loraplus' else (None, None),
            # callbacks = callbacks
        )
        trainer.save_model("lmao")
        model.config.use_cache = False

        # old_state_dict = model.state_dict
        # model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(
        #         self, old_state_dict()
        #     )
        # ).__get__(model, type(model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        # Train the model
        print("Starting standard training...")
        print(vars(trainer))
        # trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save the model after standard training
        print("Saving model after standard training...")
        model.save_pretrained(f"{output_dir}/standard")

    # Run MonteCLoRA cooperative training phase if requested
    if num_epochs_coop > 0:
        print("Starting MonteCLoRA cooperative training phase...")
        # Enable MonteCLoRA training mode
        for module in list(dict(model.named_modules()).values()):
            if type(module).__name__ == 'MonteCLoRASampler':
                module.mc_training = True
        
        # Adjust learning rate for cooperative phase if needed
        if learning_rate_coop == -1:
            learning_rate *= 5
        else:
            learning_rate = learning_rate_coop
            
        # For post-hoc application, freeze all parameters except experts
        if posthoc_app:
            for n, p in model.named_parameters():
                if "expert_weights_prior" not in n and "std_prior" not in n:
                    p.requires_grad = False
        
        print(f"MonteCLoRA cooperative phase learning rate: {learning_rate}")
        
        coop_training_args = TrainingArguments(
            output_dir=f"{output_dir}/coop",
            num_train_epochs=num_epochs_coop,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=micro_batch_size,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            eval_steps=eval_step if val_set_size > 0 else None,
            logging_dir=f"{output_dir}/coop/logs",
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            save_steps=save_step,
            save_strategy="steps",
            fp16=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if use_wandb else "tensorboard",
            run_name=f"{wandb_run_name}_coop" if use_wandb else None,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            seed=6,
        )
        
        # Setup the trainer for cooperative phase
        coop_trainer = SFTTrainer(
            model=model,
            args=coop_training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_val_data,
            tokenizer=tokenizer,
            packing=False,
            dataset_text_field="text",
            max_seq_length=cutoff_len,
        )
        
        model.config.use_cache = False
        
        # Train the model for cooperative phase
        print("Starting cooperative training...")
        coop_trainer.train()
        
        # Save the final model
        print("Saving model after cooperative training...")
        model.save_pretrained(f"{output_dir}/coop_final")

    # Save the final model
    print("Saving final model...")
    model.save_pretrained(f"{output_dir}/final")

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    print("Training complete!")


if __name__ == "__main__":
    fire.Fire(train)


# CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --model_args pretrained=/home/models/Meta-Llama-3.2-3B-Instruct,peft=/home/vaibhav/MonteCLoRA/llama32_gsm8k_finetuned_template_lora_test/checkpoint-50 --tasks gsm8k_cot_llama --log_samples --output_path ./llama_lora_gsm8k  --batch_size 64   --trust_remote_code --confirm_run_unsafe_code --limit 64 --fewshot_as_multiturn --apply_chat_template
# CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --model_args pretrained=/home/models/Meta-Llama-3.2-3B-Instruct --tasks gsm8k_cot_llama --log_samples --output_path ./llama_lora_gsm8k  --batch_size 64   --trust_remote_code --confirm_run_unsafe_code --limit 64 --fewshot_as_multiturn --apply_chat_template