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
PROMPT_TEMPLATE='''You are exceptionally skilled at crafting high-quality programming problems and
offering precise solutions.

### Problem
{problem}

### Solution
{solution}
'''


from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import shutil


def eval(
        tokenizer,
        model,
        base_model_path,
):
    # Save the PEFT adapter and tokenizer
    model.save_pretrained("adapter_model")
    tokenizer.save_pretrained("adapter_model")

    # Get the base model path from the PEFT model
    base_model_path = base_model_path

    # Reload base model
    reloaded_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load the PEFT adapter onto the base model
    from peft import PeftModel
    reloaded_model = PeftModel.from_pretrained(
        reloaded_base_model,
        "adapter_model"
    )

def eval(
        tokenizer,
        model
):
    model = model.merge_and_unload()
    model.save_pretrained("merged_model")
    tokenizer.save_pretrained("merged_model")

    eval_model = HFLM(
    pretrained="merged_model",
    tokenizer="merged_model",
    dtype="auto",
    batch_size=1
    )

    # Evaluate on GSM8K
    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=["gsm8k_cot_llama"],
        limit = 64,
        batch_size = 4,
        apply_chat_template = True,
        fewshot_as_multiturn = True,
    )
    print(results.keys())
    print(results["results"])

    import pickle
    with open("results.pickle", "wb") as f:
        pickle.dump(results, f)
    shutil.rmtree("merged_models")

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
from peft.optimizers import create_loraplus_optimizer
from trl import SFTTrainer

def train(
        # model/data params
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        data_path: str = "./ft-training_set/code_python_10k.json",
        output_dir: str = "./debug",
        method: str = 'monteclora',  # Options: 'lora', 'adalora', 'ia3', 'prompt', 'loraplus'
        apply_peft: bool = True,  # Changed from apply_lora to apply_peft for more general naming
        subsample_split: float = 1.0,
        use_qloRA: bool = False,
        
        # monteclora hyperparams
        use_monteclora: bool = False,
        monteclora_at: str = "",
        monteclora_targets: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        monteclora_n: int = 4,
        monteclora_m: int = 4,
        sample_scaler: float = 2.5e-5,
        use_entropy: bool = True,
        dirichlet_prior: int = 1,
        kl_loss_weight: float = 1e-5,
        mc_training: bool = True,
        posthoc_app: int = 0,
        use_dora: bool = False,
        
        # training hyperparams
        load_8bit: bool = False,
        batch_size: int = 8,
        micro_batch_size: int = 4,
        num_epochs: int = 1,
        num_epochs_coop: int = 0,
        learning_rate: float = 2e-5,
        learning_rate_coop: float = -1,
        cutoff_len: int = 1024,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        warmup_steps: int = 100,
        
        # lora hyperparams
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "up_proj", "down_proj"],
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
        num_virtual_tokens: int = 10,  # Added parameter for prompt tuning
        prompt_tuning_init_text: str = "",  # Added parameter for prompt tuning initialization text
        
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        
        # wandb params
        wandb_project: str = "MonteCLoRA-code",
        wandb_run_name: str = "prompt-3e-1",
        wandb_watch: str = "false",  # options: false | gradients | all
        wandb_log_model: str = "false",  # options: false | true
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
    tokenizer.padding_side = "left"  # Allow batched inference

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
    print("Loading GSM8K dataset...")
    if data_path == "gsm8k":
        data = load_dataset("gsm8k", "main")
    elif data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    print(data)
    # Format data according to the model's chat template
    def format_code_with_apply_chat_template(example):
       
        return  {'text':PROMPT_TEMPLATE.format(problem = example['problem'], solution = example['solution'])}


    # Apply formatting to the dataset
    print("Formatting dataset with chat template...")
    # if data_path == "gsm8k":
    #     train_data = data["train"].map(format_code_with_apply_chat_template)
    #     if val_set_size > 0:
    #         val_data = data["test"].map(format_code_with_apply_chat_template)
    #     else:
    #         val_data = None
    # else:
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

    train_data = train_data.map(format_code_with_apply_chat_template)
    if val_data is not None:
        val_data = val_data.map(format_code_with_apply_chat_template)
    # Tokenize dataset
    # print(train_data[0])


    # statistics for train_data
    tokenized_lengths = []
    for data in train_data:
        tokenized_lengths.append(len(tokenizer.encode(data["text"])))
    
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
    
    print(tokenizer.decode(tokenized_train_data[1]['input_ids']))
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
                bias="none",
                task_type="CAUSAL_LM",
                kl_loss_weight=kl_loss_weight,
                mc_training=mc_training,
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
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs_std,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=micro_batch_size * gradient_accumulation_steps,
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
            save_total_limit=20,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            seed=6,
            # optim="adamw_torch" if method != 'loraplus' else None,
        )

        # Setup the trainer with TRL's SFTTrainer for better handling of chat templates
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
        )

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
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print(model)
        eval(tokenizer, model)
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
        
        # Training arguments for cooperative phase
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

    print("Saving final model...")
    model.save_pretrained(f"{output_dir}/final")
        


    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    print("Training complete!")


if __name__ == "__main__":
    fire.Fire(train)
