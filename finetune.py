import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
import torch.nn as nn

"""
Unused imports:
import bitsandbytes as bnb
"""
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.getcwd()+"/peft/src/")
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
print(sys.path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import (  # noqa: E402
    LoraConfig,
    AdaLoraConfig,
    # BottleneckConfig,
    # PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, EvalPrediction, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402
from peft.optimizers import create_loraplus_optimizer

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        method: str = 'lora',
        apply_lora: bool = False,
        subsample_split: float = 1.0,

        #monteclora hyperparams
        use_monteclora: bool = False,
        monteclora_at: str = " ",
        monteclora_targets: List[str] = [],
        monteclora_n: int = 4,
        monteclora_m: int = 4,
        # sample_period: int = 1,
        sample_scaler: float = 5e-3,
        use_entropy: bool = False,
        dirichlet_prior:int = 1,
        # use_averaging: bool = False,
        kl_loss_weight: float = 1,
        mc_training : bool = True,
        # training hyperparams
        load_8bit : bool = False,
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        num_epochs_coop: int = 0,
        learning_rate: float = 3e-4,
        learning_rate_coop: float = -1,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        #loraplus lr ratio
        loraplus_lr_ratio : float = 20,
        #dora
        use_dora : bool = False,
        # adalora hyperparams
        target_r=8,
        init_r=12,
        tinit=0,
        tfinal=0,
        deltaT=1,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.5,
        total_step=None,
        rank_pattern=None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        posthoc_app = 0,
):
    num_epochs_std = num_epochs - num_epochs_coop
    print(
        # update acc to new names
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"apply_lora: {apply_lora}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"num_epochs_coop: {num_epochs_coop}\n"
        f"num_epochs_std: {num_epochs_std}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"monteclora_at: {monteclora_at}\n"
        f"posthoc_app: {posthoc_app}\n"
        f"number of experts: {monteclora_n}\n"
        f"use entropy loss: {use_entropy}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"method: {method}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"monteclora_targets: {monteclora_targets}\n"

    )

    assert {
        num_epochs > num_epochs_coop
    }, "num_epochs must be greater than num_epochs_coop"
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
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

    if load_8bit:
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
    if model.config.model_type == "llama" and not "Llama-3" in base_model or "Llama-2" in base_model :
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    print(model)

    for name, param in model._modules["lm_head"].named_parameters():
        print(name, param.requires_grad)
    if apply_lora:
        if method in ['adalora', 'AdaLoRA']: 
            config = AdaLoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                use_monteclora=use_monteclora,
                monteclora_at=monteclora_at,
                monteclora_targets = monteclora_targets,
                monteclora_n=monteclora_n,
                monteclora_m = monteclora_m,
                sample_scaler = sample_scaler,
                use_entropy = use_entropy,
                dirichlet_prior=dirichlet_prior,
                bias="none",
                task_type="CAUSAL_LM",
                kl_loss_weight=kl_loss_weight,
                mc_training=mc_training,

                # dora
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
        else:

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                use_monteclora=use_monteclora,
                monteclora_at=monteclora_at,
                monteclora_targets = monteclora_targets,
                monteclora_n=monteclora_n,
                monteclora_m = monteclora_m,
                sample_scaler = sample_scaler,
                use_entropy = use_entropy,
                dirichlet_prior=dirichlet_prior,
                bias="none",
                task_type="CAUSAL_LM",
                kl_loss_weight=kl_loss_weight,
                mc_training=mc_training,
                # dora
                use_dora=use_dora
            )
    # print(config)
    model = get_peft_model(model, config)
    print(model)


    #create loraplus optimizer if method is loraplus
    optimizer = None
    scheduler = None
    if method == 'loraplus':
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=learning_rate,
            loraplus_lr_ratio=loraplus_lr_ratio,
        )
        scheduler = None

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

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

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    print(model)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=6
        )

        # Subsample 10% of the data for quick tuning
        sample_fraction = subsample_split
        train_data = train_val["train"].shuffle(seed=6).select(range(int(len(train_val["train"]) * sample_fraction)))
        train_data = train_data.map(generate_and_tokenize_prompt)

        # train_data = (
        #     train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        # )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        # Subsample 10% of the data for quick tuning
        sample_fraction = subsample_split
        train_data = train_val["train"].shuffle(seed=6).select(range(int(len(data["train"]) * sample_fraction)))
        train_data = train_data.map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    eps = learning_rate*1e-3

    assert(
        num_epochs_std + num_epochs_coop == num_epochs
    ), "num_epochs_std + num_epochs_coop must equal num_epochs"
    
    if num_epochs_std:
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs_std,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                eval_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=eval_step if val_set_size > 0 else None,
                save_steps=save_step,
                output_dir=output_dir,
                save_total_limit=10,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else "tensorboard",
                run_name=wandb_run_name if use_wandb else None,
                seed=6,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        model.config.use_cache = False

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        model.save_pretrained(output_dir)

        print(
            "\n If there's a warning about missing keys above, please disregard :)"
        )

    if num_epochs_coop:
        print(train_data.__len__())
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=6
            )
            train_data = (
                train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = None
        print("MonteCLoRA RUN")
        for module in list(dict(model.named_modules()).values()):
            if type(module).__name__ == 'MonteCLoRASampler':
                module.mc_training = True
                # module.initialize_prior_fine()
        if learning_rate_coop == -1:
                learning_rate *= 5
        else:
            learning_rate = learning_rate_coop

        if posthoc_app :
            for n, p in model.named_parameters():
                if "expert_weights_prior" not in n and "std_prior" not in n:
                    p.requires_grad = False
        print("Learning rate: ", learning_rate)
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs_coop,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=eval_step if val_set_size > 0 else None,
                save_steps=save_step,
                output_dir=output_dir,
                save_total_limit=10,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else "tensorboard",
                run_name=wandb_run_name if use_wandb else None,
                seed=6,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        model.config.use_cache = False

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        model.save_pretrained(output_dir)

        print(
            "\n If there's a warning about missing keys above, please disregard :)"
        )



def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)