#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on SuperGLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import math
import logging
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["WANDB_PROJECT"] = "PEFT-SuperGLUE"
import random
import sys
import json 
import torch
from dataclasses import dataclass, field
from typing import Optional
import evaluate
import numpy as np
from datasets import load_dataset, load_metric
from superglue_dataprocessing import InputExample, convert_examples_to_features

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from loralib import RankAllocator 

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "copa": ("premise", "choice1", "choice2", "question"),
    "wic": ("sentence1", "sentence2", "word"),
    "wsc": ("text", "span1_text", "span2_text"),
    "cb": ("premise", "hypothesis"),
    "boolq": ("question", "passage"),
    "rte": ("premise", "hypothesis"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a SuperGLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    budget: int = field(
        default=1000000,
        metadata={"help": "Number of parameters to update"},
    )
    lora_type: Optional[str] = field(
        default="calra",
        metadata={"help": "The lora type: frd or svd or calra"},
    )
    lora_module: Optional[str] = field(
        default="query,value",
        metadata={"help": "The modules applying lora: query,key,value,intermediate,layer.output,attention.output"},
    )
    sparseft_module: Optional[str] = field(
        default="query,value",
        metadata={"help": "The modules applying sparseft: query,key,value,intermediate,layer.output,attention.output"},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    num_experts : Optional[int] = field(
        default=1,
        metadata={"help": "number of experts to use for CALRA"}
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default='houlsby',
        metadata={"help": "houlsby or pfeiffer"},
    )
    sparseft_type: Optional[int] = field(
        default=0,
        metadata={"help": "type of sparseft to use, supported are 0, 1 and 2"}
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    reg_orth_coef: Optional[float] = field(
        default=0.0,
        metadata={"help": "Orthogonal regularization coefficient"},
    )
    reg_sparse_coef: Optional[float] = field(
        default=0.0,
        metadata={"help": "Sparse regularization coefficient"},
    )
    sparsity : Optional[float] = field(
        default=2e-3,
        metadata={"help":"Fraction of parameters to be kept trainable, represents sparsity"}
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    apply_adalora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply rank selector or not."},
    )
    apply_sparseft: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply sparseft method or not."},
    )
    target_rank: Optional[int] = field(
        default=16,
        metadata={"help": "Average target rank."},
    )
    target_total_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Specifying target number of total singular values"},
    )
    init_warmup: Optional[int] = field(
        default=4500,
        metadata={"help": "Total steps of inital warmup"},
    )
    final_warmup: Optional[int] = field(
        default=12000,
        metadata={"help": "Total steps of final fine-tuning"},
    )
    mask_interval: Optional[int] = field(
        default=10,
        metadata={"help": "Masking interval"},
    )
    beta1: Optional[float] = field(
        default=0.85,
        metadata={"help": "The coefficient of EMA"},
    )
    beta2: Optional[float] = field(
        default=0.85,
        metadata={"help": "The coefficient of EMA"},
    )
    tb_writter_loginterval: Optional[int] = field(
        default=500,
        metadata={"help": "The logging interval for tb_writter."},
    )

    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    # logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

    # Setup output dir 
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.output_dir = os.path.join(training_args.output_dir, "model")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.logging_dir = os.path.join(training_args.output_dir, "log")
    os.makedirs(training_args.logging_dir, exist_ok=True)
    #training_args.run_name = training_args.output_dir 

    if "debug" in training_args.output_dir:
        import ipdb
        ipdb.set_trace()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    

    # Setup logging
    logging.basicConfig(
        filename= os.path.join(training_args.output_dir, 'log.txt'), filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN, 
        # handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.info(training_args.output_dir)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set tb_writter 
    if is_main_process(training_args.local_rank):
        tb_writter = SummaryWriter(log_dir=training_args.logging_dir)
    else:
        tb_writter = None

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a SuperGLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("super_glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a SuperGLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a SuperGLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        apply_lora=model_args.apply_lora,
        lora_type=model_args.lora_type, 
        lora_module=model_args.lora_module, 
        sparseft_module=model_args.sparseft_module, 
        sparseft_type=model_args.sparseft_type, 
        lora_alpha=model_args.lora_alpha,
        lora_r=model_args.lora_r,
        apply_adapter=model_args.apply_adapter,
        apply_sparseft=model_args.apply_sparseft,
        adapter_type=model_args.adapter_type,
        adapter_size=model_args.adapter_size,
        reg_loss_wgt=model_args.reg_loss_wgt,
        reg_sparse_coef=model_args.reg_sparse_coef,
        masking_prob=model_args.masking_prob,
        budget=model_args.budget,
    )
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    trainable_params = []
    if model_args.apply_lora:
        if model_args.lora_path is not None:
            lora_state_dict = torch.load(model_args.lora_path)
            logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
            logger.info(lora_state_dict.keys())
            model.load_state_dict(lora_state_dict, strict=False)
   
        trainable_params.append('lora')
        trainable_params.append('resweight')
    elif model_args.apply_sparseft:
        if model_args.sparseft_type == 0:
            trainable_params.append('delta_weight')
            trainable_params.append('mask_input')
            trainable_params.append('mask_output')
        elif model_args.sparseft_type == 1:
            trainable_params.append('delta_weight')
            trainable_params.append('mask_weight')
        elif model_args.sparseft_type == 2:
            trainable_params.append('delta_weight')
        else:
            raise NotImplementedError

    if model_args.apply_adapter:
        if model_args.adapter_path is not None:
            adapter_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_adapter.bin'))
            head_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_model_head.bin'))
            added_state_dict = {}
            for k, v in adapter_state_dict.items():
                new_k = k.replace(data_args.task_name + '.', '').replace('adapter_down.0.', 'adapter_A.').replace('adapter_up.', 'adapter_B.').replace('.adapters.', '.adapter.')
                added_state_dict[new_k] = v
            for k, v in head_state_dict.items():
                new_k = k.replace('heads.' + data_args.task_name + '.1', 'classifier.dense').replace('heads.' + data_args.task_name + '.4', 'classifier.out_proj')
                added_state_dict[new_k] = v
            logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
            logger.info(added_state_dict.keys())
            missing_keys, unexpected_keys = model.load_state_dict(added_state_dict, strict=False)
            for missing_key in missing_keys:
                assert 'adapter' not in missing_key, missing_key + ' is missed in the model'
            assert len(unexpected_keys) == 0, 'Unexpected keys ' + str(unexpected_keys)
        trainable_params.append('adapter')

    if model_args.apply_bitfit:
        trainable_params.append('bias')
    num_param = 0
    # if no trainable_params then perform full finetuning
    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            if name.startswith('deberta') or name.startswith('roberta'):
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name and 'lora_mask' not in name:
                        param.requires_grad = True
                        sub_num_param = 1
                        for dim in param.shape: 
                            sub_num_param *= dim  
                        num_param += sub_num_param 
                        break
            else:
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            sub_num_param = 1
            for dim in param.shape:
                sub_num_param *= dim  
            num_param += sub_num_param
            param.requires_grad = True
    
    num_train_param_correction = 0
    if model_args.apply_sparseft:
        if model_args.sparseft_type == 0 and training_args.inp_out_mask_path:
            inp_out_mask = torch.load(training_args.inp_out_mask_path)
            model_named_dict = dict(model.named_parameters())
            for name, mask_dict in inp_out_mask.items():
                delta_weight_name = name+'.delta_weight'
                delta_weight = model_named_dict[delta_weight_name]
                
                mask_input_name = name+'.mask_input'
                prev_mask_input = mask_dict['mask_input']
                mask_input = model_named_dict[mask_input_name]
                num_train_param_correction -= mask_input.numel()
                
                k_in = int(0.01*mask_input.numel())
                inp_sel = torch.topk(prev_mask_input, k=k_in).indices
                new_mask_input = torch.zeros_like(mask_input)
                sig = torch.sigmoid(prev_mask_input)
                new_mask_input[inp_sel] = 1.0
                # new_mask_input[sig>=min(sig.max(), 0.01)] = 1.0
                # k_in = new_mask_input.count_nonzero()
                new_mask_input.requires_grad = False
                new_mask_input = new_mask_input.to(torch.bool)
                mask_input.data = new_mask_input
                mask_input.requires_grad = False
                num_train_param_correction += mask_input.numel()
                
                mask_output_name = name+'.mask_output'
                prev_mask_output = mask_dict['mask_output']
                mask_output = model_named_dict[mask_output_name]
                num_train_param_correction -= mask_output.numel()
                
                k_out = int(0.01*mask_output.numel())
                out_sel = torch.topk(prev_mask_output, k=k_out).indices
                new_mask_output = torch.zeros_like(mask_output)
                new_mask_output[out_sel] = 1.0
                # sig = torch.sigmoid(prev_mask_output)
                # new_mask_output[sig>=min(sig.max(), 0.01)] = 1.0
                k_out = new_mask_output.count_nonzero()
                new_mask_output.requires_grad = False
                new_mask_output = new_mask_output.to(torch.bool)
                mask_output.data = new_mask_output
                mask_output.requires_grad = False
                num_train_param_correction += mask_output.numel()
                
                num_train_param_correction -= delta_weight.numel()
                new_delta_weight = torch.zeros(size=(k_in, k_out), requires_grad=True)
                delta_weight.data = new_delta_weight
                num_train_param_correction += delta_weight.numel()
                
        if model_args.sparseft_type == 1 and training_args.wt_mask_path:
            
            wt_mask = torch.load(training_args.wt_mask_path)
            model_named_dict = dict(model.named_parameters())
            rec = []
            for name, mask_dict in wt_mask.items():
                prev_mask_weight = mask_dict['mask_weight']
                rec.append(prev_mask_weight.flatten())
            rec = torch.cat(rec, dim=0)
            total = rec.numel()
            print(model_args.sparsity)
            chosen = int(total*model_args.sparsity)
            thresh = torch.topk(rec, chosen).values[-1]
            for name, mask_dict in wt_mask.items():
                    prev_mask_weight = mask_dict['mask_weight']
                    
                    mask_weight_name = name+'.mask_weight'    
                    mask_weight = model_named_dict[mask_weight_name]
                    
                    new_mask_weight = torch.zeros_like(mask_weight)
                    new_mask_weight[prev_mask_weight>thresh]= 1.0
                    new_mask_weight.requires_grad = False
                    mask_weight.data = new_mask_weight
                    mask_weight.requires_grad = False
            num_train_param_correction += chosen-2*total
    # correct num_param
    num_param += num_train_param_correction
    logger.info("Number of Trainable Parameters: %d"%(int(num_param))) 
    if tb_writter is not None: 
        tb_writter.add_scalar("train/num_train_param", num_param, 0)   

    # Preprocessing the datasets
    if data_args.task_name is not None:
        keys_ = task_to_keys[data_args.task_name]
    else:
        raise ValueError("Check the task name")
    
    if len(keys_) == 2:
        sentence_key1, sentence_key2 = keys_
        sentence_key3 = None
    elif len(keys_) == 3:
        sentence_key1, sentence_key2, sentence_key3 = keys_
        sentence_key4 = None
    elif len(keys_) == 4:
        sentence_key1, sentence_key2, sentence_key3, sentence_key4 = keys_

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        if sentence_key2 and sentence_key3 is None:
            example = InputExample(examples[sentence_key1], examples[sentence_key2])
        elif sentence_key2 and sentence_key3 and sentence_key4 is None:
            example = InputExample(examples[sentence_key1], examples[sentence_key2], examples[sentence_key3])
        elif sentence_key2 and sentence_key3 and sentence_key4:
            example = InputExample(examples[sentence_key1], examples[sentence_key2], examples[sentence_key3], examples[sentence_key4])
        else:
            example = InputExample(examples[sentence_key1])

        # Tokenize the texts
        
        if 'roberta' in model_args.model_name_or_path.lower():
            cls_token = '<s>'
            sep_token = '</s>'
            pad_token = 1
            sep_token_extra = True
        elif 'bert' in model_args.model_name_or_path.lower():
            cls_token = '[CLS]'
            sep_token = '[SEP]'
            pad_token = 0
            sep_token_extra = False

        if data_args.task_name == 'copa':
            result = convert_examples_to_features(example, max_seq_length, tokenizer, sequence_a_segment_id=0, 
                                 sequence_b_segment_id=0,
                                 sequence_c_segment_id=0,
                                 sequence_d_segment_id=1,
                                 cls_token=cls_token,
                                 sep_token=sep_token,
                                 pad_token=pad_token,
                                 sep_token_extra=sep_token_extra)
        elif data_args.task_name == 'wic':
            result = convert_examples_to_features(example, max_seq_length, tokenizer, sequence_a_segment_id=0, 
                                 sequence_b_segment_id=0,
                                 sequence_c_segment_id=1,
                                 cls_token=cls_token,
                                 sep_token=sep_token,
                                 pad_token=pad_token,
                                 sep_token_extra=sep_token_extra)
        elif data_args.task_name == 'wsc':
            result = convert_examples_to_features(example, max_seq_length, tokenizer, sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 sequence_c_segment_id=1,
                                 cls_token=cls_token,
                                 sep_token=sep_token,
                                 pad_token=pad_token,
                                 sep_token_extra=sep_token_extra)
        else:
            result = convert_examples_to_features(example, max_seq_length, tokenizer, sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 cls_token=cls_token,
                                 sep_token=sep_token,
                                 pad_token=pad_token,
                                 sep_token_extra=sep_token_extra)
            
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]

        if 'roberta' in model_args.model_name_or_path.lower():
            result.pop('token_type_ids')

        return result

    #for key in datasets:
    #    datasets[key] = datasets[key].map(preprocess_function, batched=True)
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=False)
    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("super_glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None


    # Initialize the rankallocator
    if (model_args.lora_type == "svd" or model_args.lora_type == 'calra') and model_args.apply_adalora:
        rankallocator = RankAllocator(
            model, 
            lora_r=model_args.lora_r,
            lora_type=model_args.lora_type,
            target_rank=model_args.target_rank,
            init_warmup=model_args.init_warmup, 
            final_warmup=model_args.final_warmup,
            mask_interval=model_args.mask_interval, 
            beta1=model_args.beta1, 
            beta2=model_args.beta2, 
            target_total_rank=model_args.target_total_rank, 
            tb_writter=tb_writter, 
            tb_writter_loginterval=model_args.tb_writter_loginterval,
        )
    else:
        rankallocator = None

    # Initialize our Trainer
    # for _, p in model.named_parameters():
    #     print(p.requires_grad)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_args=model_args, 
        tb_writter=tb_writter, 
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            for key in metrics:
                if tb_writter:
                    tb_writter.add_scalar("Eval_%s/%s"%(task, key), metrics[key], training_args.num_train_epochs)
                logger.info("{task} {key}: {value}:".format(task=task, key=key, value=metrics[key]))

            trainer.log_metrics("Eval_%s"%task, metrics)
            trainer.save_metrics("Eval_%s"%task, metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    if tb_writter is not None:
        tb_writter.close() 

    # Below code block was giving an error when run with a rank allocator, need to resolve
    # if rankallocator is not None and is_main_process(training_args.local_rank):
    #     rank_pattern = rankallocator.get_rank_pattern()
    #     with open(os.path.join(training_args.output_dir, "rank_pattern.json"), "w") as f:
    #         json.dump(rank_pattern, f) 
    
    # if apply_sparseft is True then dump mask_input and mask_output for all parameters to a file
    if (model_args.apply_sparseft):
        # this means currently a continuous relaxation has been learnt
        if  (model_args.sparseft_type == 0 and training_args.inp_out_mask_path is None):
            # so write to a file storing the inp_out_mask of all parameters
            inp_out_mask = dict()
            for name, param in model.named_parameters():
                name_ = None
                if ('mask_input' in name):
                    type = 'mask_input'
                    name_ = name[:-11]
                elif ('mask_output' in name):
                    type = 'mask_output'
                    name_ = name[:-12]
                if (name_):
                    if name_ not in inp_out_mask:
                        inp_out_mask[name_] = dict()
                    inp_out_mask[name_][type] = param
            torch.save(inp_out_mask, 'inp_out_mask_'+data_args.task_name+'.pth')
        if  (model_args.sparseft_type == 1 and training_args.wt_mask_path is None):
            # so write to a file storing the wt_mask of all parameters
            wt_mask = dict()
            for name, param in model.named_parameters():
                name_ = None
                if ('mask_weight' in name):
                    name_ = name[:-12]
                    type = 'mask_weight'
                if (name_):
                    if name_ not in wt_mask:
                        wt_mask[name_] = dict()
                    wt_mask[name_][type] = param
            torch.save(wt_mask, 'wt_mask_'+data_args.task_name+'.pth')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
