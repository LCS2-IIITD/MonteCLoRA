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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import math
import logging
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["WANDB_PROJECT"] = "PEFT-GLUE"
import random
import sys
import json 
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Union
import evaluate
import numpy as np
from datasets import load_dataset
from datasets import DatasetDict, Dataset
import warnings
warnings.filterwarnings("ignore")
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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from calflops import calculate_flops

from peft import (  # noqa: E402
    LoraConfig,
    AdaLoraConfig,
    # BottleneckConfig,
    # PrefixTuningConfig,
    get_peft_model,
    # get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    # set_peft_model_state_dict,
)

from peft.optimizers import create_loraplus_optimizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "wic": ("sentence1", "sentence2"),  #SuperGLUE tasks below
    "boolq": ("passage", "question"),   
    "cb": ("premise","hypothesis"),
    "axg": ("premise","hypothesis"),
    "axb": ("sentence1","sentence2"),
    "copa": ("premise","choice1","choice2","question"), #AdvGLUE tasks below
    "adv_mnli": ("premise","hypothesis"),
    "adv_qnli": ("question","sentence"),
    "adv_qqp" : ("question1","question2"),
    "adv_rte" : ("sentence1","sentence2"),
    "adv_sst2" : ("sentence", None)
}

logger = logging.getLogger(__name__)

def set_seed(seed, use_det = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = use_det
    torch.backends.cudnn.benchmark = False

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
    test_adv_glue: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use adversarial GLUE test set or not."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    num_epochs_coop : Optional[int] = field(
        default=3,
        metadata={"help": "num_epochs_coop"}
    )

    # learning_rate : Optional[float] = field(
    #     default=1e-4,
    #     metadata={"help": "learning_rate"}
    # )

    learning_rate_coop : Optional[float] = field(
        default=1e-4,
        metadata={"help": "learning_rate_coop"}
    )

    loraplus_lr_ratio : Optional[float] = field(
        default=20,
        metadata={"help": "ηB/ηA where ηA = learning_rate"}
    )

    wandb_project : Optional[str] = field(
        default="",
        metadata={"help": "WandB project name"}
    )

    wandb_watch : Optional[str] = field(
        default="",
        metadata={"help": "WandB watch"}
    )

    wandb_log_model : Optional[str] = field(
        default="",
        metadata={"help": "WandB log model"}
    )

    run_tag : Optional[str] = field(
        default="",
        metadata={"help": "WandB run tag"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            print(self.validation_file)
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

    method: str = field(
        metadata={"help": "PEFT method, choose from MonteCLoRA, LoRA, LoRA+, AdaLoRA, DoRA"}
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
        default=True,
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
    freeze_base: Optional[bool] = field(
        default=False,
        metadata={"help": "To freeze the weights of the base model"},
    )

    use_det: bool = field(
        default=False,
        metadata={
            "help": (
                "use_det"
            )
        },
    )

    monteclora_at: Optional[str] = field(
        default="lora_A",
        metadata={"help": "MonteCLoRA for lora_A / B"},
    )

    monteclora_targets: Optional[Union[List[str], str]] = field(
        default="query",
        metadata={
            "help": "List of module names or regex expression of the module names to replace with mcLora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    posthoc_app: Optional[int] = field(
        default=0,
        metadata={"help": "Whether to apply posthoc or not."},
    )

    sparseft_module: Optional[str] = field(
        default="query,value",
        metadata={"help": "The modules applying sparseft: query,key,value,intermediate,layer.output,attention.output"},
    )
    
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_module: Optional[str] = field(
        default="query,key,value,intermediate,layer.output,attention.output",
        metadata={"help": "The modules applying lora: query,key,value,intermediate,layer.output,attention.output"},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )

    monteclora_n : Optional[int] = field(
        default=4,
        metadata={"help": "number of samples from each gaussian "}
    )

    monteclora_m : Optional[int] = field(
        default=4,
        metadata={"help": "number of gaussians"}
    )

    sample_period : Optional[int] = field(
        default=1,
        metadata={"help": "number of sampling steps"}
    )

    sample_scaler : Optional[float] = field(
        default=5e-3,
        metadata={"help": "sample scaler"}
    )

    # learning_rate : Optional[float] = field(
    #     default=1e-4,
    #     metadata={"help": "learning_rate"}
    # )


    mc_training : Optional[bool] = field(
        default=True,
        metadata={"help": "mc_training"}
    )

    use_monteclora : Optional[bool] = field(
        default=True,
        metadata={"help": "use_monteclora"}
    )

    var_loss_scale : Optional[float] = field(
        default=1e-2,
        metadata={"help": "scaling constant for variance sampling"}
    )
    use_entropy : Optional[bool] = field(
        default=False,
        metadata={"help": "Use Entropy Loss or Not"}
    )

    use_averaging : Optional[bool] = field(
        default=False,
        metadata={"help": "Use averaring of sampled mus"}
    )

    averaging_factor: Optional[float] = field(
        default=0.1,
        metadata={"help": "Exponential averaging factor"}
    )

    kl_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Factor to multiply the KL divergence loss"}
    )

    dirichlet_prior: Optional[float] = field(
        default=1,
        metadata={"help": "Dirichlet prior"}
    )

    # adalora parameters

    target_r: Optional[int] = field(
        default=8, # set to default adalora
        metadata={"help": "Target Lora matrix dimension."}
    )

    init_r: Optional[int] = field(
        default=12, # set to default adalora
        metadata={"help": "Initial Lora matrix dimension."}
    )

    tinit: Optional[int] = field(
        default=0, # set to default adalora
        metadata={"help": "The steps of initial warmup."}
    )

    tfinal: Optional[int] = field(
        default=0, # set to default adalora
        metadata={"help": "The steps of final warmup."}
    )

    deltaT: Optional[int] = field(
        default=1, # set to default adalora
        metadata={"help": "Step interval of rank allocation."}
    )

    beta1: Optional[float] = field(
        default=0.85, # set to default adalora
        metadata={"help": "Hyperparameter of EMA."}
    )

    beta2: Optional[float] = field(
        default=0.85, # set to default adalora
        metadata={"help": "Hyperparameter of EMA."}
    )

    orth_reg_weight: Optional[float] = field(
        default=0.5, # set to default adalora
        metadata={"help": "The orthogonal regularization coefficient."}
    )

    total_step: Optional[int] = field(
        default=None, # set to default adalora
        metadata={"help": "The total training steps."}
    )

    rank_pattern: Optional[dict] = field(
        default=None, # set to default adalora
        metadata={"help": "The saved rank pattern."}
    )


    # rest

    init_warmup: Optional[int] = field(
        default=4500,
        metadata={"help": "Total steps of inital warmup"},
    )
    final_warmup: Optional[int] = field(
        default=12000,
        metadata={"help": "Total steps of final fine-tuning"},
    )
    tb_writter_loginterval: Optional[int] = field(
        default=500,
        metadata={"help": "The logging interval for tb_writter."},
    )

    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    print(os.getcwd())
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
    training_args.num_epochs_coop = data_args.num_epochs_coop 
    # training_args.learning_rate = model_args.learning_rate
    
    wandb_project = data_args.wandb_project
    wandb_watch = data_args.wandb_watch
    wandb_log_model = data_args.wandb_log_model
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
    wandb_run_name = f'{model_args.method}-{model_args.model_name_or_path}-{data_args.task_name}-{data_args.run_tag}'
    
    if use_wandb:
        training_args.report_to = ["wandb"]
        training_args.run_name=wandb_run_name

    print(training_args)

        
    training_args.loraplus_lr_ratio = data_args.loraplus_lr_ratio
    if "debug" in training_args.output_dir:
        import ipdb
        ipdb.set_trace()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = (training_args.output_dir)
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
    set_seed(training_args.seed, model_args.use_det)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
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
    if data_args.task_name in ['mrpc','rte','cola','stsb','qnli','mnli','qqp','sst2','wnli']:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    
    elif data_args.task_name in ['cb','wic','boolq','axg','axb','copa']:
        datasets = load_dataset("super_glue", data_args.task_name)
    
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

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

    # define use_dora
    use_dora = False
    if model_args.method in ['dora', 'DoRA']:
        use_dora = True

    if model_args.apply_lora == False:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # MonteCLoRA args
            expert_locations=model_args.monteclora_targets,
            monteclora_n=model_args.monteclora_n,
            monteclora_m=model_args.monteclora_m,
            sample_period=model_args.sample_period,
            var_loss_scale = model_args.var_loss_scale,
            use_entropy = model_args.use_entropy,
            use_averaging = model_args.use_averaging,
            kl_loss_weight = model_args.kl_loss_weight,
            apply_lora=model_args.apply_lora,
            lora_module=model_args.lora_module, 
            lora_alpha=model_args.lora_alpha,
            lora_r=model_args.lora_r,    
            # dora
            use_dora=use_dora
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            expert_locations=model_args.monteclora_targets,
            monteclora_n=model_args.monteclora_n,
            monteclora_m=model_args.monteclora_m,
            sample_period=model_args.sample_period,
            var_loss_scale = model_args.var_loss_scale,
            use_entropy = model_args.use_entropy,
            use_averaging = model_args.use_averaging,
            kl_loss_weight = model_args.kl_loss_weight,
            # dora
            use_dora=use_dora
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
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

    print (model)

    trainable_params = []

    if model_args.apply_lora:
        if model_args.method in ['adalora', 'AdaLoRA']: 
            config = AdaLoraConfig(
                target_modules=(model_args.target_modules).split(','),
                # monteclora_at = model_args.monteclora_at,
                monteclora_targets = (model_args.monteclora_targets).split(','),
                bias="none",
                task_type="TOKEN_CLS",
                monteclora_n=model_args.monteclora_n,
                monteclora_m=model_args.monteclora_m,
                use_entropy = model_args.use_entropy,
                kl_loss_weight=model_args.kl_loss_weight,
                lora_dropout=0.1,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,   
                use_monteclora=model_args.use_monteclora,
                monteclora_at=model_args.monteclora_at,
                sample_scaler = model_args.sample_scaler,
                dirichlet_prior=model_args.dirichlet_prior,
                # task_type="CAUSAL_LM",
                mc_training=model_args.mc_training,
                # dora
                use_dora=use_dora,
                # adalora specific
                target_r=model_args.target_r,
                init_r=model_args.init_r,
                tinit=model_args.tinit,
                tfinal=model_args.tfinal,
                deltaT=model_args.deltaT,
                beta1=model_args.beta1,
                beta2=model_args.beta2,
                orth_reg_weight=model_args.orth_reg_weight,
                total_step=model_args.total_step,
                rank_pattern=model_args.rank_pattern,
            )
        else:
            config = LoraConfig(
                target_modules=(model_args.target_modules).split(','),
                # monteclora_at = model_args.monteclora_at,
                monteclora_targets = (model_args.monteclora_targets).split(','),
                bias="none",
                task_type="TOKEN_CLS",
                monteclora_n=model_args.monteclora_n,
                monteclora_m=model_args.monteclora_m,
                use_entropy = model_args.use_entropy,
                kl_loss_weight=model_args.kl_loss_weight,
                lora_dropout=0.1,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,   
                use_monteclora=model_args.use_monteclora,
                monteclora_at=model_args.monteclora_at,
                sample_scaler = model_args.sample_scaler,
                dirichlet_prior=model_args.dirichlet_prior,
                # task_type="CAUSAL_LM",
                mc_training=model_args.mc_training,
                # dora
                use_dora=use_dora
            )

        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()
        model = get_peft_model(model, config)
        # profiler.stop()
        # print(profiler.output_text(unicode=True, color=True))

        # model.print_trainable_parameters()

        #if model_args.lora_path is not None:
        #    lora_state_dict = torch.load(model_args.lora_path)
        #    logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
        #    logger.info(lora_state_dict.keys())
        #    model.load_state_dict(lora_state_dict, strict=False)
   
        #trainable_params.append('lora')
        #trainable_params.append('resweight')

        #create loraplus optimizer if method is loraplus
        optimizer = None
        scheduler = None
        if model_args.method == 'loraplus':
            optimizer = create_loraplus_optimizer(
                model=model,
                optimizer_cls=torch.optim.AdamW,
                lr=training_args.learning_rate,
                loraplus_lr_ratio=training_args.loraplus_lr_ratio,
            )
            scheduler = None

        


    '''
    num_param = 0
    # if no trainable_params then perform full finetuning
    print (trainable_params)
    names = set()
    for name, _ in model.named_parameters():
        names.add(name)
    if len(trainable_params) > 0 :
        for name, param in model.named_parameters():
            if name.startswith('deberta') or name.startswith('roberta') or name.startswith('bert'):
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
    '''
    print (model)
    
    #logger.info("Number of Trainable Parameters: %d"%(int(num_param))) 

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

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
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    if data_args.test_adv_glue:
        adv_glue = load_dataset("json", data_files = "adv_dev.json")
        features = adv_glue['train'][data_args.task_name][0][0].keys()
        from collections import defaultdict
        mep = defaultdict(lambda : [])
        for item in adv_glue['train'][data_args.task_name][0]:
            for f in features:
                mep[f].append(item[f])

        adv_dataset = Dataset.from_dict(mep)
        datasets['adv_validation'] = adv_dataset
        
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
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
        if data_args.task_name in ['mrpc','rte','stsb','qnli','mnli','qqp','sst2','wnli']:
            metric = evaluate.load("glue", data_args.task_name)
        elif data_args.task_name=="cola" and "roberta" in model_args.model_name_or_path:
            metric = evaluate.load("glue","mrpc")
        elif data_args.task_name=="cola" and model_args.model_name_or_path!="roberta-base":
            metric = evaluate.load("glue","cola")
        elif data_args.task_name in['cb','wic','boolq','axg','axb','copa']:
            metric=evaluate.load("super_glue",data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    def expected_caliberation_error(samples,true_labels, M=10):
        bin_boundaries=torch.linspace(0,1,M+1)
        bin_lowers=bin_boundaries[:-1]
        bin_uppers=bin_boundaries[1:]
        confidences,_=torch.max(samples,dim=1)
        predicted_label=torch.argmax(samples,dim=1)
        accuracies=torch.eq(predicted_label,true_labels)
        ece=torch.zeros(1)
        for bin_lower,bin_upper in zip(bin_lowers,bin_uppers):
            in_bin=torch.logical_and(confidences>bin_lower,confidences<=bin_upper)
            prob_in_bin=torch.mean(in_bin.float())
            if prob_in_bin>0:
                accuracy_in_bin=torch.mean(accuracies[in_bin].float())
                avg_confidence_in_bin=torch.mean(confidences[in_bin])
                ece+=torch.abs(avg_confidence_in_bin-accuracy_in_bin)*prob_in_bin
        return ece[0]
    
    def negative_log_likelihood(samples, true_labels):
        probs=samples[torch.arange(len(samples)),true_labels]
        log_probs=torch.log(probs)
        nll_sum=-torch.mean(log_probs)
        return nll_sum.item()
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds2=torch.from_numpy(preds)
        if data_args.task_name!="stsb":    
            sam=nn.functional.softmax(preds2,dim=1)
            print(sam[:10])
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if data_args.task_name!='stsb':
                result["ece"]=expected_caliberation_error(sam,torch.from_numpy(p.label_ids),M=10)
                result["nll"]=negative_log_likelihood(sam,torch.from_numpy(p.label_ids))
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

    rankallocator = None

    max_seq_length = data_args.max_seq_length
    text = ""
    inputs = tokenizer(text,
                    add_special_tokens=True, 
                    return_attention_mask=True,
                    padding=True,
                    truncation="longest_first",
                    max_length=max_seq_length)

    if len(inputs["input_ids"]) < max_seq_length:
        apply_num = max_seq_length-len(inputs["input_ids"])
        inputs["input_ids"].extend([0]*apply_num)
        #inputs["token_type_ids"].extend([0]*apply_num)
        inputs["attention_mask"].extend([0]*apply_num)
    
    inputs["input_ids"] = torch.tensor([inputs["input_ids"]])
    #inputs["token_type_ids"] = torch.tensor([inputs["token_type_ids"]])
    inputs["attention_mask"] = torch.tensor([inputs["attention_mask"]])

    batch_size = 1
    input_shape = (batch_size, 128)
    #flops, macs, params = calculate_flops(model=model, 
    #                                  input_shape = input_shape,
    #                                  output_as_string=True,
    #                                  output_precision=4,
    #                                  transformer_tokenizer=tokenizer)
    print ("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    #print("T5 FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    # Initialize our Trainer
    # for _, p in model.named_parameters():
    #     print(p.requires_grad)
    
    for module in list(dict(model.named_modules()).values()):
        if type(module).__name__ == 'MonteCLoRASampler':
            module.mc_training = True #check if this is the right correction

    posthoc_flag = model_args.posthoc_app

    #print (model.roberta.encoder.layer[0].attention.self.query.weight)
    print ("Posthoc flag", posthoc_flag)




    if posthoc_flag == 1:
        training_args.num_train_epochs = training_args.num_std_epochs #training_args.num_train_epochs//2
        for module in list(dict(model.named_modules()).values()):
            if type(module).__name__ == 'MonteCLoRASampler':
                module.mc_training = False #check correction
        
            trainer = Trainer(
	        model=model,
	        args=training_args,
	        train_dataset=train_dataset if training_args.do_train else None,
	        eval_dataset=eval_dataset if training_args.do_eval else None,
	        compute_metrics=compute_metrics,
	        tokenizer=tokenizer,
	        data_collator=data_collator,
            optimizers=(optimizer, scheduler),
	        # callbacks=[tb_writter],
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
            print(f'checkpoint from pre: {checkpoint}')
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

    print("MonteCLoRA RUN")
    if model_args.method.lower() == 'monteclora':
        training_args.num_train_epochs = training_args.num_epochs_coop #orig_num_epochs - training_args.num_train_epochs
    print(training_args.num_train_epochs)
    if posthoc_flag == 1:
        training_args.learning_rate = training_args.learning_rate * 5
    for module in list(dict(model.named_modules()).values()):
        if type(module).__name__ == 'MonteCLoRASampler':
            module.mc_training = True #check if this is correct
            # module.initialize_prior_fine() #un-comment to use cov initialization
    if posthoc_flag == 1:
        for n, p in model.named_parameters():
            if "expert_weights_prior" not in n and "std_prior" not in n:
                p.requires_grad = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        # callbacks=[tb_writter],
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
                print(f'checkpoint from pre 2: {checkpoint}')
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

    if data_args.test_adv_glue:
        eval_datasets = [datasets['adv_validation']]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            for key in metrics:
                if tb_writter:
                    tb_writter.add_scalar("Eval_%s/%s"%(task, key), metrics[key], training_args.num_train_epochs)
                logger.info("{task} {key}: {value}:".format(task=task, key=key, value=metrics[key]))

            trainer.log_metrics("Adversarial Eval_%s"%task, metrics)
            trainer.save_metrics("Adversarial Eval_%s"%task, metrics)

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


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    print(os.getcwd())
    main()