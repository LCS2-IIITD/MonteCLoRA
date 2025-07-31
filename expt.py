import argparse
import subprocess
import re
import os


glue_tasks = [
    "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli",
    "wic", "boolq", "cb", "axg", "axb", "copa", "adv_mnli", "adv_qnli",
    "adv_qqp", "adv_rte", "adv_sst2"
]

summ_datasets = [
    "amazon_reviews_multi", "big_patent", "cnn_dailymail", "orange_sum", "pn_summary",
    "psc", "samsum", "thaisum", "xglue", "xsum", "wiki_summary", "multi_news"
]


def replace_args_in_script(script_path, method, args_dict):
    """Replaces placeholders in a shell script with given arguments."""
    with open(script_path, "r") as file:
        script_content = file.read()

    # Replace arguments in the shell script
    for key, value in args_dict.items():
        if key in ['do_train', 'do_eval','apply_lora']:
            pattern = rf"--{key} (\S+)"
            if re.search(pattern, script_content):
                script_content = re.sub(pattern, f"--{key} \\\\", script_content)
            else:
                script_content += f"\n--{key} \\\\"  # Append if not present
        else:
            pattern = rf"--{key} (\S+)"
            if re.search(pattern, script_content):
                script_content = re.sub(pattern, f"--{key} {value}", script_content)
            else:
                script_content += f"\n--{key} {value} \\"  # Append if not present

    # turns off monteclora training for other methods   
    if method not in ['MonteCLoRA', 'monteclora']:
        for key in ['use_monteclora', 'mc_training']:
            pattern = rf"--{key} (\S+)"
            if re.search(pattern, script_content):
                script_content = re.sub(pattern, f"--{key} False", script_content)
            else:
                script_content += f"\n--{key} False"

    return script_content

def write_temp_script(modified_script, output_dir, tag):
    """Writes the modified script to a temporary file in the specified output directory."""
    temp_script_path = os.path.join(output_dir, "run_expt.sh")
    with open(temp_script_path, "w") as file:
        file.write(modified_script)
    return temp_script_path

def main():
    parser = argparse.ArgumentParser(description="Runs experiments corresponding to dataset on specified model.")

    # required args
    parser.add_argument("--method", type=str, required=True, help="PEFT method, choose from MonteCLoRA, LoRA, LoRA+, AdaLoRA, DoRA")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_or_task_name", type=str, required=True, help="Dataset or task name, str")


    # optional args
    parser.add_argument("--use_det", type=str, default='', help="If deterministic torch backend")
    parser.add_argument("--wandb_project", type=str, default='', help="WandB project")
    parser.add_argument("--run_tag", type=str, default='', help="WandB run identifier")
    parser.add_argument("--output_dir", type=str, help="Output directory for trained model")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per device during training")
    parser.add_argument("--kl_loss_weight", type=float, help="KL loss weight")
    parser.add_argument("--use_entropy", help="Whether to use entropy regularization")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--num_epochs_coop", type=int, help="Number of cooperative training epochs")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--learning_rate_coop", type=float, help="Learning rate for cooperative training")
    parser.add_argument("--loraplus_lr_ratio", type=float, help="ηB/ηA where ηA = learning_rate")

    # _n_gpu: int = field(init=False, repr=False, default=-1)
    parser.add_argument("--n_gpu", type=int, help="Number of GPUs to use")

    # eval and logging args
    parser.add_argument("--predict_with_generate", type=bool, default=False, help="Predict with generation")
    parser.add_argument("--eval_steps", type=int, help="Evaluation step interval")
    parser.add_argument("--evaluation_strategy", type=str, default='steps', help="Steps after which evaluatio should be done")
    parser.add_argument("--save_steps", type=int, help="Model saving step interval")
    parser.add_argument("--save_strategy", type=str, default='steps', help="Strategy for evaluation") 
    parser.add_argument("--logging_steps", type=int, default=10, help="Steps after which logging should be done")   
    parser.add_argument("--report_to", type=str, default='wandb', help="Reporting tool")
    parser.add_argument("--seed", type=int, default=6, help="Random seed for initialization")

    parser.add_argument("--lora_alpha", type=int, help="LoRA scaling factor")
    parser.add_argument("--posthoc_app", type=int, help="Post-hoc application setting")
    parser.add_argument("--target_modules", type=str, help="Target modules for ada2ptation, str or List[str]")
    parser.add_argument("--use_monteclora", help="Whether to use MonteClora")
    parser.add_argument("--monteclora_at", type=str, help="MonteClora_at, str or List[str]")
    parser.add_argument("--monteclora_targets", type=str, help="MonteClora target modules, str or List[str]")
    parser.add_argument("--overwrite_output_dir", help="Overwrite output directory")
    parser.add_argument("--apply_lora", action="store_true", help="Whether to apply LoRA")
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length for input data")
    # adalora params
    parser.add_argument("--target_r", type=int, help="Target Lora matrix dimension.")
    parser.add_argument("--init_r", type=int, help="Initial Lora matrix dimension.")
    parser.add_argument("--tinit", type=int, help="The steps of initial warmup.")
    parser.add_argument("--tfinal", type=int, help="The steps of final warmup.")
    parser.add_argument("--deltaT", type=int, help="Step interval of rank allocation.")
    parser.add_argument("--beta1", type=float, help="Hyperparameter of EMA.")
    parser.add_argument("--beta2", type=float, help="Hyperparameter of EMA.")
    parser.add_argument("--orth_reg_weight", type=float, help="The orthogonal regularization coefficient.")
    parser.add_argument("--total_step", type=int, help="The total training steps.")
    parser.add_argument("--rank_pattern", type=dict, help="The saved rank pattern.")


    ## monte_clora args
    # monteclora_n, monteclora_m, dirichlet_prior - int
    # sample_scaler, kl_loss_weight - float
    parser.add_argument("--monteclora_n", type=int, default=4, help="Number of Monte Carlo samples")
    parser.add_argument("--monteclora_m", type=int, default=4, help="Number of Gaussians")
    parser.add_argument("--dirichlet_prior", type=int, default=1, help="Dirichlet prior")
    parser.add_argument("--sample_scaler", type=float, default=1e-3, help="Sample scaler")



    args = parser.parse_args()

    args_dict = {k: v for k, v in vars(args).items() if v is not None and k not in ['dataset_or_task_name']}
    

    if args.dataset_or_task_name in glue_tasks:
        args_dict['task_name'] = args.dataset_or_task_name
        script = 'run_glue.sh'
    elif args.dataset_or_task_name in summ_datasets:
        args_dict['dataset_name'] = args.dataset_or_task_name
        script = 'run_summ.sh'
    else:
        print(args.dataset_or_task_name)
        raise ValueError('Invalid dataset or task name.')
    
    method = args.method # LoRA, AdaLoRA, MonteCLoRA etc

    # directories based on model name and task name
    # Normalize model name by removing '/home/models/' or 'home/models/' if it exists
    model_name = re.sub(r"^/?home/models/", "", args.model_name_or_path)
    # Create directory based on the cleaned model name
    model_dir = os.path.join(os.getcwd(), 'expt-runs', method, model_name.replace("/", "_"))
    task_output_dir = os.path.join(model_dir, args.dataset_or_task_name + '-' + args.run_tag)
    
    os.makedirs(task_output_dir, exist_ok=True)

    args_dict["output_dir"] = task_output_dir # output inside the new model folder

    # Replace script arguments
    modified_script = replace_args_in_script(script, method, args_dict)

    # Write modified script to a temporary file inside the model directory
    temp_script_path = write_temp_script(modified_script, task_output_dir, args.run_tag)
    # print(task_output_dir)
    # print(temp_script_path)
    # Execute the modified script
    print("pls god ji")
    print(temp_script_path)
    subprocess.run(["bash", temp_script_path], check=True)

if __name__ == "__main__":
    main()
