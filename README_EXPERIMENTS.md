# Details gsm-8k, humaneval and commonsense tasks

## Relevant scripts:

### Python scripts
- finetune_gsm8k.py : Python file for fine tuning with gsm8k
- finetune_code.py : Python file for fine tuning with the code dataset
- finetune.py : Python file for fine tuning with commonsense dataset

### Bash scripts
- expt_gsm.sh : contains hyperparameters and different methods for fine tuning lora on gsm-8k
- expt_humaneval.sh : fine-tuning on humaneval

### Evaluation

We use LM-eval-harness for evaluation of gsm8k and humaneval tasks.

We use settings and prompt similar to those mentioned in the LLaMA-3 evaluations.

The evaluation is a part of the above two bash scripts.


For commonsense we use the evaluation scripts as given in the LLM Adapter repository:

- commonsense_evaluate.py
- commonsense_example.sh - Gives example fine tuning and evaluation script
