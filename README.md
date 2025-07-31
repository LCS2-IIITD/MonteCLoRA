## MonteCLoRA - Monte Carlo enhanced Low Rank Adaptation for Robust Fine Tuning of LLMs

The code base required a bit of cleaning and improvement in the config sections but other than that you can use it in any PEFT pipelines. We will be adding an extensive README.md but for now you could take a look at the finetuning scriptt to get an idea of how to use MonteCLoRA in your scripts.

### Environment Setup

```bash
conda env create -f environment.yml -y
```

Or you may use the requrements.txt file

### Installing PEFT

```bash
cd peft
pip install -e .
```

### Installing Transformers

```bash
cd transformers
pip install -e .
```
