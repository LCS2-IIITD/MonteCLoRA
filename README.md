# MonteCLoRA

Source Code of Monte Carlo-enhanced Low-Rank Adaptation.

### Installation
```
conda create -n monteclora python=3.9 && conda activate monteclora
pip install -e .
pip install -r requirements.txt
cd peft && pip install -e .
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install evaluate
pip install huggingface-hub -U
```

### Run GLUE Tasks
```
sh ./scripts/run.sh
```
