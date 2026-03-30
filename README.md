# STAT3106_Applied_Machine_Learning_Project

## PennyChem: Molecule Generation Under a Synthesis Budget

### How to run
1. Train VAE: `python SMILE_VAE.py`
2. Train LSTM baseline: `python train_lstm.py`
3. Compare results: open `compare.ipynb`

### Project structure
- `model.py` — VAE architecture
- `train.py` — VAE trainer
- `evaluate.py` — metrics (validity, QED, SA score)
- `generate.py` — constrained generation with λ penalty
- `lstm_baseline.py` — LSTM baseline
