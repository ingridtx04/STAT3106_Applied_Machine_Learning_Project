# STAT3106_Applied_Machine_Learning_Project

## PennyChem: Molecule Generation Under a Synthesis Budget

### How to run

The default comparison architecture is `EMBED_DIM=128`, `NUM_LAYERS=2`, `LATENT_DIM=128`, with VAE encoder layers `LAYER_DIMS=256,512`. The generator parameter match uses VAE decoder `HIDDEN_DIM=480` and LSTM `HIDDEN_DIM=512`.

1. Train VAE: `CKPT_DIR=checkpoints python SMILE_VAE.py`
2. Train LSTM baseline: `CKPT_DIR=checkpoints python train_lstm.py`
3. Train normalized property heads: `CKPT_DIR=checkpoints RESULTS_DIR=results python train_property_heads.py`
4. Compare/generate results: open `compare.ipynb`

Full retrain:

```bash
MAX_ROWS_TRAIN=1000000 MAX_ROWS_PRED=50000 \
SAMPLE_POOL_ROWS=1000000 \
VAE_HIDDEN_DIM=480 LSTM_HIDDEN_DIM=512 \
VAE_EPOCHS=30 LSTM_EPOCHS=15 PRED_EPOCHS=100 \
PROPERTIES=qed,sa,mw ./run_all_training
```

### Project structure

- `model.py` — VAE, LSTM support heads, and normalized property heads
- `train.py` — VAE trainer
- `train_property_heads.py` — QED/SA/MW/SCScore/SYBA heads trained on frozen VAE latents with standardized inputs and targets; use `SAMPLE_POOL_ROWS` to match the VAE training row pool
- `evaluate.py` — metrics (validity, QED, SA score)
- `generate.py` — constrained generation with λ penalty
- `lstm_baseline.py` — LSTM baseline
- `checkpoints/` — one VAE checkpoint, one LSTM checkpoint, and `predictors/`
- `results/` — generated metrics, plots, and notebook outputs
- `Results.ipynb` — Main notebook that shows results
