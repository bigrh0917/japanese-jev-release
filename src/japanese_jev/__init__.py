"""japanese_jev: train a Japanese System One (Jev-style) decision model on a laptop.

The pipeline has four stages; see cli.py for the commands:

    build  ──>  data/*.jsonl         (data_pipeline/: public datasets -> three-primitive template)
    train  ──>  <work_dir>/adapters  (train_trl.py for CUDA GPUs, train_mlx.py for Mac)
    calibrate ─> <work_dir>/temperature.json
    evaluate ──> <work_dir>/evaluation.json
"""

__version__ = "0.1.0"
