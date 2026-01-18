ECG_CONFIG = {
    'input_dim': 140,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'num_classes': 2,
    'batch_size_train': 128,  # Match Colab
    'batch_size_test': 128,
    'learning_rate': 0.0001,    # Match Colab
    'weight_decay': 0.01,      # Match Colab
    'local_epochs': 5,         # 5 epochs per round
    'num_rounds': 3,           # 3 rounds × 5 = 15 total epochs
    'test_size': 0.2,
    'num_workers': 0,
    'seed': 42
}

SCHEDULER_CONFIG = {
    'max_lr': 0.0001,
    'pct_start': 0.3,
    'anneal_strategy': 'linear',
    'div_factor': 25.0,
    'final_div_factor': 10000.0
}

USE_MIXED_PRECISION = True  # Enable like Colab