# Mimic Tear

Mimic Tear is a neural network model that learns to mimic a player's behavior to defeat Elden Ring bosses autonomously.

## Codebase Architecture

```text
    mimic_tear/
    ├── model/          # the neural network components
    ├── training/       # trains the model
    ├── player/         # live inference
    └── mimic.py        # orchestrates Mimic Tear

    data/
    ├── capture/        # captures data
    ├── write/          # writes captured data to disk
    ├── process/        # processes data into Tensor datasets
    ├── models/         # data models
    └── pipeline.py     # orchestrates the data pipeline

    configs/
    ├── models/         # config models
    ├── config.toml     # config file for Mimic Tear
    └── config.py       # orchestrates config loading

    utils/              # utility functions
```