# CNN Architecture Experimentation and Optimization

## Overview

The pre-training pipeline (`core/pre_trainer.py`) has been designed to support experimentation with different Convolutional Neural Network (CNN) architectures. This flexibility allows for testing various network depths, layer configurations, and regularization techniques to find the optimal setup for specific pattern recognition tasks.

The results of each training run, including validation metrics like loss, accuracy, precision, recall, and F1-score, are logged in `memory/pre_train_log.json`. This log file serves as a crucial resource for comparing the performance of different architectures.

## Defining Architectures

New CNN architectures can be defined within the `core/params_manager.py` module, which are then persisted in the `memory/learned_params.json` file.

### `cnn_architecture_configs` Parameter

The core of this flexibility lies in the `cnn_architecture_configs` parameter found under the `global` key in `learned_params.json`. This parameter is a dictionary where:
*   Each **key** is a unique string identifier for an architecture (e.g., `"default_simple"`, `"my_custom_arch"`).
*   Each **value** is another dictionary specifying the parameters for the `SimpleCNN` constructor in `core/pre_trainer.py`.

The parameters for each architecture definition must match those accepted by `SimpleCNN.__init__`:
*   `num_conv_layers: int`
*   `filters_per_layer: list`
*   `kernel_sizes_per_layer: list`
*   `strides_per_layer: list`
*   `padding_per_layer: list`
*   `pooling_types_per_layer: list` (elements can be 'max', 'avg', or `None`)
*   `pooling_kernel_sizes_per_layer: list`
*   `pooling_strides_per_layer: list`
*   `use_batch_norm: bool`
*   `dropout_rate: float`

### Example: Adding a New Architecture

To define a new architecture, you would edit the `_get_default_params()` method in `core/params_manager.py` (if you want it as a new default) or modify the `memory/learned_params.json` file directly (for an existing setup).

**Example of adding "my_custom_arch" to `cnn_architecture_configs`:**

```json
{
    "global": {
        // ... other global parameters ...
        "current_cnn_architecture_key": "default_simple",
        "cnn_architecture_configs": {
            "default_simple": {
                "num_conv_layers": 2,
                "filters_per_layer": [16, 32],
                "kernel_sizes_per_layer": [3, 3],
                "strides_per_layer": [1, 1],
                "padding_per_layer": [1, 1],
                "pooling_types_per_layer": ["max", "max"],
                "pooling_kernel_sizes_per_layer": [2, 2],
                "pooling_strides_per_layer": [2, 2],
                "use_batch_norm": false,
                "dropout_rate": 0.0
            },
            "deeper_with_batchnorm": {
                // ... (definition as before) ...
            },
            "my_custom_arch": {
                "num_conv_layers": 3,
                "filters_per_layer": [32, 64, 128],
                "kernel_sizes_per_layer": [5, 3, 3],
                "strides_per_layer": [1, 1, 1],
                "padding_per_layer": [2, 1, 1],
                "pooling_types_per_layer": ["max", "avg", null],
                "pooling_kernel_sizes_per_layer": [2, 2, 1], // Dummy for None pool
                "pooling_strides_per_layer": [2, 2, 1],    // Dummy for None pool
                "use_batch_norm": true,
                "dropout_rate": 0.3
            }
        }
        // ... other global parameters ...
    }
    // ... strategies, etc. ...
}
```

## Selecting an Architecture for Training

To instruct the `PreTrainer` to use a specific architecture for its next training run, you need to modify the `current_cnn_architecture_key` parameter in `memory/learned_params.json`.

Set its value to the key of the desired architecture defined in `cnn_architecture_configs`. For example, to use the custom architecture defined above:

```json
{
    "global": {
        // ...
        "current_cnn_architecture_key": "my_custom_arch", // Changed from "default_simple"
        "cnn_architecture_configs": {
            // ...
        },
        // ...
    }
}
```
When the pipeline runs, `PreTrainer` will fetch this key and use the corresponding dictionary from `cnn_architecture_configs` to instantiate the `SimpleCNN` model.

## Running the Pre-Training Pipeline

As described in `docs/data_pipeline.md`, the pipeline is typically run using:

```bash
python -m core.pre_trainer
```
This will use the architecture specified by `current_cnn_architecture_key` for all model training sessions in that run.

## Comparing Results

The primary way to compare the performance of different CNN architectures is by examining the `memory/pre_train_log.json` file.

*   Each entry in this JSON log corresponds to a training session for a specific combination of:
    *   Trading symbol (e.g., "ETH_EUR")
    *   Timeframe (e.g., "1h")
    *   Pattern type (e.g., "bullFlag")
    *   **CNN Architecture** (e.g., "default_simple", "my_custom_arch")
*   The `model_type` field in each log entry is a composite string that now includes the architecture key, for example: `ETH_EUR_1h_bullFlag_default_simple` or `ETH_EUR_1h_bullFlag_my_custom_arch`.
*   You can compare metrics such as `best_validation_loss`, `best_validation_accuracy`, `best_validation_precision`, `best_validation_recall`, and `best_validation_f1` for the same symbol/timeframe/pattern across different architecture keys. This allows for quantitative assessment of which architecture performs better for specific tasks.
*   Trained models and their corresponding scaler parameters are saved with the architecture key in their filenames, allowing for distinct storage and later retrieval:
    *   Models: `data/models/{symbol_sanitized}/{timeframe}/cnn_model_{pattern_type}_{architecture_key}.pth`
    *   Scalers: `data/models/{symbol_sanitized}/{timeframe}/scaler_params_{pattern_type}_{architecture_key}.json`

## Next Steps (Advanced)

While manually defining and selecting architectures provides good flexibility, future enhancements could include:

*   **Automated Hyperparameter Optimization:** Tools like Optuna could be integrated to systematically search for optimal architectural parameters (number of layers, filter sizes, etc.) and training hyperparameters.
*   **Experiment Tracking Tools:** Platforms like MLflow or Weights & Biases (W&B) could be used for more sophisticated tracking of experiments, model versions, metrics, and parameters, making comparisons and collaboration easier.

By leveraging the current flexible architecture configuration, users can already perform systematic experiments to improve model performance.
