# Model Training & Evaluation Use Cases

This diagram shows the use cases for model training and evaluation processes in the Vectorize system.

```plantuml
--8<-- "docs/use-cases-processes.plantuml"
```

## Training Management

### Core Training Operations

- **Start Training**: Initiate model training with specified datasets and parameters
- **Monitor Progress**: Track training progress with detailed metrics and logs
- **Cancel Training**: Safely abort training processes when needed
- **Save Results**: Persist trained models and training artifacts

### Training Features

- **Dataset Integration**: Use uploaded datasets for training
- **Parameter Tuning**: Configure learning rates, batch sizes, and other hyperparameters
- **Checkpoint Management**: Automatic saving of training checkpoints
- **Resource Monitoring**: Track GPU/CPU usage and memory consumption

## Evaluation Management

### Core Evaluation Operations

- **Start Evaluation**: Launch evaluation tasks for trained models
- **Save Results**: Store evaluation metrics and detailed reports
- **Cancel Evaluation**: Abort long-running evaluation processes
- **View Results**: Access comprehensive evaluation reports and visualizations

### Evaluation Features

- **Multiple Metrics**: Support for various evaluation metrics (cosine similarity, accuracy, etc.)
- **Baseline Comparison**: Compare models against baseline and previous versions
- **Dataset Splitting**: Automatic train/validation/test splits
- **Statistical Analysis**: Detailed statistical analysis of model performance

## Workflow Integration

Both training and evaluation processes integrate with:

- **Task Management**: Background processing with status tracking
- **Model Registry**: Automatic model versioning and storage
- **Dataset Pipeline**: Seamless dataset access and preprocessing
- **Notification System**: Progress updates and completion notifications
