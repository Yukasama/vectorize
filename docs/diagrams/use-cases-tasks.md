# Task Management Use Cases

This diagram illustrates the unified task monitoring capabilities in the Vectorize system.

![file](out/use-cases-tasks.svg)

## Use Case Overview

### Task Monitoring

The task management system provides a single endpoint (`/tasks`) that aggregates and monitors all background operations across the entire Vectorize platform:

- **Get All Tasks with Filters**: Retrieve tasks from multiple modules (upload, training, evaluation, synthesis, dataset processing) through a unified interface
- **Filter by Task Type**: Focus on specific operation types such as model uploads, training processes, evaluations, or data synthesis
- **Filter by Status**: Monitor tasks based on their current state (queued, running, completed, failed, cancelled)
- **Filter by Time Window**: Limit results to recent activities within a specified hour range
- **Filter by Tag**: Organize and track tasks using custom labels and identifiers
- **Paginate Results**: Handle large numbers of tasks efficiently with configurable page sizes

## Supported Task Types

- **`model_upload`**: Model uploads from HuggingFace, GitHub, or local files
- **`dataset_upload`**: Dataset uploads and processing operations
- **`training`**: Model training and fine-tuning processes
- **`evaluation`**: Model evaluation and benchmarking tasks
- **`synthesis`**: Synthetic data generation operations

## Key Features

### Unified Task Aggregation

- Single endpoint for monitoring all background operations
- Combines data from multiple database tables (`UploadTask`, `TrainingTask`, `EvaluationTask`, `SynthesisTask`, `UploadDatasetTask`)
- Consistent response format across all task types

### Advanced Filtering Capabilities

- **Multiple filters**: Combine task type, status, time window, and tag filters
- **Flexible status filtering**: Support for multiple status codes in a single request
- **Time-based queries**: Configurable time windows from 1 hour to weeks
- **Tag-based organization**: Custom labeling for experiments and workflows

### Performance Optimization

- Database-level pagination and sorting
- Efficient SQL query building with `UNION ALL` operations
- Indexed columns for fast filtering on status, creation date, and foreign keys

## Integration with Other Modules

The task monitoring system seamlessly integrates with:

- **Dataset Module**: Tracks local file uploads and HuggingFace dataset downloads
- **Training Module**: Monitors model training progress and completion status
- **Evaluation Module**: Tracks model evaluation and benchmarking processes
- **Synthesis Module**: Monitors synthetic data generation workflows
- **Upload Module**: Tracks model uploads from various sources

This unified approach provides comprehensive visibility into all background operations, enabling effective monitoring, debugging, and workflow management across the entire Vectorize platform.
