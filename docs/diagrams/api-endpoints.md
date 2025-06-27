# API Endpoints Architecture

This diagram shows the overall API endpoint structure and internal processes of the Vectorize service.

![file](out/api-endpoints.svg)

## Endpoint Categories

The Vectorize API is organized into several main categories:

### Model Endpoints (`/models/...`)

- Model retrieval and management
- Model upload from various sources
- Training and evaluation initiation
- Status monitoring

### Dataset Endpoints (`/datasets/...`)

- Dataset upload and management
- Synthetic data generation
- Format conversion and validation

### Internal Processes

- Asynchronous task management
- Background processing
- Database operations
- File system management

## API Flow

1. **Upload Phase**: Models and datasets are uploaded from various sources
2. **Processing Phase**: Background tasks handle conversion, validation, and storage
3. **Training/Evaluation Phase**: Asynchronous processes train and evaluate models
4. **Inference Phase**: Trained models generate embeddings for text inputs
