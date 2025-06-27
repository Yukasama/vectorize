# API Reference

Welcome to the Vectorize API documentation. This section provides comprehensive documentation for all API endpoints, services, and modules.

## API Overview

The Vectorize API is organized into several main modules:

- **[Common/Health](api/common.md)**: Health checks and basic utilities
- **[Datasets](api/datasets.md)**: Dataset upload, management, and Hugging Face integration
- **[AI Models](api/models.md)**: Model management and CRUD operations
- **[Training](api/training.md)**: Model training with background task management
- **[Evaluation](api/evaluation.md)**: Model evaluation and performance metrics
- **[Inference](api/inference.md)**: Text embedding generation and inference
- **[Upload](api/upload.md)**: Model upload from various sources
- **[Synthesis](api/synthesis.md)**: Synthetic data generation
- **[Tasks](api/tasks.md)**: Background task management and monitoring

## Quick Reference

### Base URL

```
http://localhost:8000
```

### Authentication

Currently, the API does not require authentication.

### Content Types

- **Request**: `application/json`, `multipart/form-data` (for file uploads)
- **Response**: `application/json`

### Common Response Codes

- `200 OK`: Successful operation
- `201 Created`: Resource created successfully
- `202 Accepted`: Request accepted for background processing
- `304 Not Modified`: Resource is already up to date
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
