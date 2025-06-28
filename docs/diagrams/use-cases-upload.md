# Model Upload Use Cases

This diagram illustrates the various use cases for model upload and management in the Vectorize system.

![file](out/use-cases-uploads.svg)

## Use Case Categories

### Model Upload

- **Load Hugging Face Model by Tag**: Import pre-trained models directly from Hugging Face Hub
- **Upload Local Files**: Upload sentence transformer models from local filesystem
- **Load GitHub Repository by URL**: Import models from GitHub repositories

### Model Management

- **Rename Model**: Change display names and identifiers of uploaded models
- **Delete Model**: Remove models and associated data from the system

## Supported Sources

- **Hugging Face Hub**: Direct import using model tags and identifiers
- **Local Files**: Support for ZIP, TAR.GZ archives and individual model files
- **GitHub Repositories**: Public and private repository support with authentication
- **Multiple Formats**: Compatible with various sentence transformer model formats

## Workflow Integration

Model upload integrates seamlessly with:

- Training pipelines for base model selection
- Evaluation processes for model assessment
- Generation workflows for text embedding and similarity tasks
