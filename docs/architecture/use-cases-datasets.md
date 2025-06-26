# Dataset Management Use Cases

This diagram illustrates the various use cases for dataset management in the Vectorize system.

```plantuml
--8<-- "docs/use-cases-datasets.plantuml"
```

## Use Case Categories

### Dataset Upload

- **Single File Upload**: Upload individual datasets in supported formats (.json, .csv, .xml)
- **Batch Upload**: Upload multiple datasets simultaneously for efficient processing
- **Format Validation**: Automatic validation and conversion of dataset formats

### Dataset Management

- **Rename Datasets**: Modify dataset names and metadata
- **Delete Datasets**: Remove datasets and associated data
- **Retrieve Datasets**: Access individual datasets or browse collections
- **Search and Filter**: Find datasets by various criteria

## Supported Formats

- **JSON**: Structured data with flexible schema support
- **CSV**: Tabular data with automatic delimiter detection
- **XML**: Hierarchical data with schema validation
- **Hugging Face**: Direct integration with HF dataset hub

## Workflow Integration

Dataset management integrates seamlessly with:

- Training pipelines for model fine-tuning
- Evaluation processes for model assessment
- Synthesis workflows for data augmentation
