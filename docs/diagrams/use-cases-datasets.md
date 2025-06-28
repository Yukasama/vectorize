# Dataset Use Cases

This diagram illustrates the various use cases for dataset management in the Vectorize system.

![file](out/use-cases-datasets.svg)

## Use Case Categories

### Dataset Upload

- **Upload Local Files**: Upload individual datasets in multiple supported formats (.json, .csv, .xml, .xlsx)
- **Upload ZIP Archive with Multiple Files**: Batch upload multiple datasets in a compressed archive for efficient processing
- **Download from Hugging Face Hub**: Direct integration with HuggingFace dataset repository with automatic schema validation and conversion

### Dataset Management

- **List Datasets with Pagination**: Browse all datasets with configurable page sizes and efficient pagination
- **Get Dataset Details**: Retrieve comprehensive information about individual datasets including metadata and statistics
- **Update Dataset Name**: Modify dataset names with optimistic locking using ETags for concurrent access control
- **Delete Dataset**: Remove datasets and associated files from the system

### Background Processing

- **Monitor Upload Status**: Track the progress of local file uploads and processing tasks
- **Track HF Download Progress**: Monitor HuggingFace dataset download and conversion operations

## Supported Formats

- **CSV**: Comma-separated values with configurable delimiters and automatic format detection
- **JSON**: Standard JSON format for structured data with flexible schema support
- **JSONL**: JSON Lines format (one JSON object per line) for streaming data processing
- **XML**: XML documents with automatic parsing and validation
- **Excel**: Microsoft Excel files (.xlsx, .xls) with configurable sheet selection
- **ZIP**: Compressed archives containing multiple dataset files for batch processing

## Key Features

### Schema Validation & Conversion

- Automatic validation against HuggingFace dataset compatibility schemas
- Standardized conversion to question/positive/negative format
- Column mapping and renaming with configurable field names
- Support for various dataset types: preference learning, instruction tuning, Q&A, and general training

### File Processing Pipeline

- Size and file count limits with configurable thresholds
- Automatic format detection and parsing
- Conversion to standardized JSONL format for consistent processing
- Metadata extraction and database storage

### HuggingFace Integration

- Dataset filtering based on supported schema patterns
- Automatic splitting by subsets and splits into separate dataset records
- Column extraction and standardization (drops unused columns)
- Background processing with status monitoring

### Version Control & Concurrency

- Optimistic locking using ETags for update operations
- Version tracking for datasets and metadata
- Concurrent access protection for update and delete operations

## Workflow Integration

Dataset management integrates seamlessly with:

- **Training Pipelines**: Provides standardized datasets for model fine-tuning and training workflows
- **Evaluation Processes**: Supplies test datasets for model assessment and benchmarking
- **Synthesis Workflows**: Sources training data for synthetic data generation
- **Background Task System**: Coordinates with the unified task monitoring system for progress tracking
- **API Integration**: RESTful endpoints for programmatic access and automation

## Error Handling & Validation

The system provides comprehensive error handling for:

- Unsupported file formats and invalid data structures
- File size and count limit violations
- Missing required columns and schema mismatches
- HuggingFace dataset availability and access issues
- Concurrent modification conflicts and version mismatches
