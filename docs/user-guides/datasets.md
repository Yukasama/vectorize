# Dataset Management

The Dataset module provides comprehensive functionality for uploading, processing, and managing training datasets in Vectorize. It supports multiple file formats, validation, and seamless integration with HuggingFace datasets.

## 🚀 What the Dataset Module Can Do

The dataset management system offers powerful capabilities for handling training data:

- **📁 Multi-Format Upload**: Support for CSV, JSON, JSONL, XML, Excel files, and ZIP archives
- **🤖 HuggingFace Integration**: Direct download and processing of HF datasets
- **🔄 Format Conversion**: Automatic conversion to standardized JSONL format
- **✅ Schema Validation**: Ensure datasets match required column structures
- **📦 Batch Processing**: Handle multiple files and ZIP archives efficiently
- **🏷️ CRUD Operations**: Create, read, update, and delete dataset records

## 📂 Local File Upload

### Supported Formats

The system accepts various file formats for dataset upload:

- **CSV**: Comma-separated values with configurable delimiters
- **JSON**: Standard JSON format for structured data
- **JSONL**: JSON Lines format (one JSON object per line)
- **XML**: XML documents for structured data
- **Excel**: Microsoft Excel files (.xlsx, .xls)
- **ZIP**: Compressed archives containing multiple dataset files

For the complete list of supported formats and configuration options, see the [Dataset Configuration](../configuration.md#dataset-configuration-appdataset) section.

### Upload Examples

#### Single File Upload

```bash
# Upload a CSV file
curl -X POST "http://localhost:8000/datasets" \
  -F "files=@my_dataset.csv"

# Upload with custom options
curl -X POST "http://localhost:8000/datasets" \
  -F "files=@my_dataset.csv" \
  -F "question_name=query" \
  -F "positive_name=answer" \
  -F "negative_name=distractor"
```

```python
import httpx

async with httpx.AsyncClient() as client:
    # Upload a single file
    with open("dataset.json", "rb") as f:
        response = await client.post(
            "http://localhost:8000/datasets",
            files={"files": ("dataset.json", f, "application/json")}
        )

    print(f"Dataset uploaded: {response.headers['Location']}")
```

#### Multiple File Upload

```bash
# Upload multiple files at once
curl -X POST "http://localhost:8000/datasets" \
  -F "files=@dataset1.csv" \
  -F "files=@dataset2.json" \
  -F "files=@dataset3.xml"
```

```python
# Upload multiple files
files = [
    ("files", ("train.csv", open("train.csv", "rb"), "text/csv")),
    ("files", ("test.json", open("test.json", "rb"), "application/json")),
    ("files", ("val.xml", open("val.xml", "rb"), "application/xml"))
]

response = await client.post("http://localhost:8000/datasets", files=files)
```

#### ZIP Archive Upload

```bash
# Upload a ZIP file containing multiple datasets
curl -X POST "http://localhost:8000/datasets" \
  -F "files=@datasets.zip"
```

```python
# Upload ZIP archive
with open("datasets.zip", "rb") as f:
    response = await client.post(
        "http://localhost:8000/datasets",
        files={"files": ("datasets.zip", f, "application/zip")}
    )
```

### Upload Options

| Parameter       | Type    | Description                       | Default      |
| --------------- | ------- | --------------------------------- | ------------ |
| `question_name` | string  | Column name for questions/queries | `"question"` |
| `positive_name` | string  | Column name for positive examples | `"positive"` |
| `negative_name` | string  | Column name for negative examples | `"negative"` |
| `sheet_index`   | integer | Excel sheet index to process      | `0`          |

### File Processing Pipeline

1. **Validation**: Check file format, size, and structure
2. **Parsing**: Convert to pandas DataFrame based on file type
3. **Schema Mapping**: Map columns to standard question/positive/negative format
4. **Conversion**: Save as JSONL format for consistent processing
5. **Database Storage**: Create dataset record with metadata

### Size and File Limits

The upload system enforces several limits to ensure system stability:

- **Maximum file size**: Configurable per upload (default: 50GB)
- **Maximum ZIP members**: Configurable file count in archives (default: 10,000)
- **Filename length**: Maximum characters in filenames (default: 255)

These limits can be adjusted in the configuration. See [File Handling & Storage](../configuration.md) for details.

## 🤖 HuggingFace Dataset Integration

### Overview

The HuggingFace integration allows you to download and process datasets directly from the HuggingFace Hub. The system automatically handles dataset validation, splitting, and conversion.

### Schema Filtering

Before downloading, datasets are validated against supported schema patterns. The system supports various column combinations for different use cases:

- **Preference Learning**: `["prompt", "chosen", "rejected"]`
- **Instruction Tuning**: `["instruction", "output_1", "output_2"]`
- **Q&A Training**: `["question", "positive", "negative"]`
- **General Training**: `["input", "response_a", "response_b"]`

For the complete list of supported schemas, see [HuggingFace Schema Validation](../configuration.md#hugging-face-schema-validation).

### Column Mapping

The system automatically maps HuggingFace dataset columns to the standard format:

- **Question columns**: `anchor`, `q`, `query`, `prompt` → `question`
- **Positive columns**: `answer`, `chosen` → `positive`
- **Negative columns**: `random`, `rejected`, `no_context` → `negative`

Column mappings are configurable. See [Dataset Configuration](../configuration.md#dataset-configuration-appdataset) for customization options.

### Upload Examples

#### Basic HuggingFace Upload

```bash
# Upload a HuggingFace dataset
curl -X POST "http://localhost:8000/datasets/huggingface" \
  -H "Content-Type: application/json" \
  -d '{"dataset_tag": "squad"}'
```

```python
# Upload HuggingFace dataset
response = await client.post(
    "http://localhost:8000/datasets/huggingface",
    json={"dataset_tag": "Anthropic/hh-rlhf"}
)

# Get task ID for monitoring
task_id = response.headers["Location"].split("/")[-1]
print(f"Upload task ID: {task_id}")
```

#### Monitor Upload Progress

```bash
# Check upload status
curl "http://localhost:8000/datasets/huggingface/status/{task_id}"
```

```python
# Monitor upload progress
async def wait_for_upload(task_id: str):
    while True:
        response = await client.get(f"http://localhost:8000/datasets/huggingface/status/{task_id}")
        task = response.json()

        if task["task_status"] == "D":  # Done
            print("✅ Upload completed successfully!")
            break
        elif task["task_status"] == "F":  # Failed
            print(f"❌ Upload failed: {task.get('error_msg', 'Unknown error')}")
            break

        print("🔄 Upload in progress...")
        await asyncio.sleep(10)
```

### Dataset Splitting and Processing

The HuggingFace integration creates separate dataset records for each subset and split:

#### Example Processing Flow

```python
# Dataset: "Anthropic/hh-rlhf"
# Original structure:
# - Subsets: ["default"]
# - Splits: ["train", "test"]

# Result: Two separate datasets created:
# 1. "Anthropic/hh-rlhf_train" (from train split)
# 2. "Anthropic/hh-rlhf_test" (from test split)
```

#### Complex Dataset Example

```python
# Dataset: "squad"
# Original structure:
# - Subsets: ["plain_text", "validation"]
# - Splits: ["train", "validation"]

# Result: Four separate datasets created:
# 1. "squad_plain_text_train"
# 2. "squad_plain_text_validation"
# 3. "squad_validation_train"
# 4. "squad_validation_validation"
```

### Column Processing

During HuggingFace processing, the system:

1. **Identifies** relevant columns using configurable mapping
2. **Extracts** only the question, positive, and negative columns
3. **Renames** them to the standard format
4. **Drops** all other columns to maintain consistency

```python
# Example transformation:
# Original HF dataset columns: ["prompt", "chosen", "rejected", "metadata", "score"]
#
# After processing:
# - "prompt" → "question"
# - "chosen" → "positive"
# - "rejected" → "negative"
# - "metadata" and "score" are dropped
```

## 📋 Dataset Management Operations

### List Datasets

```bash
# Get all datasets with pagination
curl "http://localhost:8000/datasets?limit=20&offset=0"
```

```python
# Get datasets with pagination
response = await client.get("http://localhost:8000/datasets", params={
    "limit": 50,
    "offset": 0
})

datasets = response.json()
print(f"Found {datasets['total']} datasets")
for dataset in datasets['items']:
    print(f"📊 {dataset['name']} ({dataset['rows']} rows)")
```

### Get Dataset Details

```bash
# Get specific dataset
curl "http://localhost:8000/datasets/{dataset_id}"
```

```python
# Get dataset details
response = await client.get(f"http://localhost:8000/datasets/{dataset_id}")
dataset = response.json()

print(f"Dataset: {dataset['name']}")
print(f"Source: {dataset['source']}")
print(f"Rows: {dataset['rows']}")
print(f"Classification: {dataset['classification']}")
```

### Update Dataset

```bash
# Update dataset name
curl -X PUT "http://localhost:8000/datasets/{dataset_id}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Dataset Name"}'
```

```python
# Update dataset metadata
response = await client.put(
    f"http://localhost:8000/datasets/{dataset_id}",
    json={"name": "My Updated Dataset"}
)
```

### Delete Dataset

```bash
# Delete dataset
curl -X DELETE "http://localhost:8000/datasets/{dataset_id}"
```

```python
# Delete dataset and associated files
response = await client.delete(f"http://localhost:8000/datasets/{dataset_id}")
print("Dataset deleted successfully")
```

## 🔧 Response Structure

### Dataset Record

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "my_training_dataset",
  "file_name": "my_training_dataset_uuid.jsonl",
  "classification": "SENTENCE_TRIPLES",
  "source": "LOCAL",
  "rows": 1500,
  "version": 0,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Upload Response

```json
{
  "successful_uploads": 2,
  "failed": [
    {
      "filename": "invalid_dataset.csv",
      "error": "Missing required columns: ['positive', 'negative']"
    }
  ]
}
```

### HuggingFace Upload Task

```json
{
  "id": "task-uuid",
  "tag": "Anthropic/hh-rlhf",
  "task_status": "R",
  "created_at": "2024-01-15T10:30:00Z",
  "end_date": null,
  "error_msg": null
}
```

## 🎯 Common Use Cases

### Training Pipeline Setup

```python
async def setup_training_data():
    """Upload and prepare datasets for training."""

    # Upload local training data
    with open("train_data.csv", "rb") as f:
        response = await client.post(
            "http://localhost:8000/datasets",
            files={"files": ("train_data.csv", f, "text/csv")}
        )

    local_dataset_id = response.headers["Location"].split("/")[-1]

    # Download HuggingFace validation data
    hf_response = await client.post(
        "http://localhost:8000/datasets/huggingface",
        json={"dataset_tag": "squad"}
    )

    # Wait for HF download to complete
    task_id = hf_response.headers["Location"].split("/")[-1]
    await wait_for_upload(task_id)

    print(f"✅ Training setup complete!")
    print(f"Local dataset: {local_dataset_id}")
    print(f"HF task: {task_id}")
```

### Batch Data Processing

```python
async def process_data_archive():
    """Process a ZIP archive containing multiple datasets."""

    # Upload ZIP file
    with open("datasets.zip", "rb") as f:
        response = await client.post(
            "http://localhost:8000/datasets",
            files={"files": ("datasets.zip", f, "application/zip")}
        )

    result = response.json()
    print(f"📦 Processed {result['successful_uploads']} files")

    if result.get('failed'):
        print("❌ Failed uploads:")
        for failure in result['failed']:
            print(f"  - {failure['filename']}: {failure['error']}")
```

### Dataset Quality Check

```python
async def validate_dataset_quality(dataset_id: str):
    """Check dataset quality and structure."""

    response = await client.get(f"http://localhost:8000/datasets/{dataset_id}")
    dataset = response.json()

    print(f"📊 Dataset Analysis: {dataset['name']}")
    print(f"📈 Rows: {dataset['rows']:,}")
    print(f"🏷️ Classification: {dataset['classification']}")
    print(f"📅 Created: {dataset['created_at']}")

    if dataset['rows'] < 100:
        print("⚠️ Warning: Dataset has fewer than 100 rows")

    if dataset['classification'] != 'SENTENCE_TRIPLES':
        print("⚠️ Warning: Non-standard classification detected")
```

## 📈 Best Practices

### File Upload Optimization

- **Use ZIP archives** for multiple small files to reduce upload overhead
- **Validate data locally** before uploading to catch issues early
- **Choose appropriate file formats**: JSONL for large datasets, CSV for tabular data
- **Monitor upload progress** for large files or HuggingFace datasets

### Schema Design

- **Use consistent column names** across datasets for easier processing
- **Include all required columns**: question, positive, negative
- **Avoid empty values** in critical columns
- **Validate schema** before uploading large datasets

### HuggingFace Integration

- **Check dataset schemas** on HuggingFace Hub before uploading
- **Monitor background tasks** for large dataset downloads
- **Use dataset tags carefully** - ensure they exist on HuggingFace Hub
- **Plan for multiple datasets** when working with complex HF datasets

## 🚨 Error Handling

### Common Upload Errors

| Error                    | Cause                     | Solution                                             |
| ------------------------ | ------------------------- | ---------------------------------------------------- |
| `UnsupportedFormatError` | File format not supported | Use supported formats (CSV, JSON, JSONL, XML, Excel) |
| `EmptyFileError`         | File contains no data     | Ensure file has valid content                        |
| `InvalidCSVFormatError`  | Missing required columns  | Include question, positive, negative columns         |
| `FileTooLargeError`      | File exceeds size limit   | Split large files or adjust size limits              |
| `TooManyFilesError`      | ZIP has too many files    | Reduce file count or adjust ZIP limits               |

### HuggingFace Errors

| Error                               | Cause                           | Solution                                 |
| ----------------------------------- | ------------------------------- | ---------------------------------------- |
| `DatasetNotFoundError`              | Dataset doesn't exist on HF Hub | Verify dataset tag exists                |
| `UnsupportedHuggingFaceFormatError` | Schema not supported            | Check supported schemas in configuration |
| `DatasetAlreadyExistsError`         | Dataset already uploaded        | Use existing dataset or delete first     |

---

The Dataset module provides robust tools for managing training data in Vectorize. Whether uploading local files or downloading from HuggingFace, the system ensures consistent formatting and seamless integration with training workflows.
