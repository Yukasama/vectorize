"""Dataset module for Vectorize.

Provides all functionality for uploading, validating, processing, and managing datasets.
This module supports a wide range of file formats, schema mapping, HuggingFace
integration, background processing, and unified API endpoints for dataset workflows.

Key Features:
- Multi-format upload: CSV, JSON, JSONL, XML, Excel, and ZIP archives
- Automatic schema mapping and validation for question/positive/negative columns
- HuggingFace dataset integration with schema filtering and column renaming
- Batch upload and ZIP archive extraction with configurable limits
- Background processing for uploads and HuggingFace downloads
- CRUD operations for datasets with pagination and metadata
- Error handling for unsupported formats, missing columns, and file size/count limits
- Unified API endpoints for all dataset operations
"""

from .classification import Classification
from .column_mapper import ColumnMapping
from .exceptions import (
    DatasetAlreadyExistsError,
    DatasetIsAlreadyBeingUploadedError,
    DatasetNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    HuggingFaceDatasetNotFoundError,
    InvalidCSVColumnError,
    InvalidCSVFormatError,
    InvalidXMLFormatError,
    MissingColumnError,
    TooManyFilesError,
    UnsupportedFormatError,
    UnsupportedHuggingFaceFormatError,
)
from .file_format import FileFormat
from .models import Dataset, DatasetAll, DatasetCreate, DatasetPublic, DatasetUpdate
from .repository import (
    delete_dataset_db,
    find_dataset_by_name_db,
    get_dataset_db,
    get_datasets_db,
    get_upload_dataset_task_db,
    is_dataset_being_uploaded_db,
    save_upload_dataset_task_db,
    update_dataset_db,
    update_upload_task_status,
    upload_dataset_db,
)
from .router import router
from .schemas import DatasetUploadOptions
from .service import (
    delete_dataset_svc,
    get_dataset_svc,
    get_datasets_svc,
    get_hf_upload_status_svc,
    update_dataset_svc,
    upload_dataset_svc,
    upload_hf_dataset_svc,
)

__all__ = [
    "Classification",
    "ColumnMapping",
    "Dataset",
    "DatasetAll",
    "DatasetAlreadyExistsError",
    "DatasetCreate",
    "DatasetIsAlreadyBeingUploadedError",
    "DatasetNotFoundError",
    "DatasetPublic",
    "DatasetUpdate",
    "DatasetUploadOptions",
    "EmptyFileError",
    "FileFormat",
    "FileTooLargeError",
    "HuggingFaceDatasetNotFoundError",
    "InvalidCSVColumnError",
    "InvalidCSVFormatError",
    "InvalidXMLFormatError",
    "MissingColumnError",
    "TooManyFilesError",
    "UnsupportedFormatError",
    "UnsupportedHuggingFaceFormatError",
    "delete_dataset_db",
    "delete_dataset_svc",
    "find_dataset_by_name_db",
    "get_dataset_db",
    "get_dataset_svc",
    "get_datasets_db",
    "get_datasets_svc",
    "get_hf_upload_status_svc",
    "get_upload_dataset_task_db",
    "is_dataset_being_uploaded_db",
    "router",
    "save_upload_dataset_task_db",
    "update_dataset_db",
    "update_dataset_svc",
    "update_upload_task_status",
    "upload_dataset_db",
    "upload_dataset_svc",
    "upload_hf_dataset_svc",
]
