"""Dataset module."""

from .classification import Classification
from .column_mapper import ColumnMapping
from .exceptions import (
    DatasetNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    InvalidCSVColumnError,
    InvalidCSVFormatError,
    InvalidFileError,
    MissingColumnError,
    TooManyFilesError,
    UnsupportedFormatError,
)
from .file_format import FileFormat
from .models import Dataset, DatasetAll, DatasetCreate, DatasetPublic, DatasetUpdate
from .repository import get_all_datasets, get_dataset, save_dataset, update_dataset
from .router import router
from .service import read_all_datasets, read_dataset, update_dataset_srv, upload_file
from .upload_options_model import DatasetUploadOptions
from .utils.csv_escaper import escape_csv_formulas
from .utils.dataset_classifier import classify_dataset
from .utils.file_df_converter import convert_file_to_df
from .utils.file_loaders import load_file
from .utils.file_size_formatter import format_file_size
from .utils.save_dataset import save_dataframe
from .utils.validate_zip import validate_zip_file

__all__ = [
    "Classification",
    "ColumnMapping",
    "Dataset",
    "DatasetAll",
    "DatasetCreate",
    "DatasetNotFoundError",
    "DatasetPublic",
    "DatasetUpdate",
    "DatasetUploadOptions",
    "EmptyFileError",
    "FileFormat",
    "FileTooLargeError",
    "InvalidCSVColumnError",
    "InvalidCSVFormatError",
    "InvalidFileError",
    "MissingColumnError",
    "TooManyFilesError",
    "UnsupportedFormatError",
    "classify_dataset",
    "convert_file_to_df",
    "escape_csv_formulas",
    "format_file_size",
    "get_all_datasets",
    "get_dataset",
    "load_file",
    "read_all_datasets",
    "read_dataset",
    "router",
    "save_dataframe",
    "save_dataset",
    "update_dataset",
    "update_dataset_srv",
    "upload_file",
    "validate_zip_file",
]
