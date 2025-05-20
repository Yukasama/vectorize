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
from .repository import (
    get_all_datasets_db,
    get_dataset_db,
    save_dataset_db,
    update_dataset_db,
)
from .router import router
from .service import (
    get_dataset_srv,
    get_datasets_srv,
    update_dataset_srv,
    upload_file_srv,
)
from .upload_options_model import DatasetUploadOptions
from .utils.csv_escaper import _escape_csv_formulas
from .utils.dataset_classifier import _classify_dataset
from .utils.file_df_converter import _convert_file_to_df
from .utils.file_loaders import _load_file
from .utils.file_size_formatter import _format_file_size
from .utils.save_dataset import _save_dataframe_to_fs
from .utils.validate_zip import _validate_zip_file

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
    "_classify_dataset",
    "_convert_file_to_df",
    "_escape_csv_formulas",
    "_format_file_size",
    "_load_file",
    "_save_dataframe_to_fs",
    "_validate_zip_file",
    "get_all_datasets_db",
    "get_dataset_db",
    "get_dataset_srv",
    "get_datasets_srv",
    "router",
    "save_dataset_db",
    "update_dataset_db",
    "update_dataset_srv",
    "upload_file_srv",
]
