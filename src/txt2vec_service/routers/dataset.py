import os
import tempfile
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/dataset", tags=["dataset"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DatasetResponse(BaseModel):
    filename: str
    rows: int
    columns: List[str]
    preview: List[dict]


@router.post("/", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    delimiter: str = ",",
    sheet_name: Optional[int] = 0,
):
    """
    Upload a dataset file (CSV, JSON, or XML) and convert it to CSV format.

    Parameters:
    - file: The file to upload
    - delimiter: Delimiter for CSV files (default: comma)
    - sheet_name: Sheet index for Excel files (default: 0)

    Returns:
    - Information about the processed dataset
    """
    # Check file extension
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_extension = filename.split(".")[-1].lower()

    if file_extension not in ["csv", "json", "xml", "xlsx", "xls"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a CSV, JSON, XML, or Excel file.",
        )

    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    try:
        # Load the file into a pandas DataFrame based on its format
        if file_extension == "csv":
            df = pd.read_csv(temp_path, delimiter=delimiter)
        elif file_extension == "json":
            df = pd.read_json(temp_path)
        elif file_extension == "xml":
            df = pd.read_xml(temp_path)
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(temp_path, sheet_name=sheet_name)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Generate a unique filename for the saved CSV
        base_name = os.path.splitext(filename)[0]
        csv_filename = f"{base_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = UPLOAD_DIR / csv_filename

        # Save as CSV
        df.to_csv(csv_path, index=False)

        # Prepare the response
        response = DatasetResponse(
            filename=csv_filename,
            rows=len(df),
            columns=df.columns.tolist(),
            preview=df.head(5).to_dict(orient="records"),
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)


@router.get("/", response_model=List[str])
async def list_datasets():
    """List all available datasets"""
    try:
        files = [f.name for f in UPLOAD_DIR.glob("*.csv")]
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@router.get("/{filename}", response_model=DatasetResponse)
async def get_dataset(filename: str):
    """Get information about a specific dataset"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists() or not filename.endswith(".csv"):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(file_path)
        return DatasetResponse(
            filename=filename,
            rows=len(df),
            columns=df.columns.tolist(),
            preview=df.head(5).to_dict(orient="records"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dataset: {str(e)}"
        )


@router.delete("/{filename}")
async def delete_dataset(filename: str):
    """Delete a dataset"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists() or not filename.endswith(".csv"):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        os.remove(file_path)
        return JSONResponse(
            content={"message": f"Dataset {filename} deleted successfully"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")
