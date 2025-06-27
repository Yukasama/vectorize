# Configuration

## Environment Variables

Copy the `.env.example` file and configure your environment:

```sh
cp .env.example .env
```

## Configuration File

The main configuration is stored in `src/vectorize/config/resources/app.toml`. Below are all available configuration options:

### Server Configuration (`[app.server]`)

Controls the FastAPI server behavior:

- **`host_binding`** (default: `"127.0.0.1"`): Host address the server binds to
  - Set to `"0.0.0.0"` to accept connections from all interfaces
  - Set to `"127.0.0.1"` for localhost only
- **`port`** (default: `8000`): Network port the server listens on
- **`version`** (default: `"0.1.0"`): Application version string
- **`reload`** (default: `true`): Enable auto-reload on code changes (development only)

### Database Configuration (`[app.db]`)

SQLite database settings:

- **`logging`** (default: `false`): Enable SQL query logging for debugging
- **`timeout`** (default: `30`): Database operation timeout in seconds
- **`future`** (default: `true`): Enable SQLAlchemy 2.0 future mode
- **`max_overflow`** (default: `10`): Maximum connections beyond pool size
- **`pool_size`** (default: `5`): Base connection pool size
- **`pool_pre_ping`** (default: `true`): Validate connections before use
- **`pool_recycle`** (default: `300`): Recycle connections after N seconds
- **`pool_timeout`** (default: `30`): Timeout for acquiring connections

### Logging Configuration (`[app.logging]`)

Application logging settings:

- **`rotation`** (default: `"10 MB"`): Log file rotation trigger
  - Examples: `"10 MB"`, `"1 GB"`, `"daily"`, `"weekly"`
- **`log_dir`** (default: `"log"`): Directory for log files
- **`log_file`** (default: `"app.log"`): Log filename

### Dataset Configuration (`[app.dataset]`)

Dataset upload and processing settings:

#### File Handling

- **`upload_dir`** (default: `"data/datasets"`): Directory for uploaded datasets
- **`max_upload_size`** (default: `53687091200`): Maximum file size (50 GB in bytes)
- **`max_filename_length`** (default: `255`): Maximum filename length
- **`max_zip_members`** (default: `10000`): Maximum files in ZIP archives

#### File Formats

- **`allowed_extensions`**: Supported file types for upload
  - `csv`: Comma-separated values
  - `json`: JSON format
  - `jsonl`: JSON Lines format
  - `xml`: XML documents
  - `xlsx`, `xls`: Excel spreadsheets
- **`default_delimiter`** (default: `";"`): Default CSV delimiter

#### Hugging Face Schema Validation

- **`hf_allowed_schemas`**: Permitted column combinations for HuggingFace datasets

**Supported Schema Categories:**

1. **Standard Preference Formats**

   ```toml
   ["prompt", "chosen", "rejected"]
   ["prompt", "chosen_response", "rejected_response"]
   ["prompt", "response_chosen", "response_rejected"]
   ```

2. **Instruction-Based Formats**

   ```toml
   ["instruction", "chosen", "rejected"]
   ["instruction", "output_1", "output_2"]
   ["instruction", "output"]
   ```

3. **Query/Question Formats**

   ```toml
   ["query", "chosen", "rejected"]
   ["question", "positive", "negative"]
   ["question", "answer"]
   ```

4. **Input/Output Formats**

   ```toml
   ["input", "chosen", "rejected"]
   ["input", "response_a", "response_b"]
   ["input", "target"]
   ```

5. **Conversation Formats**
   ```toml
   ["system", "user", "chosen", "rejected"]
   ["human", "assistant"]
   ["messages", "chosen", "rejected"]
   ```

### Model Configuration (`[app.model]`)

AI model storage settings:

- **`upload_dir`** (default: `"data/models"`): Directory for uploaded models
- **`max_upload_size`** (default: `53687091200`): Maximum model size (50 GB in bytes)

### Inference Configuration (`[app.inference]`)

Model inference settings:

- **`device`** (default: `"cpu"`): Compute device for inference
  - `"cpu"`: Use CPU for inference
  - `"cuda"`: Use GPU with CUDA (if available)

### Evaluation Configuration (`[app.evaluation]`)

Model evaluation settings:

- **`default_max_samples`** (default: `1000`): Default sample size for evaluation
- **`default_random_seed`** (default: `42`): Random seed for reproducible evaluation

## Environment Variable Overrides

Several settings can be overridden with environment variables:

```sh
# Application environment
ENV=development|testing|production

# Database
DATABASE_URL=sqlite+aiosqlite:///app.db
CLEAR_DB_ON_RESTART=true

# Logging
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL
```

## Example Configuration

```toml
[app.server]
host_binding = "0.0.0.0"  # Accept external connections
port = 8080
reload = false  # Disable in production

[app.db]
logging = true  # Enable for debugging
pool_size = 10  # Increase for high load

[app.dataset]
max_upload_size = 10737418240  # 10 GB limit
default_delimiter = ","  # Use comma for CSV

[app.inference]
device = "cuda"  # Use GPU if available
```

## Validation

The configuration is validated on startup. Invalid values will cause the application to fail with descriptive error messages.
