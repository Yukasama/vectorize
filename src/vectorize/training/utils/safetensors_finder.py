import os

def find_safetensors_file(model_dir):
    """Recursively search for a .safetensors file in a directory tree (Huggingface snapshot compatible)."""
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".safetensors"):
                return os.path.join(root, file)
    return None
