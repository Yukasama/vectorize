from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import snapshot_download
from huggingface_hub import HfFolder

# Globale Pipeline (wird beim Laden gesetzt)
classifier = None

def load_model_with_tag(model_id: str, tag: str):
    global classifier

    # Lade lokalen Snapshot vom HF-Model mit Tag
    snapshot_path = snapshot_download(repo_id=model_id, revision=tag)

    # Lade Tokenizer und Model
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)

    # Erstelle Pipeline
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_classifier():
    if classifier is None:
        raise ValueError("Kein Modell geladen.")
    return classifier
