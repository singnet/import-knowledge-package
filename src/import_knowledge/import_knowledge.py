import os
import json
from pathlib import Path
import chromadb
import argparse
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import torch

# Load API Key from .env
load_dotenv()

# Look for the DB path
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
EMBEDDING_MODE = "openai" # "openai", "asicloud", or "local"
EMBEDDING_MODEL = "text-embedding-3-large"
LOCAL_MODEL_NAME = "intfloat/e5-large-v2"

# Cloud embedding providers. Each maps to an OpenAI-compatible endpoint.
PROVIDERS = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "text-embedding-3-large",
    },
    "asicloud": {
        "base_url": "https://inference.asicloud.cudos.org/v1",
        "api_key_env": "ASI_API_KEY",
        "default_model": "WhereIsAI/UAE-Large-V1",
    },
}
COLLECTION_NAME = "memories"
KNOWLEDGE_FILES = [
    os.path.join(_PACKAGE_DIR, "KB", "oma_distilled_knowledge.jsonl"),
    os.path.join(_PACKAGE_DIR, "KB", "max_distilled_knowledge.jsonl")
]
CURRICULUM_FILE = os.path.join(_PACKAGE_DIR, "KB", "curriculum.metta")

DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    "/PeTTa/chroma_db" if os.path.isdir("/PeTTa/chroma_db") else
    os.path.join(_PACKAGE_DIR, "..", "..", "chroma_db")
)
LOCAL_BATCH_SIZE = int(os.environ.get("LOCAL_EMBED_BATCH_SIZE", "128"))

LOCAL_DEVICE = os.environ.get(
    "LOCAL_EMBED_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
)

_embedding_model = None
_openai_client = None

def init_embeddings(mode="openai", model_name=None):
    """Initialize the embedding system.

    Can be called programmatically after import to switch modes.

    Args:
        mode: "local" for offline SentenceTransformers, or a provider id
            from PROVIDERS (e.g. "openai", "asicloud") for a cloud API.
        model_name: Overrides the default model for the selected mode.

    Raises:
        ValueError: If mode is not "local" and not a known provider id.
        RuntimeError: If the provider's required API key environment
            variable is not set.
    """
    global EMBEDDING_MODE, EMBEDDING_MODEL, LOCAL_MODEL_NAME, _embedding_model, _openai_client
    EMBEDDING_MODE = mode

    if mode == "local":
        if model_name:
            LOCAL_MODEL_NAME = model_name
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local SentenceTransformer model: {LOCAL_MODEL_NAME}...")
            _embedding_model = SentenceTransformer(LOCAL_MODEL_NAME)
    elif mode == "openai":
        EMBEDDING_MODEL = model_name or PROVIDERS["openai"]["default_model"]
        if _openai_client is None:
            _openai_client = openai.OpenAI()
    elif mode in PROVIDERS:
        provider = PROVIDERS[mode]
        EMBEDDING_MODEL = model_name or provider["default_model"]
        if _openai_client is None:
            api_key = os.environ.get(provider["api_key_env"])
            if not api_key:
                raise RuntimeError(
                    f"Missing required environment variable {provider['api_key_env']} "
                    f"for provider '{mode}'"
                )
            _openai_client = openai.OpenAI(api_key=api_key, base_url=provider["base_url"])
    else:
        raise ValueError(f"Unknown embedding provider: {mode!r}")

def initLocalEmbedding(model_name=None):
    """Convenience function matching user snippet."""
    init_embeddings(mode="local", model_name=model_name)
    return _embedding_model

def embed_batch(texts):
    """Embed a list of texts. Uses either OpenAI or local SentenceTransformer."""
    global EMBEDDING_MODE, EMBEDDING_MODEL, _embedding_model, _openai_client
    
    if EMBEDDING_MODE == "local":
        # Ensure model is loaded (handles case where init_embeddings wasn't called)
        if _embedding_model is None:
            init_embeddings(mode="local")
        return _embedding_model.encode(texts).tolist()
    else:
        # OpenAI mode
        if _openai_client is None:
            _openai_client = openai.OpenAI()
        resp = _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in resp.data]

def main():
    parser = argparse.ArgumentParser(description="Import knowledge into ChromaDB.")
    parser.add_argument("--local", action="store_true", help="Use local SentenceTransformer embeddings")
    parser.add_argument(
        "--provider",
        type=str,
        choices=sorted(PROVIDERS),
        default="openai",
        help="Cloud embedding provider to use (ignored when --local is set)",
    )
    parser.add_argument("--model", type=str, help="Override default model name (OpenAI or local)")
    args = parser.parse_args()

    # Initialize based on arguments
    mode = "local" if args.local else args.provider
    init_embeddings(mode=mode, model_name=args.model)

    has_any_knowledge = any(Path(f).exists() for f in KNOWLEDGE_FILES)
    if not has_any_knowledge and not Path(CURRICULUM_FILE).exists():
        print("Error: Neither knowledge nor curriculum files were found.")
        return

    print(f"Connecting to Agent LTM at: {DB_PATH}")
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    ids = []
    documents = []
    metadatas = []
    seen_ids = set()

    for k_file in KNOWLEDGE_FILES:
        if Path(k_file).exists():
            print(f"Loading knowledge...")
            with open(k_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error: Failed to parse line in {k_file}: {e}")
                        print(f"Line content: {line.strip()}")
                        continue
                    
                    record_id = record.get("id")
                    if not record_id:
                        print(f"Warning: Missing 'id' in record from {k_file}. Skipping.")
                        continue
                    
                    if record_id in seen_ids:
                        continue
                        
                    seen_ids.add(record_id)
                    ids.append(record_id)
                    documents.append(record["document"])

                    
                    meta = record.get("metadata", {})
                    clean_meta = {
                        "source": "distilled_memory",
                        "breadcrumb": f"LTM > {meta.get('domain', 'general')} > {meta.get('type', 'fact')}",
                        "type": "chunk",
                        "time": "knowledge_prior"
                    }
                    
                    for k, v in meta.items():
                        if isinstance(v, list):
                            clean_meta[k] = " | ".join(v) if v else "None"
                        else:
                            clean_meta[k] = v
                    
                    metadatas.append(clean_meta)
        else:
            print(f"Warning: {k_file} not found. Skipping...")

    if Path(CURRICULUM_FILE).exists():
        print("Loading curriculum...")
        with open(CURRICULUM_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            
            for idx, chunk in enumerate(chunks):
                curriculum_id = f"curriculum_mem_{idx}"
                
                if curriculum_id in seen_ids:
                    continue
                    
                seen_ids.add(curriculum_id)
                ids.append(curriculum_id)
                documents.append(chunk)
                metadatas.append({
                    "source": "curriculum",
                    "breadcrumb": "LTM > curriculum",
                    "type": "chunk",
                    "time": "knowledge_prior"
                })
    else:
        print(f"Warning: {CURRICULUM_FILE} not found. Skipping...")



    count = len(ids)
    if count == 0:
        print("No documents to process.")
        return

    current_model = LOCAL_MODEL_NAME if EMBEDDING_MODE == "local" else EMBEDDING_MODEL
    print(f"Generating {EMBEDDING_MODE} '{current_model}' embeddings for {count} records. Please wait...")

    batch_size = 500
    with tqdm(total=count, desc="Upserting Knowledge") as pbar:
        for i in range(0, count, batch_size):
            end = min(i + batch_size, count)
            
            batch_docs = documents[i:end]
            batch_ids = ids[i:end]
            batch_metas = metadatas[i:end]
            
            try:
                batch_embeddings = embed_batch(batch_docs)
            except Exception as e:
                print(f"Fatal error generating embeddings: {e}")
                return
            
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_metas
            )
            pbar.update(len(batch_docs))

    print("\nKnowledge Transfer Complete!")
    print(f"New Agent Database now has {collection.count()} total memories.")

if __name__ == "__main__":
    main()