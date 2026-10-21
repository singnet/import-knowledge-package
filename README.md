# Import Knowledge and Memory Portability (`import-kb`)

A utility package for importing distilled knowledge into a ChromaDB-based Long-Term Memory (LTM) system and backing up or restoring Omega user memory.

## Purpose
The `import-kb` package is designed to bridge the gap between static knowledge files (JSONL, MeTTa) and an active agent's memory. It processes structured knowledge, generates vector embeddings, and upserts them into a ChromaDB collection, enabling semantic search and retrieval for AI agents.

The package also provides `memory_portability`, a programmatic interface for exporting and restoring Omega conversation history and user LTM records. Omega Core remains responsible for its CLI, container lifecycle, transfer-directory mount, and user-facing decisions.

## Supported Embedding Models
This package supports the following embedding modes:

- **OpenAI (Cloud)**:
  - Default model: `text-embedding-3-large`
  - High accuracy but requires an internet connection and an API key.
- **ASI Cloud**:
  - Default model: `WhereIsAI/UAE-Large-V1`
  - Uses an OpenAI-compatible endpoint hosted by ASI Cloud, requires an API key.
- **SentenceTransformers (Local)**:
  - Default model: `intfloat/e5-large-v2`
  - Runs fully offline on your local machine.
  - Can be configured to use any model compatible with the `sentence-transformers` library (e.g., `all-MiniLM-L6-v2`).

## Installation

You can install the package directly from PyPI:

```bash
pip install import-kb
```

Or install it locally in editable mode:

```bash
git clone <repository-url>
cd import-knowledge-package
pip install -e .
```

## Setup
Create a `.env` file in your project root or set the following environment variables:

- `OPENAI_API_KEY`: Required if using OpenAI embeddings.
- `ASI_API_KEY`: Required if using ASI Cloud embeddings (`--provider asicloud`).
- `CHROMA_DB_PATH`: (Optional) Custom path to your Chroma database. Defaults to looking for `/PeTTa/chroma_db` or a local `chroma_db` folder.

## How to Run

### Command Line Interface (CLI)
After installation, you can run the import via the provided entry point:

```bash
# Use OpenAI embeddings (default)
import-knowledge

# Same as above, made explicit
import-knowledge --provider openai

# Override OpenAI model
import-knowledge --provider openai --model "text-embedding-3-small"

# Use ASI Cloud embeddings
import-knowledge --provider asicloud

# Override the ASI Cloud model
import-knowledge --provider asicloud --model "<model-name>"

# Use Local embeddings
import-knowledge --local

# Use a specific local model
import-knowledge --local --model "all-MiniLM-L6-v2"
```

Alternatively, run it as a module:
```bash
python3 -m import_knowledge.import_knowledge --local
```

### Programmatic Usage
You can initialize the embedding system and trigger the import programmatically from your Python scripts:

```python
from import_knowledge import initLocalEmbedding, main

# Initialize for local use
initLocalEmbedding(model_name="intfloat/e5-large-v2")

# Run the import process
main()
```

### Memory portability

The host application owns memory configuration and must pass the resolved paths
to the package explicitly:

```python
from pathlib import Path

from memory_portability import MemoryStore, MemoryTransfer

store = MemoryStore(
    memory_dir=Path("/path/to/omega/memory"),
    chroma_path=Path("/path/to/chroma_db"),
    collection_name="memories",
)
transfer = MemoryTransfer(
    transfer_dir=Path("/path/to/memory-transfer"),
    store=store,
)

exported = transfer.export(component="all")
transfer.import_archive(exported["filename"])
transfer.recover()
```

`MemoryStore` does not infer Omega paths or read them from environment
variables. Resolve these values in the host application's configuration layer.

Default exports use `omega-memory-<UTC timestamp>-<ID>.tar.gz`, where the
ID is the first eight hexadecimal characters of a UUIDv4.

## Dependencies
- `openai`: For cloud-based embeddings.
- `sentence-transformers`: For local, offline embeddings.
- `chromadb`: Vector database for storage.
- `python-dotenv`: Management of environment variables.
- `tqdm`: Progress bars for batch processing.

---

## Publishing to PyPI

The [publishing workflow](.github/workflows/publish.yml) builds and validates a
source distribution and wheel when a GitHub Release is published, then uploads
them to PyPI using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
No PyPI API token or GitHub secret is needed.

One-time setup:

1. Create a GitHub environment named `pypi` in the repository settings. Configure
   required reviewers if releases should require approval before uploading.
2. In the PyPI `import-kb` project's **Publishing** settings, add a GitHub trusted
   publisher with the following values:
   - **Owner:** the GitHub user or organization that will publish releases
     (`blackhammer116` for this fork, or `singnet` for the upstream repository).
   - **Repository:** `import-knowledge-package`.
   - **Workflow filename:** `publish.yml`.
   - **Environment:** `pypi`.

To release, update `project.version` in `pyproject.toml` to an unused version,
commit the change, and publish a GitHub Release tagged with that version
(for example, `v0.2.5` or `0.2.5`). The tag must point to the commit containing
the version update and this workflow. Draft releases and tag pushes alone do
not publish; publishing a GitHub prerelease also triggers the workflow, so use
a corresponding Python prerelease version such as `0.3.0rc1`.

The workflow rejects tags that do not match `project.version`. PyPI does not
allow an existing release file to be overwritten; use a new version for each
release.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.
