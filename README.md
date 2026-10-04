# Local RAG Pipeline with Modular Design

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline that allows for configurable chunking strategies, retrieval, and reranking. The pipeline is designed to be easily extensible and production-ready.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [As a Command-Line Tool](#as-a-command-line-tool)
  - [As a Python Module](#as-a-python-module)
- [Chunking Strategies](#chunking-strategies)
- [Reranking](#reranking)
- [Example](#example)
- [License](#license)

## Features
- **Modular Design**: Separation of concerns into services (chunking, retrieval, reranking) and configuration.
- **Configurable Chunking**: Supports multiple chunking strategies:
  - Fixed Size
  - Recursive
  - Document-based (by paragraphs)
  - Semantic (placeholder)
  - Propositional (placeholder)
  - Agentic (placeholder)
- **Configurable Retrieval**: Uses a vector store (Chroma) for efficient similarity search.
- **Reranking**: Optionally reranks retrieved documents using embedding similarity to improve relevance.
- **YAML Configuration**: All settings can be configured via a YAML file.
- **Command-Line Interface**: Easy to use from the terminal for ingestion and querying.
- **Python Package**: Can be installed and used as a module in other Python projects.

## Project Structure
```
local-rag-pipeline/
├── data/                         # Data directory (CSV files, vector store)
├── rag_pipeline/                 # Main package
│   ├── __init__.py
│   ├── config/                   # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── pipeline.py               # Main RAG pipeline class
│   ├── cli.py                    # Command-line interface
│   └── services/                 # Services for different functionalities
│       ├── chunking/             # Chunking strategies
│       │   ├── __init__.py
│       │   └── strategies.py
│       ├── retrieval/            # Retrieval from vector store
│       │   ├── __init__.py
│       │   └── retriever.py
│       └── reranking/            # Reranking of retrieved documents
│           ├── __init__.py
│           └── reranker.py
├── pyproject.toml                # Project metadata and dependencies
├── README.md
├── requirements.txt              # Legacy requirements (for backward compatibility)
└── LICENSE
```

## Installation
You can install the package in development mode using:

```bash
pip install -e .
```

Or install from the built distribution:

```bash
pip install local-rag-pipeline
```

## Configuration
The pipeline can be configured via a YAML file or by using the default settings. The configuration options are:

| Setting | Description | Default |
|---------|-------------|---------|
| `data_path` | Path to the data directory | `./data` |
| `vector_db_path` | Path to the vector store directory | `./data/chrome_langchain_db` |
| `collection_name` | Name of the collection in the vector store | `restaurant_reviews` |
| `embedding_model` | Name of the embedding model (Ollama) | `mxbai-embed-large` |
| `llm_model` | Name of the language model (Ollama) | `llama3.2` |
| `chunking_strategy` | Strategy for chunking documents | `recursive` |
| `chunk_size` | Size of each chunk (in characters) | `1000` |
| `chunk_overlap` | Overlap between chunks (in characters) | `200` |
| `k` | Number of documents to retrieve initially | `5` |
| `use_reranking` | Whether to enable reranking | `True` |
| `reranker_type` | Type of reranker to use | `embedding` |
| `rerank_top_k` | Number of documents to keep after reranking | `3` |

To create a configuration file, you can run:

```bash
local-rag --config config.yaml
```

This will create a default configuration file at `config.yaml` (if you specify a path) or you can manually create one. An example configuration file:

```yaml
data_path: "./data"
vector_db_path: "./data/chrome_langchain_db"
collection_name: "restaurant_reviews"
embedding_model: "mxbai-embed-large"
llm_model: "llama3.2"
chunking_strategy: "recursive"
chunk_size: 1000
chunk_overlap: 200
k: 5
use_reranking: true
reranker_type: "embedding"
rerank_top_k: 3
```

## Usage

### As a Command-Line Tool
After installation, you can use the `local-rag` command.

#### Ingesting Data
To ingest data from a CSV file (expecting columns: `Title`, `Review`, etc.):

```bash
local-rag --data path/to/your/data.csv
```

#### Asking a Question
To ask a question and get the relevant context:

```bash
local-rag --question "What is the best pizza in town?"
```

#### Interactive Mode
To run in interactive mode (ask multiple questions):

```bash
local-rag
```

#### Overriding Settings
You can override settings from the command line:

```bash
local-rag --data data.csv --k 10 --chunking-strategy fixed_size --no-reranking
```

### As a Python Module
You can also use the pipeline in your own Python code:

```python
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.config.settings import Settings

# Use default settings
pipeline = RAGPipeline()

# Or use custom settings
settings = Settings(
    chunking_strategy="fixed_size",
    k=10,
    use_reranking=False
)
pipeline = RAGPipeline(settings=settings)

# Add documents (from a list of langchain_core.documents.Document)
documents = [...]  # Your documents
pipeline.add_documents(documents)

# Retrieve context for a query
context = pipeline.get_context("What is the best pizza in town?")
print(context)
```

## Chunking Strategies
The following chunking strategies are available:

1. **fixed_size**: Splits text into chunks of a fixed size with optional overlap.
2. **recursive**: Splits text by recursively trying different separators (default: `["\n\n", "\n", " ", ""]`).
3. **document**: Splits text by double newline (paragraphs) and then further splits if needed.
4. **semantic**: Placeholder for semantic chunking (currently splits by sentence and groups).
5. **propositional**: Placeholder for propositional chunking (currently similar to semantic).
6. **agentic**: Placeholder for agentic chunking (returns the whole text as one chunk).

## Reranking
The pipeline supports reranking of retrieved documents using embedding similarity. The reranker computes the cosine similarity between the query embedding and each document embedding, then returns the top `k` documents after reranking.

To disable reranking, set `use_reranking: false` in the configuration or use the `--no-reranking` flag.

## Example
See the `realistic_restaurent_reviews.csv` file in the `data` directory for an example dataset. The pipeline can be run on this data as follows:

```bash
# Ingest the example data
local-rag --data data/realistic_restaurent_reviews.csv

# Ask a question
local-rag --question "What is the best pizza in town?"
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.