# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository contains pipelines for ingesting data from various sources into vector databases to be used by Composer AI. The primary focus is on:

1. Processing Red Hat product documentation
2. Processing website content
3. Processing Open Project Library (OPL) content
4. Converting content to embeddings and storing in vector databases (Elasticsearch and Weaviate)

## Project Structure

The project is organized into Kubeflow Pipelines (KFP) for different data sources:

- `kfp/redhat-product-documentation-ingestor/`: Pipelines for ingesting Red Hat product documentation
- `kfp/website-ingestor/`: Pipelines for ingesting website content
- `kfp/opl-ingestor/`: Pipelines for ingesting Open Project Library content
- `kfp/pipeline/`: YAML definitions for Kubeflow pipelines
- `pipeline/`: Additional pipeline YAML definitions

Each directory contains multiple versions of pipelines:

- Production pipeline files (e.g., `ingestion-pipeline.py`, `website-ingestion-pipeline.py`)
- Local execution versions (e.g., `ingestion-pipeline-local.py`, `ingestion-pipeline-website-local.py`)
- Elasticsearch-specific versions (e.g., `ingestion-pipeline-elastic.py`)
- OPL-specific versions (e.g., `ingestion-pipeline-elastic-opl.py`)

## Development Commands

### Linting

```bash
# Check code with ruff
ruff check .

# Format code with ruff
ruff format .
```

### Running Local Pipelines

To run a document ingestion pipeline locally (using Elasticsearch):

```bash
# Set Elasticsearch credentials
export ES_USER=your_username
export ES_PASS=your_password
export ES_HOST=http://your_elasticsearch_host:9200

# Run Red Hat documentation ingestion pipeline
python kfp/redhat-product-documentation-ingestor/ingestion-pipeline-local.py
```

To run website ingestion locally:

```bash
# Set Elasticsearch credentials
export ES_USER=your_username
export ES_PASS=your_password
export ES_HOST=http://your_elasticsearch_host:9200

# Run website ingestion pipeline
export WEBSITE_URL=https://example.com
export VECTORDB_INDEX=index_name
python kfp/website-ingestor/ingestion-pipeline-website-local.py
```

To run OPL ingestion locally:

```bash
# Set Elasticsearch credentials
export ES_USER=your_username
export ES_PASS=your_password
export ES_HOST=http://your_elasticsearch_host:9200

# Run OPL ingestion pipeline
python kfp/opl-ingestor/opl-injestion-pipeline-elastic.py
```

### Running on Kubeflow

To compile and submit pipelines to Kubeflow:

```bash
# Set Kubeflow endpoint
export KUBEFLOW_ENDPOINT=https://your_kubeflow_endpoint
export BEARER_TOKEN=your_bearer_token

# Run Red Hat documentation ingestion pipeline
python kfp/redhat-product-documentation-ingestor/ingestion-pipeline.py

# Run website ingestion pipeline
export WEBSITE_URL=https://example.com
export VECTORDB_INDEX=index_name
python kfp/website-ingestor/website-ingestion-pipeline.py

# Run OPL ingestion pipeline
python kfp/opl-ingestor/ingestion-pipeline.py
```

### Running with Weaviate

For pipelines that use Weaviate instead of Elasticsearch:

```bash
# Set Weaviate credentials
export WEAVIATE_API_KEY=your_api_key
export WEAVIATE_HOST=https://your_weaviate_host
export WEAVIATE_PORT=your_weaviate_port

# Run Weaviate-backed pipeline
python kfp/website-ingestor/ingestion-pipeline-website-local.py
```

## Key Pipeline Components

1. **Document Loading**: Fetches documents/websites from source URLs
2. **Document Processing**:
   - Extracts meaningful content (removes unwanted sections)
   - Converts HTML to Markdown
   - Splits content into semantically meaningful chunks
3. **Embedding Generation**: Creates vector embeddings using Hugging Face models (primarily nomic-ai/nomic-embed-text-v1)
4. **Vector Database Storage**: Stores chunks and embeddings in Elasticsearch or Weaviate

## Core Ingestion Flow

For each data source, the ingestion pipeline follows these general steps:

1. **Source Acquisition**: Fetches content from the specified URL or file source
2. **HTML Processing**: For web content, extracts relevant sections using BeautifulSoup
3. **Markdown Conversion**: Transforms HTML content to structured markdown
4. **Document Preparation**: Organizes the content with proper metadata for vector database storage
5. **Embedding Generation**: Generates vector embeddings using Hugging Face models
6. **Database Ingestion**: Uploads documents with embeddings to Elasticsearch/Weaviate in batches

## Dependencies

The pipelines depend on:

- LangChain for document processing and vector store integration
- HuggingFace for embedding generation (primarily nomic-ai/nomic-embed-text-v1 model)
- BeautifulSoup for HTML parsing and content extraction
- Elasticsearch/Weaviate as vector stores
- Kubeflow Pipelines (KFP) for orchestration
- Required Python packages are specified in requirements.txt within each pipeline directory
