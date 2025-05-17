"""
Elasticsearch ingestion utilities for Open Practice Library content.

This module contains functions for preparing and ingesting content
into Elasticsearch with vector embeddings.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

try:
    from elasticsearch import Elasticsearch
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_elasticsearch import ElasticsearchStore
except ImportError:
    _log = logging.getLogger(__name__)
    _log.error(
        "Required libraries not found. Please install 'elasticsearch', 'langchain', and 'langchain-elasticsearch'"
    )
    sys.exit(1)

# Configure module logger
_log = logging.getLogger(__name__)


def _load_document_splits(input_artifact):
    """Load document splits from input artifact."""
    if isinstance(input_artifact, str) or hasattr(input_artifact, "path"):
        path = input_artifact.path if hasattr(input_artifact, "path") else input_artifact
        try:
            with open(path) as input_file:
                return json.loads(input_file.read())
        except Exception as e:
            _log.error(f"Error reading input artifact: {str(e)}")
            return None
    elif isinstance(input_artifact, list):
        return input_artifact
    else:
        _log.error(f"Unsupported input_artifact type: {type(input_artifact)}")
        return None


def _init_elasticsearch_client():
    """Initialize and return Elasticsearch client."""
    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    _log.info("Environment variables for Elasticsearch:")
    _log.info(f"ES_USER: {'SET' if es_user else 'NOT SET'}")
    _log.info(f"ES_PASS: {'SET' if es_pass else 'NOT SET'}")
    _log.info(f"ES_HOST: {'SET' if es_host else 'NOT SET'}")

    if not all([es_user, es_pass, es_host]):
        _log.error("Elasticsearch config not present. Please set ES_USER, ES_PASS, and ES_HOST environment variables.")
        if __name__ != "__main__":
            return None, 1
        import sys

        sys.exit(1)

    try:
        es_client = Elasticsearch(es_host, basic_auth=(es_user, es_pass), request_timeout=30, verify_certs=False)
        health = es_client.cluster.health()
        _log.info(f"Elasticsearch cluster status: {health['status']}")
        return es_client, 0
    except Exception as e:
        _log.error(f"Failed to initialize Elasticsearch client: {str(e)}")
        return None, 1


def _init_embeddings():
    """Initialize and return embeddings model."""
    device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
    _log.info(f"Using device: {device} for embeddings")
    model_name = "nomic-ai/nomic-embed-text-v1"
    model_kwargs = {"trust_remote_code": True, "device": device}

    _log.info(f"Initializing embeddings model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        show_progress=True,
        encode_kwargs={"normalize_embeddings": True},
    )


def _process_index(index_data, es_client, embeddings):
    """Process a single index and its documents."""
    if not isinstance(index_data, dict) or "index_name" not in index_data or "splits" not in index_data:
        return

    index_name = index_data["index_name"]
    splits = index_data["splits"]
    _log.info(f"Processing index {index_name} with {len(splits)} documents")

    documents = [Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits]

    try:
        db = ElasticsearchStore(
            index_name=index_name.lower(),
            embedding=embeddings,
            es_connection=es_client,
        )

        batch_size = 50
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            _log.info(f"Uploading batch {batch_num}/{total_batches} to index {index_name}")
            db.add_documents(batch)

        _log.info(f"Successfully uploaded all documents to index {index_name}")
    except Exception as e:
        _log.error(f"Error ingesting documents: {str(e)}")
        import traceback

        traceback.print_exc()


def ingest_to_elasticsearch(input_artifact):
    """
    Function to ingest the processed content into Elasticsearch.

    This can be called directly or used as a KFP component.

    Args:
        input_artifact: Either a path to a JSON file or a list of document splits
    """
    # Load document splits
    document_splits = _load_document_splits(input_artifact)
    if not document_splits:
        return

    # Initialize Elasticsearch client
    es_client, status = _init_elasticsearch_client()
    if status != 0:
        return

    # Initialize embeddings
    try:
        embeddings = _init_embeddings()
    except Exception as e:
        _log.error(f"Failed to initialize embeddings: {str(e)}")
        return

    # Process each index
    if isinstance(document_splits, list):
        for index_data in document_splits:
            _process_index(index_data, es_client, embeddings)
    elif isinstance(document_splits, dict):
        for index_name, splits in document_splits.items():
            _process_index({"index_name": index_name, "splits": splits}, es_client, embeddings)

    _log.info("Document ingestion complete!")


def prepare_documents_for_es(output_dir):
    """
    Prepare processed markdown documents for Elasticsearch ingestion.

    Args:
        output_dir: Directory containing the markdown files

    Returns:
        dict: Dictionary mapping index name to list of document splits
    """
    _log.info(f"Preparing documents from {output_dir} for Elasticsearch ingestion")

    # Check if output directory exists and contains files
    if not output_dir.exists() or not output_dir.is_dir():
        _log.error(f"Output directory '{output_dir}' not found")
        return []  # Return empty list instead of None to avoid errors with JSON serialization

    # Find all markdown files
    md_files = list(output_dir.glob("*.md"))

    if not md_files:
        _log.error(f"No markdown files found in '{output_dir}'")
        return []  # Return empty list instead of None to avoid errors with JSON serialization

    _log.info(f"Found {len(md_files)} markdown files to ingest")

    # Create index name for Open Practice Library
    index_name = "open_practice_library_en_us_2024".lower()

    # Initialize documents list
    documents = []

    # Process each markdown file
    for md_file in md_files:
        try:
            # Read markdown content
            with open(md_file, encoding="utf-8") as f:
                content = f.read()

            # Extract title from first line (assumes markdown starts with # Title)
            title_match = re.match(r"^# (.+)$", content.split("\n")[0])
            title = title_match.group(1) if title_match else md_file.stem

            # Create metadata
            metadata = {
                "source": "open_practice_library",
                "source_full_name": "Open Practice Library",
                "title": title,
                "filename": md_file.name,
                "version": "2024",
                "language": "en-US",
                "url": f"https://openpracticelibrary.com/practice/{md_file.stem}",
            }

            # Create document
            document = {"page_content": content, "metadata": metadata}

            documents.append(document)

        except Exception as e:
            _log.error(f"Error processing {md_file}: {str(e)}")

    # Return document splits for Elasticsearch ingestion in the format expected by the pipeline
    # Format as an array of objects with index_name and splits fields
    return [{"index_name": index_name, "splits": documents}]


if __name__ == "__main__":
    # This allows the module to be run directly for testing
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Elasticsearch ingestion utility")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing markdown files")
    args = parser.parse_args()

    # Prepare and ingest documents
    output_dir = Path(args.input_dir)
    document_splits = prepare_documents_for_es(output_dir)

    if document_splits:
        ingest_to_elasticsearch(document_splits)
    else:
        _log.error("No documents to ingest")
        import sys

        sys.exit(1)
