"""
Elasticsearch ingestion utilities for Open Practice Library content.

This module contains functions for preparing and ingesting content
into Elasticsearch with vector embeddings.
"""

import json
import logging
import os
import re

# Configure module logger
_log = logging.getLogger(__name__)


def ingest_to_elasticsearch(input_artifact):
    """
    Function to ingest the processed content into Elasticsearch.

    This can be called directly or used as a KFP component.

    Args:
        input_artifact: Either a path to a JSON file or a list of document splits
    """
    try:
        from elasticsearch import Elasticsearch
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_elasticsearch import ElasticsearchStore
    except ImportError:
        _log.error(
            "Required libraries not found. Please install 'elasticsearch', 'langchain', and 'langchain-elasticsearch'"
        )
        return

    # Process the input artifact, which can be a file path or a list of document splits
    document_splits = []

    if isinstance(input_artifact, str) or hasattr(input_artifact, "path"):
        # If it's a path string or has a path attribute (like KFP artifact)
        path = input_artifact.path if hasattr(input_artifact, "path") else input_artifact
        try:
            with open(path) as input_file:
                splits_artifact = input_file.read()
                document_splits = json.loads(splits_artifact)
        except Exception as e:
            _log.error(f"Error reading input artifact: {str(e)}")
            return
    elif isinstance(input_artifact, list):
        # If it's already a list of document splits
        document_splits = input_artifact
    else:
        _log.error(f"Unsupported input_artifact type: {type(input_artifact)}")
        return

    # Get Elasticsearch credentials from environment variables
    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    _log.info("Environment variables for Elasticsearch:")
    _log.info(f"ES_USER: {'SET' if es_user else 'NOT SET'}")
    _log.info(f"ES_PASS: {'SET' if es_pass else 'NOT SET'}")
    _log.info(f"ES_HOST: {'SET' if es_host else 'NOT SET'}")

    if not es_user or not es_pass or not es_host:
        _log.error(
            "Elasticsearch config not present. Please set ES_USER, ES_PASS, and ES_HOST environment variables."
        )
        # Return error code instead of just returning
        if __name__ != "__main__":  # Don't exit if imported as a module
            return 1
        else:
            import sys

            sys.exit(1)

    # Initialize Elasticsearch client
    _log.info(f"Connecting to Elasticsearch at {es_host}")
    try:
        es_client = Elasticsearch(
            es_host, basic_auth=(es_user, es_pass), request_timeout=30, verify_certs=False
        )
    except Exception as e:
        _log.error(f"Failed to initialize Elasticsearch client: {str(e)}")
        return

    # Health check for Elasticsearch client connection
    try:
        health = es_client.cluster.health()
        _log.info(f"Elasticsearch cluster status: {health['status']}")
    except Exception as e:
        _log.error(f"Error connecting to Elasticsearch: {str(e)}")
        return

    # Process each index and its documents directly (like the working example)
    if isinstance(document_splits, list):
        for index_data in document_splits:
            if isinstance(index_data, dict) and "index_name" in index_data and "splits" in index_data:
                index_name = index_data["index_name"]
                splits = index_data["splits"]
                _log.info(f"Processing index {index_name} with {len(splits)} documents")

                # Create document objects
                documents = [
                    Document(page_content=split["page_content"], metadata=split["metadata"])
                    for split in splits
                ]

                # Initialize embedding model - using Nomic AI's embedding model
                model_name = "nomic-ai/nomic-embed-text-v1"

                # Set model parameters
                try:
                    device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
                    _log.info(f"Using device: {device} for embeddings")
                    model_kwargs = {"trust_remote_code": True, "device": device}

                    _log.info(f"Initializing embeddings model: {model_name}")
                    embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        show_progress=True,
                        encode_kwargs={"normalize_embeddings": True},
                    )

                    # Initialize Elasticsearch store with embeddings
                    _log.info(f"Creating/updating index: {index_name}")
                    db = ElasticsearchStore(
                        index_name=index_name.lower(),
                        embedding=embeddings,
                        es_connection=es_client,
                    )

                    # Add documents in batches
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

    elif isinstance(document_splits, dict):
        # Handle the dictionary format
        for index_name, splits in document_splits.items():
            _log.info(f"Processing index {index_name} with {len(splits)} documents")

            # Create document objects
            documents = [
                Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits
            ]

            # Use the same embedding and processing logic as above
            # (Code would be duplicated here - consider refactoring)
            try:
                device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
                _log.info(f"Using device: {device} for embeddings")
                model_name = "nomic-ai/nomic-embed-text-v1"
                model_kwargs = {"trust_remote_code": True, "device": device}

                _log.info(f"Initializing embeddings model: {model_name}")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    show_progress=True,
                    encode_kwargs={"normalize_embeddings": True},
                )

                # Initialize Elasticsearch store with embeddings
                _log.info(f"Creating/updating index: {index_name}")
                db = ElasticsearchStore(
                    index_name=index_name.lower(),
                    embedding=embeddings,
                    es_connection=es_client,
                )

                # Add documents in batches
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
            with open(md_file, "r", encoding="utf-8") as f:
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
