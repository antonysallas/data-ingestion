"""
Kubeflow Pipeline components for Open Practice Library content ingestion.

This module contains the Kubeflow Pipeline components and pipeline
definition for the OPL ingestion pipeline.
"""

import logging
import os

from kfp.dsl import Artifact, Input, Output

# Import Kubeflow Pipeline dependencies
import kfp
from kfp import dsl, kubernetes

# Configure module logger
_log = logging.getLogger(__name__)


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "beautifulsoup4"],
)
def load_documents() -> list:
    """
    KFP Component to load documents from Open Practice Library.

    Returns:
        List of URLs to process
    """
    import logging
    from urllib.parse import urljoin

    import requests
    from bs4 import BeautifulSoup

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("load_documents")

    # Base URL for the Open Practice Library
    opl_base_url = "https://openpracticelibrary.com/"

    logger.info(f"Fetching practice URLs from {opl_base_url}")
    response = requests.get(opl_base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all practice card links
    practice_links = soup.select('div[data-testid="practicecardgrid"] a')
    logger.info(f"Found {len(practice_links)} practice links")

    # Extract and normalize URLs
    practice_urls = []
    for link in practice_links:
        href = link.get("href")
        if href and href.startswith("/practice/"):
            full_url = urljoin(opl_base_url, href)
            practice_urls.append(full_url)

    # Limit the number of practices to process for testing
    max_practices = 0  # Set to 0 for no limit
    if max_practices > 0:
        logger.info(f"Limiting to {max_practices} practices for processing")
        practice_urls = practice_urls[:max_practices]

    logger.info(f"Extracted {len(practice_urls)} practice URLs")
    return practice_urls


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "beautifulsoup4==4.12.2",
        "html2text==2024.2.26",
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "lxml==5.1.0",
        "tqdm==4.66.2",
        "elastic-transport==8.15.1",
        "elasticsearch==8.16.0",
        "langchain-elasticsearch==0.3.0",
    ],
)
def format_documents(documents: list, splits_artifact: Output[Artifact]):
    """
    KFP Component to format OPL documents into markdown and prepare for Elasticsearch.

    Args:
        documents: List of URLs to process

    Returns:
        Dictionary containing document splits for Elasticsearch
    """
    import json
    import logging
    import os
    import re
    from pathlib import Path

    import requests
    from bs4 import BeautifulSoup

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("format_documents")

    # Use a temporary directory that we have permission to create/write in container environment
    import tempfile

    # Create a temporary directory for storing the processed files
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir) / "practices"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created temporary output directory: {output_dir}")

    # Import directly from the current directory
    import os
    import sys

    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Direct imports from current directory
    from elasticsearch_ingest import prepare_documents_for_es
    from html_processing import extract_opl_content
    from markdown_processing import process_practice

    # Process each URL
    successful = 0
    failed = 0

    for i, url in enumerate(documents):
        logger.info(f"Processing practice {i+1}/{len(documents)}: {url}")

        if process_practice(url, output_dir):
            successful += 1
        else:
            failed += 1

    logger.info(f"Successfully processed {successful} practices. Failed: {failed}")

    # Prepare documents for Elasticsearch
    document_splits = prepare_documents_for_es(output_dir)

    # Write splits to output artifact
    with open(splits_artifact.path, "w") as f:
        f.write(json.dumps(document_splits))


@dsl.component(
    base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/minimal-gpu:2024.2",
    packages_to_install=[
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "elastic-transport==8.15.1",
        "elasticsearch==8.16.0",
        "langchain-elasticsearch==0.3.0",
        "sentence-transformers==2.4.0",
        "einops==0.7.0",
    ],
)
def ingest_documents(input_artifact: Input[Artifact]) -> None:
    """
    KFP Component to ingest documents into Elasticsearch.

    Args:
        input_artifact: Artifact containing document splits
    """
    import json
    import logging
    import os

    from elasticsearch import Elasticsearch
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_elasticsearch import ElasticsearchStore

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ingest_documents")

    # Read input artifact
    with open(input_artifact.path) as input_file:
        splits_artifact = input_file.read()
        document_splits = json.loads(splits_artifact)

    # Log environment variables (for debugging)
    logger.info("Environment variables for Elasticsearch:")
    logger.info(f"ES_USER: {'SET' if os.environ.get('ES_USER') else 'NOT SET'}")
    logger.info(f"ES_PASS: {'SET' if os.environ.get('ES_PASS') else 'NOT SET'}")
    logger.info(f"ES_HOST: {'SET' if os.environ.get('ES_HOST') else 'NOT SET'}")

    # Get Elasticsearch credentials from environment variables
    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    # Check if required credentials are set
    if not es_user or not es_pass or not es_host:
        logger.error(
            "Elasticsearch config not present. Check ES_USER, ES_PASS, and ES_HOST environment variables."
        )
        return

    # Initialize Elasticsearch client
    logger.info(f"Connecting to Elasticsearch at {es_host}")
    es_client = Elasticsearch(es_host, basic_auth=(es_user, es_pass), request_timeout=30, verify_certs=False)

    # Health check for Elasticsearch client connection
    logger.info(f"Elasticsearch cluster status: {es_client.cluster.health()}")

    def ingest(index_name, splits):
        """Ingest documents into Elasticsearch with embeddings."""
        # Initialize embedding model
        model_name = "nomic-ai/nomic-embed-text-v1"

        # Set model parameters for GPU (if available)
        device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
        logger.info(f"Using device: {device} for embeddings")

        model_kwargs = {"trust_remote_code": True, "device": device}

        # Initialize embeddings
        logger.info(f"Initializing embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            show_progress=True,
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize Elasticsearch store with embeddings
        logger.info(f"Creating/updating index: {index_name}")
        db = ElasticsearchStore(
            index_name=index_name.lower(),  # index names in elastic must be lowercase
            embedding=embeddings,
            es_connection=es_client,
        )

        # Add documents in batches
        batch_size = 50
        total_batches = (len(splits) + batch_size - 1) // batch_size

        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(
                f"Uploading batch {batch_num}/{total_batches} ({len(batch)} documents) to index {index_name}"
            )
            db.add_documents(batch)

        logger.info(f"Successfully uploaded all documents to index {index_name}")

    # Process each index and its documents
    # Additional logging to help debug
    logger.info(f"Got document_splits: type={type(document_splits)}, content format={type(document_splits[0]) if document_splits else 'empty'}")

    for index_data in document_splits:
        index_name = index_data["index_name"]
        splits = index_data["splits"]
        logger.info(f"Processing index {index_name} with {len(splits)} documents")

        # Convert to Document objects
        documents = [
            Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits
        ]

        # Ingest documents
        ingest(index_name=index_name, splits=documents)

    logger.info("Elasticsearch ingestion complete")


@dsl.pipeline(name="Document Ingestion")
def ingestion_pipeline():
    """
    Kubeflow Pipeline for OPL document ingestion.
    """
    # Step 1: Load documents from OPL website
    load_docs_task = load_documents()

    # Step 2: Format documents into markdown
    format_docs_task = format_documents(documents=load_docs_task.output)
    format_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    # Step 3: Ingest documents into Elasticsearch
    ingest_docs_task = ingest_documents(input_artifact=format_docs_task.outputs["splits_artifact"])
    ingest_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    # Set environment variables for Elasticsearch - following exactly the same pattern as the working pipeline
    kubernetes.use_secret_as_env(
        ingest_docs_task,
        secret_name="elasticsearch-es-elastic-user",
        secret_key_to_env={"elastic": "ES_PASS"},
    )
    ingest_docs_task.set_env_variable("ES_HOST", "http://elasticsearch-es-http:9200")
    ingest_docs_task.set_env_variable("ES_USER", "elastic")

    # Add GPU tolerations
    kubernetes.add_toleration(format_docs_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    kubernetes.add_toleration(ingest_docs_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")


def run_kubeflow_pipeline():
    """
    Run the Kubeflow Pipeline with the configured client.

    Returns:
        str: ID of the created run
    """
    # Get Kubeflow endpoint and authentication token
    KUBEFLOW_ENDPOINT = os.environ.get("KUBEFLOW_ENDPOINT")
    _log.info(f"Connecting to kfp: {KUBEFLOW_ENDPOINT}")

    # Get service account token
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            BEARER_TOKEN = f.read().rstrip()
    else:
        BEARER_TOKEN = os.environ.get("BEARER_TOKEN")

    # Get service account certificate
    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    if os.path.isfile(sa_ca_cert):
        ssl_ca_cert = sa_ca_cert
    else:
        ssl_ca_cert = None

    # Create KFP client and run pipeline
    client = kfp.Client(
        host=KUBEFLOW_ENDPOINT,
        existing_token=BEARER_TOKEN,
        ssl_ca_cert=None,
    )

    result = client.create_run_from_pipeline_func(
        ingestion_pipeline,
        experiment_name="document_ingestion",
    )

    _log.info(f"Pipeline run created: {result.run_id}")
    return result.run_id
