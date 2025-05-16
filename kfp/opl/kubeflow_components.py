"""
Kubeflow Pipeline components for Open Practice Library content ingestion.

This module contains the Kubeflow Pipeline components and pipeline
definition for the OPL ingestion pipeline.
"""

import logging
import os

from kfp.kubernetes import kubernetes

# Import Kubeflow Pipeline dependencies
import kfp
from kfp import dsl

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
    max_practices = 10  # Set to 0 for no limit
    if max_practices > 0:
        logger.info(f"Limiting to {max_practices} practices for processing")
        practice_urls = practice_urls[:max_practices]

    logger.info(f"Extracted {len(practice_urls)} practice URLs")
    return practice_urls


@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "requests",
        "beautifulsoup4",
        "langchain",
        "langchain-elasticsearch",
        "elasticsearch",
        "huggingface-hub",
        "transformers",
        "torch",
    ],
)
def format_documents(documents: list) -> dict:
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

    # Create output directory
    output_dir = Path("practices")
    output_dir.mkdir(parents=True, exist_ok=True)

    # In KFP component, manually import the functions we need
    # For simplicity in this component, we'll include the processing code here
    # In a real application, you would use proper imports

    # HTML processing functions
    def convert_html_to_markdown(html_content):
        """Convert HTML content to Markdown"""
        # Implementation...
        pass

    def process_list(ul_element, indent=0, global_processed=None):
        """Process an unordered list element"""
        # Implementation...
        pass

    def process_ordered_list(ol_element, indent=0, global_processed=None):
        """Process an ordered list element"""
        # Implementation...
        pass

    def clean_text(text):
        """Clean up text by removing special characters"""
        # Implementation...
        pass

    def clean_markdown(markdown_text):
        """Clean and normalize markdown text"""
        # Implementation...
        pass

    def extract_opl_content(html_content):
        """Extract content from OPL HTML"""
        # Implementation...
        pass

    def format_output(practice_content, output_format="markdown"):
        """Format practice content into specified format"""
        # Implementation...
        pass

    def process_practice(source, output_dir):
        """Process a practice source and generate markdown"""
        # Implementation...
        pass

    def prepare_documents_for_es(output_dir):
        """Prepare markdown documents for Elasticsearch"""
        # Implementation...
        pass

    # In a real component, you would import these functions from their respective modules
    from opl.elasticsearch_ingest import prepare_documents_for_es
    from opl.html_processing import extract_opl_content
    from opl.markdown_processing import process_practice

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

    # Convert to JSON for passing between components
    splits_json = json.dumps(document_splits)

    # Return splits as an artifact
    return {"splits_artifact": splits_json}


@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "elasticsearch",
        "langchain",
        "langchain-elasticsearch",
        "huggingface-hub",
        "transformers",
        "torch",
    ],
)
def ingest_documents(input_artifact: str) -> None:
    """
    KFP Component to ingest documents into Elasticsearch.

    Args:
        input_artifact: JSON string containing document splits
    """
    import json
    import logging
    import os

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ingest_documents")

    # Log environment variables (for debugging)
    logger.info("Environment variables available:")
    for key in sorted(os.environ.keys()):
        if key.startswith("ES_"):
            logger.info(f"  {key}: {'set' if os.environ.get(key) else 'NOT SET'}")

    # Get Elasticsearch credentials from environment variables
    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    if not es_user or not es_pass or not es_host:
        logger.error(
            "Elasticsearch config not present. Check ES_USER, ES_PASS, and ES_HOST environment variables."
        )
        return

    # Parse input artifact
    document_splits = json.loads(input_artifact)

    # In a real component, you would import this function from its module
    from opl.elasticsearch_ingest import ingest_to_elasticsearch

    # Call ingest_to_elasticsearch function
    ingest_to_elasticsearch(document_splits)

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

    # Set environment variables for Elasticsearch
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
        # enable_caching=False
    )

    _log.info(f"Pipeline run created: {result.run_id}")
    return result.run_id
