"""
Main entry point for Open Practice Library Ingestion Pipeline.

This script fetches practice content from the Open Practice Library website,
processes it into markdown, and ingests it into Elasticsearch.

Environment variables:
  - ES_USER: Elasticsearch username
  - ES_PASS: Elasticsearch password
  - ES_HOST: Elasticsearch host URL
  - KUBEFLOW_ENDPOINT: Kubeflow pipeline endpoint (for pipeline usage)
  - BEARER_TOKEN: Authentication token for Kubeflow (for pipeline usage)

Usage:
  python main.py  # Run in standalone mode
  python main.py --pipeline  # Run in pipeline mode
"""

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

_log = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OPL Ingestion Pipeline")

    # Mode selection
    parser.add_argument("--pipeline", action="store_true", help="Run in pipeline mode")

    # Configuration options
    parser.add_argument(
        "--output-dir", type=str, default="practices", help="Directory for output files (default: practices)"
    )
    parser.add_argument(
        "--base-url", type=str, default="https://openpracticelibrary.com/", help="Base URL for OPL website"
    )
    parser.add_argument(
        "--max-practices",
        type=int,
        default=10,
        help="Maximum number of practices to process (0 for no limit)",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")

    # Elasticsearch options
    parser.add_argument("--skip-es", action="store_true", help="Skip Elasticsearch ingestion")

    # Elasticsearch configuration (when not using environment variables)
    parser.add_argument("--es-user", type=str, help="Elasticsearch username")
    parser.add_argument("--es-pass", type=str, help="Elasticsearch password")
    parser.add_argument("--es-host", type=str, help="Elasticsearch host URL")

    return parser.parse_args()


def run_pipeline_mode():
    """Run in Kubeflow Pipeline mode."""
    _log.info("Running in pipeline mode")

    KUBEFLOW_ENDPOINT = os.environ.get("KUBEFLOW_ENDPOINT")
    if not KUBEFLOW_ENDPOINT:
        _log.error("KUBEFLOW_ENDPOINT environment variable not set")
        return 1

    _log.info(f"Connecting to kfp: {KUBEFLOW_ENDPOINT}")

    # Import dynamically to handle import errors gracefully
    try:
        # Try to import from package first
        try:
            from opl.kubeflow_components import run_kubeflow_pipeline
        except ImportError:
            # If that fails, try importing directly
            import kubeflow_components

            run_kubeflow_pipeline = kubeflow_components.run_kubeflow_pipeline

        run_id = run_kubeflow_pipeline()
        _log.info(f"Pipeline run created: {run_id}")
        return 0
    except Exception as e:
        _log.error(f"Error running pipeline: {str(e)}")
        import traceback

        _log.debug(traceback.format_exc())
        return 1


def run_standalone_mode(args):
    """Run in standalone mode."""
    _log.info("Running in standalone mode")

    # Import modules using a helper function to handle different import methods
    def import_module(module_name, fallback_file):
        try:
            # Try to import from package first
            return importlib.import_module(f"opl.{module_name}")
        except ImportError:
            # If that fails, try importing directly
            if importlib.util.find_spec(module_name):
                return importlib.import_module(module_name)
            # If all else fails, provide a helpful error
            _log.error(
                f"Could not import module '{module_name}'. Make sure '{fallback_file}' is in the current directory."
            )
            raise

    try:
        # Import HTML processing module
        html_module = import_module("html_processing", "html_processing.py")
        get_all_practice_urls = html_module.get_all_practice_urls

        # Import Markdown processing module
        markdown_module = import_module("markdown_processing", "markdown_processing.py")
        process_practice = markdown_module.process_practice

        # Directory for output files
        output_dir = Path(args.output_dir)

        # Base URL for the Open Practice Library
        opl_base_url = args.base_url

        # Limit the number of practices to process
        max_practices = args.max_practices

        # Delay between requests to avoid overloading the server
        request_delay = args.delay

        # Skip Elasticsearch ingestion if explicitly disabled
        skip_es = args.skip_es

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Output directory: {output_dir.absolute()}")

        # Process practices from the website
        _log.info(f"Starting to process practices from {opl_base_url}")

        # Get all practice URLs
        practice_urls = get_all_practice_urls(opl_base_url)

        if not practice_urls:
            _log.error("No practice URLs found from website. Exiting.")
            return 1

        # Limit the number of practices to process if max_practices is set
        if max_practices > 0:
            _log.info(f"Limiting to {max_practices} practices for processing")
            practice_urls = practice_urls[:max_practices]

        _log.info(f"Found {len(practice_urls)} practices to process")

        # Process each practice
        successful = 0
        failed = 0

        for i, url in enumerate(practice_urls):
            _log.info(f"Processing practice {i+1}/{len(practice_urls)}: {url}")

            if process_practice(url, output_dir):
                successful += 1
            else:
                failed += 1

            # Add a delay to avoid overloading the server
            if i < len(practice_urls) - 1:  # Don't delay after the last request
                time.sleep(request_delay)

        _log.info(
            f"Website processing complete. Successfully processed {successful} practices. Failed: {failed}"
        )
        _log.info(f"Markdown files saved to {output_dir.absolute()}")

        # Ingest documents into Elasticsearch by default unless skip_es is provided
        if not skip_es:
            _log.info("Elasticsearch ingestion is enabled")

            # Import Elasticsearch module only if needed
            try:
                es_module = import_module("elasticsearch_ingest", "elasticsearch_ingest.py")
                prepare_documents_for_es = es_module.prepare_documents_for_es
                ingest_to_elasticsearch = es_module.ingest_to_elasticsearch

                _log.info("Preparing to ingest documents into Elasticsearch")

                # Get Elasticsearch credentials from environment variables or command line arguments
                es_user = args.es_user or os.environ.get("ES_USER")
                es_pass = args.es_pass or os.environ.get("ES_PASS")
                es_host = args.es_host or os.environ.get("ES_HOST")

                # Try to use defaults if not specified
                if not es_user:
                    es_user = "elastic"
                    _log.info("ES_USER not specified, using default value 'elastic'")

                if not es_host:
                    es_host = "http://elasticsearch-es-http:9200"
                    _log.info(f"ES_HOST not specified, using default value '{es_host}'")

                # Log which variables are set (without revealing values)
                _log.info("ES_USER is %s", "set" if es_user else "NOT SET")
                _log.info("ES_PASS is %s", "set" if es_pass else "NOT SET")
                _log.info("ES_HOST is %s", "set" if es_host else "NOT SET")

                if not es_user or not es_pass or not es_host:
                    _log.error(
                        "Elasticsearch config not present. Check ES_USER, ES_PASS, and ES_HOST environment variables "
                        "or provide them via command line arguments."
                    )
                    _log.error("Exiting without ingesting to Elasticsearch.")
                    return 1

                # Set environment variables for the current process
                os.environ["ES_USER"] = es_user
                os.environ["ES_PASS"] = es_pass
                os.environ["ES_HOST"] = es_host

                try:
                    # Prepare documents for Elasticsearch
                    document_splits = prepare_documents_for_es(output_dir)
                    if document_splits:
                        # Ingest into Elasticsearch
                        _log.info("Ingesting processed documents into Elasticsearch...")
                        ingest_to_elasticsearch(document_splits)
                        _log.info("Successfully ingested documents into Elasticsearch.")
                    else:
                        _log.error("No documents prepared for ingestion. Skipping Elasticsearch ingestion.")
                except Exception as e:
                    _log.error(f"Error during Elasticsearch ingestion: {str(e)}")
                    import traceback

                    _log.debug(traceback.format_exc())
                    return 1
            except ImportError:
                _log.warning("Elasticsearch module could not be imported. Skipping Elasticsearch ingestion.")
        else:
            _log.info("Elasticsearch ingestion is disabled with --skip-es flag")

        return 0

    except Exception as e:
        _log.error(f"Error in standalone execution: {str(e)}")
        import traceback

        _log.debug(traceback.format_exc())
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Auto-detect pipeline mode by checking environment variables
    # If KUBEFLOW_ENDPOINT is set and we're running from a container, assume pipeline mode
    auto_pipeline_mode = os.environ.get("KUBEFLOW_ENDPOINT") is not None and os.path.exists("/.dockerenv")

    if args.pipeline or auto_pipeline_mode:
        # In pipeline mode, don't skip Elasticsearch ingestion by default
        if not hasattr(args, "skip_es") or args.skip_es is None:
            args.skip_es = False
        return run_pipeline_mode()
    else:
        return run_standalone_mode(args)


if __name__ == "__main__":
    sys.exit(main())
