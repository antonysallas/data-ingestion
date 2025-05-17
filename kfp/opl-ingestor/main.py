"""
Main entry point for Open Practice Library Ingestion Pipeline.

This script fetches practice content from the Open Practice Library website,
processes it into markdown, and ingests it into Elasticsearch using Kubeflow Pipelines.

Environment variables:
  - KUBEFLOW_ENDPOINT: Kubeflow pipeline endpoint (required)
  - BEARER_TOKEN: Authentication token for Kubeflow (required)
  - ES_USER: Elasticsearch username (used only in local mode)
  - ES_PASS: Elasticsearch password (used only in local mode)
  - ES_HOST: Elasticsearch host URL (used only in local mode)

Usage:
  python main.py  # Run in pipeline mode (default)
  python main.py --local  # Run in local standalone mode
"""

import argparse
import importlib.util
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

_log = logging.getLogger(__name__)

# Constants
ENV_NOT_SET = "NOT SET"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OPL Ingestion Pipeline")

    # Mode selection - keeping for backward compatibility but not using by default
    parser.add_argument("--local", action="store_true", help="Run in local standalone mode instead of pipeline mode")

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

    # Elasticsearch options - keeping for backward compatibility but not using by default
    parser.add_argument("--skip-es", action="store_true", help="Skip Elasticsearch ingestion (for local mode only)")

    # Elasticsearch configuration (when not using environment variables)
    parser.add_argument("--es-user", type=str, help="Elasticsearch username (for local mode only)")
    parser.add_argument("--es-pass", type=str, help="Elasticsearch password (for local mode only)")
    parser.add_argument("--es-host", type=str, help="Elasticsearch host URL (for local mode only)")

    return parser.parse_args()


def run_pipeline_mode():
    """Run in Kubeflow Pipeline mode."""
    _log.info("Running in pipeline mode")

    kubeflow_endpoint = os.environ.get("KUBEFLOW_ENDPOINT")
    if not kubeflow_endpoint:
        _log.error("KUBEFLOW_ENDPOINT environment variable not set")
        return 1

    _log.info(f"Connecting to kfp: {kubeflow_endpoint}")

    # Import dynamically to handle import errors gracefully
    try:
        # Try to import from package first
        try:
            from .kubeflow_components import run_kubeflow_pipeline
        except ImportError:
            # If that fails, try importing directly
            from .kubeflow_components import run_kubeflow_pipeline

        run_id = run_kubeflow_pipeline()
        _log.info(f"Pipeline run created: {run_id}")
        return 0
    except Exception as e:
        _log.error(f"Error running pipeline: {str(e)}")
        _log.debug(traceback.format_exc())
        return 1


def _ingest_to_elasticsearch(output_dir, args, logger):
    """Handle Elasticsearch ingestion process."""
    logger.info("Elasticsearch ingestion is enabled")
    try:
        from .elasticsearch_ingest import ingest_to_elasticsearch, prepare_documents_for_es

        logger.info("Preparing to ingest documents into Elasticsearch")
        es_user = args.es_user or os.environ.get("ES_USER", "elastic")
        es_pass = args.es_pass or os.environ.get("ES_PASS")
        es_host = args.es_host or os.environ.get("ES_HOST", "http://elasticsearch-es-http:9200")

        logger.info("ES_USER is %s", "set" if es_user else ENV_NOT_SET)
        logger.info("ES_PASS is %s", "set" if es_pass else ENV_NOT_SET)
        logger.info("ES_HOST is %s", "set" if es_host else ENV_NOT_SET)

        if not es_pass:
            logger.error("ES_PASS not set. Exiting without ingesting to Elasticsearch.")
            return 1

        os.environ.update({"ES_USER": es_user, "ES_PASS": es_pass, "ES_HOST": es_host})
        document_splits = prepare_documents_for_es(output_dir)

        if document_splits:
            logger.info("Ingesting processed documents into Elasticsearch...")
            ingest_to_elasticsearch(document_splits)
            logger.info("Successfully ingested documents into Elasticsearch.")
        else:
            logger.error("No documents prepared for ingestion. Skipping Elasticsearch ingestion.")
        return 0
    except ImportError:
        logger.warning("Elasticsearch module could not be imported. Skipping Elasticsearch ingestion.")
        return 0
    except Exception as e:
        logger.error(f"Error during Elasticsearch ingestion: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1


def _process_practices(practice_urls, output_dir, delay, logger, process_func):
    """Process a list of practice URLs and save them to the output directory."""
    successful = failed = 0
    for i, url in enumerate(practice_urls):
        logger.info(f"Processing practice {i + 1}/{len(practice_urls)}: {url}")
        if process_func(url, output_dir):
            successful += 1
        else:
            failed += 1
        if i < len(practice_urls) - 1:
            time.sleep(delay)
    return successful, failed


def run_standalone_mode(args):
    """Run in standalone mode."""
    _log.info("Running in standalone mode")

    # Import modules using a helper function to handle different import methods
    def import_module(module_name, fallback_file):
        try:
            # First try direct import from current directory
            return importlib.import_module(module_name)
        except ImportError:
            try:
                # Try to import from current package using relative import
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                return importlib.import_module(module_name)
            except ImportError:
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

        # Process practices
        successful, failed = _process_practices(practice_urls, output_dir, request_delay, _log, process_practice)
        _log.info(f"Website processing complete. Successfully processed {successful} practices. Failed: {failed}")
        _log.info(f"Markdown files saved to {output_dir.absolute()}")

        # Handle Elasticsearch ingestion
        if not skip_es:
            return _ingest_to_elasticsearch(output_dir, args, _log)
        else:
            _log.info("Elasticsearch ingestion is disabled with --skip-es flag")
            return 0

    except Exception as e:
        _log.error(f"Error in standalone execution: {str(e)}")
        _log.debug(traceback.format_exc())
        return 1


def _get_mode_reason(args):
    """Get the reason for running in local mode."""
    if args.local:
        return "--local flag specified"
    if args.skip_es:
        return "--skip-es flag specified"
    return "KUBEFLOW_ENDPOINT not set"


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Check for conditions that should trigger local mode
    run_local = args.local or args.skip_es or not os.environ.get("KUBEFLOW_ENDPOINT")

    if run_local:
        mode_reason = _get_mode_reason(args)
        _log.info(f"Running in local standalone mode ({mode_reason})")
        return run_standalone_mode(args)
    else:
        # In pipeline mode, ensure Elasticsearch ingestion is enabled
        args.skip_es = False
        _log.info("Running in pipeline mode (default)")
        return run_pipeline_mode()


if __name__ == "__main__":
    sys.exit(main())
