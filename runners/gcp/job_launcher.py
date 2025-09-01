"""
Google Batch job launcher for nucleobench.

USAGE:
    # Basic usage - launch jobs from TSV file
    python -m runners.gcp.job_launcher runners/testdata/adabeam_test.tsv
    
    # With options
    python -m runners.gcp.job_launcher \
        --dry-run \
        --verbose \
        runners/testdata/adabeam_test.tsv
    
    # Check help
    python -m runners.gcp.job_launcher --help

REQUIREMENTS:
    - Google Cloud authentication (gcloud auth login)
    - Enabled APIs: batch.googleapis.com, storage.googleapis.com
    - Docker image pushed to Google Container Registry
    - Environment variables set (see config.py)

EXAMPLES:
    # Test configuration without launching
    python -m runners.gcp.job_launcher --dry-run runners/gcp/testdata/ledidi-testrun.tsv
    
    # Launch with limited concurrency
    python -m runners.gcp.job_launcher runners/gcp/testdata/ledidi-testrun.tsv
"""

import argparse
import csv
import datetime
import logging
import os
import sys
from typing import Optional
import traceback

from google.cloud import batch_v1
from google.auth import default

from runners.gcp import config
from runners.gcp.job_template import create_job_definition


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def read_tsv_hyperparameters(tsv_path: str) -> list[dict[str, str]]:
    """Read hyperparameters from TSV file."""
    jobs = []
    
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row_num, row in enumerate(reader, 1):
            # Generate job name if missing
            assert 'job_name' not in row
            assert 'model' in row and row['model']
            assert 'optimization' in row and row['optimization']
            
            possible_keys = ['target_feature', 'protein', 'aggregation_type']
            possible_values = [row[k] for k in possible_keys if k in row and row[k]]
            assert len(possible_values) == 1
            target_feature = possible_values[0]
            target_feature = target_feature.replace('_', '-').lower()
            
            job_name = f"{row['model']}-{target_feature}-{row['optimization']}-{row_num:03d}"
            row['job_name'] = job_name + datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
            logging.info(f"Row {row_num}: Generated job name: {job_name}")
            
            jobs.append(row)
    
    logging.info(f"Loaded {len(jobs)} jobs from {tsv_path}")
    return jobs


def launch_job(
    client: batch_v1.BatchServiceClient,
    project_id: str,
    region: str,
    job: batch_v1.Job,
    job_name: str,
    dry_run: bool = False,
) -> Optional[str]:
    """Launch a single Google Batch job."""
    try:
        # Create job request from the Job object
        parent = f"projects/{project_id}/locations/{region}"
        
        request = batch_v1.CreateJobRequest(
            parent=parent,
            job_id=job_name,
            job=job
        )
        
        if dry_run:
            job_name_full = 'dry-run-job-id'
            logging.info(f"DRY RUN - Would have launched job: launched job: {job_name}")
        else:
            created_job = client.create_job(request=request)
            job_name_full = created_job.name
            logging.info(f"Successfully launched job: {job_name_full}")
        return job_name_full
        
    except Exception as e:
        logging.error(f"Failed to launch job {job_name}: {e}")
        logging.error(traceback.format_exc())
        return None


def launch_jobs(
    tsv_path: str,
    dry_run: bool = False
) -> None:
    """Launch multiple jobs sequentially from TSV file."""
    # Read hyperparameters
    jobs = read_tsv_hyperparameters(tsv_path)
    
    if not jobs:
        logging.error("No valid jobs found in TSV file")
        return
    
    # Setup Google Cloud clients
    credentials, _ = default()
    project_id = config.PROJECT_ID # Explicitly use the project from config
    
    client = batch_v1.BatchServiceClient(credentials=credentials)
    
    # Launch jobs sequentially
    successful_jobs = []
    failed_jobs = []
    
    for hyperparams in jobs:
        assert isinstance(hyperparams, dict)
        hyperparams = {k: v for k, v in hyperparams.items() if k is not None and v is not None and len(v) > 0}
        
        job_name = hyperparams['job_name']
        try:
            job_def = create_job_definition(hyperparams)
            job_id = launch_job(
                client=client,
                project_id=project_id,
                region=config.REGION,
                job=job_def,
                job_name=job_name,
                dry_run=dry_run,
            )
            if job_id:
                successful_jobs.append(job_name)
            else:
                failed_jobs.append(job_name)
        except Exception as e:
            logging.error(f"Job {job_name} failed with exception: {e}")
            logging.error(traceback.format_exc())
            failed_jobs.append(job_name)

    # Summary
    if dry_run:
        logging.info(f"DRY RUN MODE - No jobs were launched. Would have launched: {len(successful_jobs) + len(failed_jobs)}")
    else:
        logging.info(f"\nJob Launch Summary:")
        logging.info(f"Successful: {len(successful_jobs)}")
        logging.info(f"Failed: {len(failed_jobs)}")
        if successful_jobs:
            logging.info(f"Successful jobs: {', '.join(successful_jobs)}")
        if failed_jobs:
            logging.info(f"Failed jobs: {', '.join(failed_jobs)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch Google Batch jobs from TSV file")
    parser.add_argument("tsv_file", help="Path to TSV file with hyperparameters")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be launched without actually launching")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate TSV file
    if not os.path.exists(args.tsv_file):
        logging.error(f"TSV file not found: {args.tsv_file}")
        sys.exit(1)
        
    # Validate config.
    if config.PROJECT_ID is None:
        logging.error(f"config.PROJECT_ID is not set. Please set it in config.py.")
        sys.exit(1)
    if config.BUCKET_NAME is None:
        logging.error(f"config.BUCKET_NAME is not set. Please set it in config.py.")
        sys.exit(1)
    if config.DOCKER_IMAGE is None:
        logging.error(f"config.DOCKER_IMAGE is not set. Please set it in config.py.")
        sys.exit(1)
    
    # Launch jobs
    launch_jobs(
        args.tsv_file,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
