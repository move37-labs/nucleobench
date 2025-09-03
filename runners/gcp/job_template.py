"""
Google Batch job template generator.

USAGE:
    This module generates Google Batch job definitions from hyperparameters.
    It's used internally by job_launcher.py to create job configurations.
    
    # Direct usage (if needed)
    from runners.gcp.job_template import create_job_definition
    
    job_config = create_job_definition({
        'job_name': 'test_job',
        'model': 'malinois',
        'start_sequence': 'ATCG...',
        'optimization': 'fastseqprop'
    })
    
    # Create example TSV
    example_tsv = create_example_tsv()
    with open('example.tsv', 'w') as f:
        f.write(example_tsv)

FUNCTIONS:
    - create_job_definition(hyperparams): Creates job JSON from TSV row
    - build_docker_command(hyperparams): Maps TSV columns to docker args
"""

from google.cloud import batch_v1
from runners.gcp import config


def create_job_definition(hyperparams: dict[str, str]) -> batch_v1.Job:
    """Create a Google Batch job definition from hyperparameters."""
    assert isinstance(hyperparams, dict)
    for k, v in hyperparams.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        assert len(v) > 0
    
    # Extract key parameters with defaults
    job_name = hyperparams['job_name']
    job_type = 'enformer' if 'enformer' in job_name else 'default'
    
    if job_type == 'enformer':
        machine_type = config.ENFORMER_MACHINE_TYPE
        cpu_count = int(config.ENFORMER_CPU_COUNT)
        memory_gb = int(config.ENFORMER_MEMORY_GB)
    else:
        machine_type = config.DEFAULT_MACHINE_TYPE
        cpu_count = int(config.DEFAULT_CPU_COUNT)
        memory_gb = int(config.DEFAULT_MEMORY_GB)
    
    # Build docker command from hyperparameters
    docker_cmd = build_docker_command(hyperparams)

    # Define a container runnable
    container = batch_v1.Runnable.Container(
        image_uri=config.DOCKER_IMAGE,
        commands=docker_cmd,
    )

    runnable = batch_v1.Runnable(container=container)

    # Define the task spec
    if 'max_seconds' not in hyperparams or not hyperparams['max_seconds']:
        raise ValueError("The 'max_seconds' column is required in the TSV file and cannot be empty.")
    
    # Double the timeout to allow for a natural job completion.
    # Also add time to allow for the job to start.
    timeout_seconds = int(hyperparams['max_seconds']) * 2 + 600
    task_spec = batch_v1.TaskSpec(
        runnables=[runnable],
        compute_resource=batch_v1.ComputeResource(
            cpu_milli=cpu_count * 1000,
            memory_mib=memory_gb * 1024,
        ),
        max_run_duration=f"{timeout_seconds}s",
        environment=batch_v1.Environment(
            variables={
                "GOOGLE_CLOUD_PROJECT_ID": config.PROJECT_ID,
                "GOOGLE_CLOUD_BUCKET": config.BUCKET_NAME,
                "DOCKER_IMAGE": config.DOCKER_IMAGE
            }
        )
    )

    # Define the task group
    task_group = batch_v1.TaskGroup(
        task_count=1,
        parallelism=1,
        task_spec=task_spec
    )

    # Define the allocation policy
    service_account = batch_v1.ServiceAccount(
        email=config.SERVICE_ACCOUNT_EMAIL
    ) if config.SERVICE_ACCOUNT_EMAIL else None

    allocation_policy = batch_v1.AllocationPolicy(
        instances=[
            batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                install_ops_agent=True,
                policy=batch_v1.AllocationPolicy.InstancePolicy(
                    machine_type=machine_type
                )
            )
        ],
        service_account=service_account,
    )

    # Define the job
    job = batch_v1.Job(
        task_groups=[task_group],
        allocation_policy=allocation_policy,
        logs_policy=batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        ),
    )
    
    return job


def build_docker_command(hyperparams: dict[str, str]) -> list[tuple[str, str]]:
    """Build docker command arguments from hyperparameters."""
    for k in ['Docker image', 'job_name', 'Entrypoint']:
        if k in hyperparams:
            del hyperparams[k]
    if 'seed_sequence' in hyperparams:
        hyperparams['start_sequence'] = hyperparams['seed_sequence']
        del hyperparams['seed_sequence']
    return [f"--{k}={v}" for k, v in hyperparams.items() if v is not None and len(v) > 0]
