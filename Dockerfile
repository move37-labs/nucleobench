# To build:
    # docker build -t nucleobench -f Dockerfile .

# IMPORTANT: The docker image for Google Batch must be linux/amd64.
# To build for linux/amd64:
    # docker buildx build --platform linux/amd64 -t an_linuxamd64 -f Dockerfile .

# To test:
    # hadolint Dockerfile

# To run locally:
    # docker run nucleobench

# To authenticate pushing to gcr.io:
    # gcloud init
    # gcloud auth configure-docker us-east1-docker.pkg.dev

# To push the linux/amd64to gcr.io:
    # docker tag an_linuxamd64 us-east1-docker.pkg.dev/nucleorave/sequence-optimization/an_linuxamd64:latest
    # docker push us-east1-docker.pkg.dev/nucleorave/sequence-optimization/an_linuxamd64:latest

FROM ubuntu:20.04
FROM mambaorg/micromamba:2.0.5

# Set the working directory
WORKDIR /nucleobench

# Get pip. Needed once we switch to ubuntu.
# TODO(joelshor): Pin the versions of `libssl-dev` and `curl`
USER root
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    libssl-dev \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
USER 1001

# Install dependencies via micromamba.
USER root
COPY environment.yml /tmp/environment.yml
USER 1001
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

# Copy only the needed subset of files.
COPY  docker_entrypoint.py /nucleobench/
COPY docker_entrypoint_multi.py /nucleobench/
COPY nucleobench /nucleobench/nucleobench
# These starting sequences for enformer are too long to include in the csv.
# NOTE: This should be removed in the final docker image, since it is 95% the size of the image.
# COPY experiments/google_batch/start_sequences/muscle_expression /nucleobench/experiments/google_batch/start_sequences/muscle_expression

# Change permissions to allow reading / writing.
USER root
RUN chmod -R 755 /nucleobench/nucleobench
USER 1001

#(otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  

# Uncomment the entrypoint we want this Docker image to use.
# For the bulk runs.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "docker_entrypoint.py"]
# For convergence runs, where multiple optimizers run on the same machine.
#ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "docker_entrypoint_multi.py"]