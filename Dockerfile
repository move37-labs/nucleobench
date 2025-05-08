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
    # docker tag an_linuxamd64 us-east1-docker.pkg.dev/nucleobench/sequence-optimization/an_linuxamd64:latest
    # docker push us-east1-docker.pkg.dev/nucleobench/sequence-optimization/an_linuxamd64:latest

FROM mambaorg/micromamba:2.0.5

# These can be overridden at build time if needed, e.g., --build-arg APP_USER_UID=$(id -u)
# but true host UID matching for runtime requires an entrypoint script.
ARG APP_USER_NAME=appuser
ARG APP_USER_UID=1001
ARG APP_USER_GID=1001

# Set the working directory
WORKDIR /nucleobench

# Get pip. Needed once we switch to ubuntu.
USER root
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    libssl-dev=1.1.1w-0+deb11u2 \
    curl=7.74.0-1.3+deb11u11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Create the application group and user.
RUN groupadd --gid ${APP_USER_GID} ${APP_USER_NAME} && \
    useradd --uid ${APP_USER_UID} --gid ${APP_USER_GID} --create-home --shell /bin/bash ${APP_USER_NAME}

# Copy the environment file with correct ownership for the appuser.
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} environment.yml /tmp/environment.yml

# Install dependencies via micromamba.
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Copy only the needed subset of files.
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} docker_entrypoint.py /nucleobench/
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} docker_entrypoint_multi.py /nucleobench/
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} nucleobench /nucleobench/nucleobench
# These starting sequences for enformer are too long to include in the csv.
# NOTE: This should be removed in the final docker image, since it is 95% the size of the image.
# COPY experiments/google_batch/start_sequences/muscle_expression /nucleobench/experiments/google_batch/start_sequences/muscle_expression

#(otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  

# Uncomment the entrypoint we want this Docker image to use.
# For the bulk runs.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "docker_entrypoint.py"]
# For convergence runs, where multiple optimizers run on the same machine.
#ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "docker_entrypoint_multi.py"]