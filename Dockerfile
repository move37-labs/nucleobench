# To build:
    # docker build -t nucleobench -f Dockerfile .

# IMPORTANT: The docker image for Google Batch must be linux/amd64.
# To build for linux/amd64:
    # docker buildx build --platform linux/amd64 -t nucleobench_linuxamd64:latest -f Dockerfile . --load

# To test:
    # hadolint Dockerfile

# To run locally:
    # docker run nucleobench

# To push to docker hub:
    # docker tag nucleobench_linuxamd64:latest joelshor/nucleobench:latest
    # docker push joelshor/nucleobench:latest

# To push to Github Container Registry:
    # docker tag nucleobench_linuxamd64:latest ghcr.io/move37-labs/nucleobench:latest
    # docker push ghcr.io/move37-labs/nucleobench:latest

FROM mambaorg/micromamba:2.0.5

# These can be overridden at build time if needed, e.g., --build-arg APP_USER_UID=$(id -u)
# but true host UID matching for runtime requires an entrypoint script.
ARG APP_USER_NAME=appuser
ARG APP_USER_UID=1001
ARG APP_USER_GID=1001

# Set the working directory
WORKDIR /nucleobench

# Get needed OS tools.
USER root
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    libssl-dev \
    curl && \
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
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} nucleobench /nucleobench/nucleobench

#(otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  

# Set the entrypoint.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "docker_entrypoint.py"]