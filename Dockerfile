# To build:
    # docker build -t nucleobench -f Dockerfile .

# IMPORTANT: The docker image for Google Batch must be linux/amd64.
# To build for linux/amd64:
    # docker buildx build --platform linux/amd64 -t nucleobench_linuxamd64:latest -f Dockerfile . --load

# To test:
    # hadolint Dockerfile

# To run locally:
    # docker run nucleobench

# To build and push to docker hub:
    # docker login
    # docker buildx build --platform linux/amd64 -t joelshor/nucleobench:latest -f Dockerfile . --push

# To push to Github Container Registry:
    # ** ghcr login **
    # docker buildx build --platform linux/amd64 -t ghcr.io/move37-labs/nucleobench:latest -f Dockerfile . --push

# ==============================================================================
# Builder Stage
# ==============================================================================
# This stage creates the full micromamba environment. The resulting /opt/conda
# directory will be copied to the final stage.
FROM mambaorg/micromamba:2.0.5 AS builder

# Create a user to avoid running as root.
USER root
ARG APP_USER_NAME=appuser
ARG APP_USER_UID=1001
ARG APP_USER_GID=1001
RUN groupadd --gid ${APP_USER_GID} ${APP_USER_NAME} && \
    useradd --uid ${APP_USER_UID} --gid ${APP_USER_GID} --create-home --shell /bin/bash ${APP_USER_NAME}

# Copy the environment file with correct ownership.
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} environment.yml /tmp/environment.yml

# Install all dependencies into the base environment.
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# ==============================================================================
# Final Stage
# ==============================================================================
# This stage creates the final, smaller image by copying the pre-built mamba
# environment and the application code.
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
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Create the application group and user.
RUN groupadd --gid ${APP_USER_GID} ${APP_USER_NAME} && \
    useradd --uid ${APP_USER_UID} --gid ${APP_USER_GID} --create-home --shell /bin/bash ${APP_USER_NAME}

# Copy the environment file with correct ownership for the appuser.
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} environment.yml /tmp/environment.yml

# Copy the pre-built environment from the builder stage.
COPY --from=builder /opt/conda /opt/conda

# Copy the application code.
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} docker_entrypoint.py /nucleobench/
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} docker_entrypoint_test.py /nucleobench/
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} nucleobench /nucleobench/nucleobench

#(otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  

# Set the entrypoint.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "docker_entrypoint.py"]