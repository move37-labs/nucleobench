# syntax=docker/dockerfile:1
# ^ This line is important to enable advanced caching features.
# To build:
    # docker build -t nucleobench -f Dockerfile .

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
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} environment_runtime.yml /tmp/environment.yml

# OPTIMIZATION: Add pip cache mount.
# Even if PyTorch is in Mamba, 'gReLU' and its deps are in Pip.
# Caching /root/.cache/pip prevents redownloading pip wheels on every build.
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    micromamba install -y -n base -f /tmp/environment.yml

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

# Copy the pre-built environment from the builder stage.
COPY --from=builder --chown=${APP_USER_NAME}:${APP_USER_NAME} /opt/conda /opt/conda

# Copy the application code.
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} docker_entrypoint.py /nucleobench/
COPY --chown=${APP_USER_NAME}:${APP_USER_NAME} nucleobench /nucleobench/nucleobench

# Switch to non-root user
USER ${APP_USER_NAME}

# Force /nucleobench into the path.
# We don't include ${PYTHONPATH} because it is empty in the base image,
# and we want to avoid a leading colon (:/nucleobench).
ENV PYTHONPATH="/nucleobench"

#(otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["python", "docker_entrypoint.py"]