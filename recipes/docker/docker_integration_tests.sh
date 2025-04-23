#!/bin/bash
#
# Integration test for docker image.
#
# To run:
# ```zsh
#   bash recipes/docker/docker_integration_tests.sh
# ````
#
# Steps:
# 1) Build new docker image
# 2) Run docker image with a few task/algo combinations.

# Fail on error.
set -e

source recipes/docker/docker_utils.sh

echo "***BUILDING DOCKER IMAGE***"
build_docker_image

# From this and below, commont or uncomment which tests you want to run.

echo "***RUNNING: Dummy opt, malinois***"
docker_run_dummy_malinois

echo "***RUNNING: FSP, malinois***"
docker_run_fsp_malinois

echo "***RUNNING: Adalead, malinois***"
docker_run_adalead_malinois

echo "***RUNNING: Simanneal, malinois***"
docker_run_simanneal_malinois