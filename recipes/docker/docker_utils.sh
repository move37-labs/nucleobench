# Utils for basic operatings around the docker image.
#
# Used in the docker integration tests, and example uses.
#
# TODO(joelshor): Add example that writes output to GCP bucket.
#

DOCKER_IMG_NAME="nucleobench-docker"
LOCAL_OUTPUT_DIR="docker_integration_tests3"

# An example start sequence.
readonly NT_200="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


function build_docker_image() {
    docker build -t "${DOCKER_IMG_NAME}" -f Dockerfile .
}

function setup_outputdir(){
    local output_dir=$1
    mkdir -p "${output_dir}"
}

function docker_run_dummy_malinois() {
    local output="${LOCAL_OUTPUT_DIR}/dummy_malinois"
    setup_outputdir "${output}"
    local fullpath="$(realpath $output)"

    docker run -v "${fullpath}":"${fullpath}"  "${DOCKER_IMG_NAME}" \
        --start_sequence AAAAAAAAA \
        --model dummy \
        --optimization dummy \
        --optimization_steps_per_output 20 \
        --proposals_per_round 2 \
        --max_number_of_rounds 10 \
        --output_path "${fullpath}"
}

function docker_run_fsp_malinois() {
    local output="${LOCAL_OUTPUT_DIR}/fsp_malinois"
    setup_outputdir "${output}"
    local fullpath="$(realpath $output)"

    docker run -v "${fullpath}":"${fullpath}"  "${DOCKER_IMG_NAME}" \
        --start_sequence ${NT_200} \
        --model malinois \
            --target_feature 0 \
            --bending_factor 1.0 \
        --optimization fastseqprop \
            --learning_rate 0.1 \
            --eta_min 1e-6 \
            --batch_size 1 \
        --optimization_steps_per_output 1 \
        --proposals_per_round 1 \
        --max_number_of_rounds 1 \
        --output_path "${fullpath}"
}

function docker_run_ledidi_gata2() {
    local output="${LOCAL_OUTPUT_DIR}/fsp_malinois"
    setup_outputdir "${output}"
    local fullpath="$(realpath $output)"

    docker run -v "${fullpath}":"${fullpath}"  "${DOCKER_IMG_NAME}" \
        --start_sequence ${NT_200} \
        --model bpnet \
            --protein "GATA2" \
        --optimization ledidi \
            --lr 0.1 \
            --train_batch_size 1 \
        --optimization_steps_per_output 1 \
        --proposals_per_round 1 \
        --max_number_of_rounds 1 \
        --output_path "${fullpath}"
}

function docker_run_simanneal_malinois() {
    local output="${LOCAL_OUTPUT_DIR}/simanneal_malinois"
    setup_outputdir "${output}"
    local fullpath="$(realpath $output)"

    docker run -v "${fullpath}":"${fullpath}"  "${DOCKER_IMG_NAME}" \
        --start_sequence ${NT_200} \
        --model malinois \
            --target_feature 0 \
            --bending_factor 1.0 \
        --optimization simulated_annealing \
            --polynomial_decay_a 1.0 \
            --polynomial_decay_b 1.0 \
            --polynomial_decay_p 0.1 \
            --n_mutations_per_proposal 1 \
            --rng_seed 42 \
        --optimization_steps_per_output 1 \
        --proposals_per_round 1 \
        --max_number_of_rounds 1 \
        --output_path "${fullpath}"
}