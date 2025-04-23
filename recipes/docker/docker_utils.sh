# Utils for basic operatings around the docker image.
#
# Used in the docker integration tests, and example uses.
#

DOCKER_IMG_NAME="nucleobench-docker"
OUTPUT_DIR="docker_integration_tests"

function build_docker_image() {
    docker build -t "${DOCKER_IMG_NAME}" -f Dockerfile .
}

function docker_run_dummy_malinois() {
    docker run "${DOCKER_IMG_NAME}" \
        --seed_sequence AAAAAAAAA \
        --model dummy \
        --optimization dummy \
        --optimization_steps_per_output 20 \
        --proposals_per_round 2 \
        --max_number_of_rounds 10 \
        --output_path "${OUTPUT_DIR}/dummy"
}

function docker_run_fsp_malinois() {
    docker run "${DOCKER_IMG_NAME}" \
        --seed_sequence ${NT_200} \
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
        --output_path "${OUTPUT_DIR}/fsp_malinois"
}

function docker_run_adalead_malinois() {
    docker run "${DOCKER_IMG_NAME}" \
        --seed_sequence ${NT_200} \
        --model malinois \
            --target_feature 0 \
            --bending_factor 1.0 \
        --optimization adalead \
            --sequences_batch_size 256 \
            --model_queries_per_batch 1024 \
            --mutation_rate 0.001 \
            --recombination_rate 0.001 \
            --threshold 0.1 \
            --rho 1 \
            --eval_batch_size 1 \
            --rng_seed 42 \
        --optimization_steps_per_output 1 \
        --proposals_per_round 1 \
        --max_number_of_rounds 1 \
        --output_path "${OUTPUT_DIR}/adalead_malinois"
}


function docker_run_simanneal_malinois() {
    docker run "${DOCKER_IMG_NAME}" \
        --seed_sequence ${NT_200} \
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
        --output_path "${OUTPUT_DIR}/simanneal_malinois"
}