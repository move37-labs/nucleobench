#
# bash recipes/python/adalead_malinois.sh
#
python -m docker_entrypoint \
    --model malinois \
        --target_feature 1 \
        --bending_factor 1.0 \
    --optimization adalead \
        --sequences_batch_size 2 \
        --model_queries_per_batch 50 \
        --mutations_per_sequence 1 \
        --recombinations_per_sequence 1 \
        --threshold 0.1 \
        --rho 1 \
        --eval_batch_size 1 \
        --rng_seed 0 \
    --max_seconds 3000 \
    --optimization_steps_per_output 50 \
    --proposals_per_round 1 \
    --output_path ./docker_entrypoint_test/adalead_malinois \
    --seed_sequence GATAAGTGACACGGTGCAGCTCGGGTATCGTCTACGGGTGAAAACGGAAGGGTTCTATCCCATGTGGCCTGCTGACCTACGCACGATAATGAGCATTTAAGTAAGTCGGTGGGCTTTCACATGTTTACCGTCGGGCTCGAAGGCGGGTCCGGAAAACTAATTTCGGATCACCCTACCCAGGACGAACGTCGGGGGTGGCC \
