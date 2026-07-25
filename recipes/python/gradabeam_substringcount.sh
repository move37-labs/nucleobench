#
# bash recipes/python/gradabeam_substringcount.sh
#
python -m docker_entrypoint \
    --model substring_count \
        --substring 'ATGTC' \
    --optimization gradabeam \
        --beam_size 2 \
        --n_rollouts_per_root 4 \
        --mutations_per_sequence 2 \
        --exploration_alpha 0.05 \
        --rng_seed 0 \
    --max_seconds 15 \
    --optimization_steps_per_output 5 \
    --proposals_per_round 2 \
    --output_path ./output/python_recipe/gradabeam_substringcount \
    --start_sequence AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA