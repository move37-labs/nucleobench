# NucleoBench: A Large-Scale Benchmark of Neural Nucleic Acid Design Algorithms

**We have developed a new, large-scale benchmark to compare modern nucleic acid sequence design algorithms (NucleoBench). We also present a new hybrid design algorithm that outperforms existing designers (AdaBeam).  Please see [https://github.com/move37-labs/nucleobench](https://github.com/move37-labs/nucleobench) for more details.**

[comment]: <> (Consider an image here.)

**NucleBench** is a large-scale comparison of modern sequence design algorithms across 16 biological tasks (such as
transcription factor binding and gene expression) and 9 design algorithms. NucleoBench, compares design algorithms on the same
tasks and start sequences across more than 400K experiments, allowing us to derive unique modeling insights on the importance of using gradient information, the role of randomness, scaling properties, and reasonable starting hyperparameters on
new problems. We use these insights to present a novel hybrid design algorithm, **AdaBeam**, that outperforms existing algorithms on 11 of 16 tasks and demonstrates superior scaling properties on long sequences and large predictors. Our benchmark and algorithms are freely available online.

We describe NucleoBench and AdaBeam in the paper ["NucleoBench: A Large-Scale Benchmark of Neural Nucleic Acid Design
Algorithms"](https://www.biorxiv.org/content/10.1101/2025.06.20.660785), to appear at the 2025 ICML GenBio Workshop.

This repo is intended to be used in a few days:
1. Run any of the NucleoBench design algorithms on a new design problem.
1. Run AdaBeam on a new design problem.
1. Run a new design algorithm on NucleoBench tasks, and avoid recomputing performances for existing designers.

![results](assets/images/results_summary.png)

![results](assets/images/benchmarks.png)

![results](assets/images/tasks.png)

![results](assets/images/designers.png)

## Contents

- [Setup](#setup)
  - [Installation](#installation)
    - [PyPi](#pypi)
    - [Source](#source)
    - [Docker](#docker)
- [Usage](#usage)
    - [Recipes](#recipes)
    - [Python, commandline](#python-commandline)
    - [Docker, commandline](#docker-commandline)
    - [Python, code](#python-code)
- [Citation](#citation)

## Setup

NucleoBench is provided via **PyPi**, **source**, or **Docker**.

### Installation

#### PyPi

```bash
pip install nucleobench  # optimizers and tasks
pip install nucleopt  # smaller, faster install for just optimizers
```

Then you can use it in python:
```python
from nucleobench import optimizations
opt = optimizations.get_optimization('beam_search_unordered')  # Any optimizer name.
```

#### Source

```bash
# Clone the repo.
git clone https://github.com/move37-labs/nucleobench.git
cd nucleobench

# Create and activate the conda environment.
conda env create -f environment.yml
conda activate nucleobench

# Run all the unittests.
pytest nucleobench/
```

You can also run the integration tests, which require an internet connection:

```bash
pytest docker_entrypoint_test.py
```

#### Docker

To help deploy NucleoBench to the cloud, we've created a Docker container. To build it yourself, see the top of `Dockerfile` for instructions. One way of creating a docker file is:

```bash
docker build -t nucleobench -f Dockerfile .
```

## Usage

### Recipes

See the `recipes/colab` folder for examples of how to run the designers with PyPi.
See the `recipes/docker` folder for examples of how to run the designers with Docker.
See the `recipes/python` folder for examples of how to run the designers with the cloned github repo.


### Python, commandline

An example of how to run on the commandline, using Python:

```bash
python -m docker_entrypoint \
    --model bpnet \
        --protein 'ATAC' \
    --optimization adabeam \
        --beam_size 2 \
        --n_rollouts_per_root 4 \
        --mutations_per_sequence 2 \
        --rng_seed 0 \
    --max_seconds 240 \
    --optimization_steps_per_output 5 \
    --proposals_per_round 2 \
    --output_path ./python_recipe/adabeam_atac \
    --start_sequence {YOUR START SEQUENCE}
```
### Docker, commandline

An example of how to run on the commandline, using Docker:

```bash
readonly output="./output/docker_recipe/adabeam_atac"
mkdir -p "${output}"
readonly fullpath="$(realpath $output)"

docker build -t nucleobench-docker -f Dockerfile .
docker run \
    -v "${fullpath}":"${fullpath}"  \
    "${docker_image_name}" \
    --model bpnet \
        --protein 'ATAC' \
    --optimization adabeam \
        --beam_size 2 \
        --n_rollouts_per_root 4 \
        --mutations_per_sequence 2 \
        --rng_seed 0 \
    --max_seconds 240 \
    --optimization_steps_per_output 5 \
    --proposals_per_round 2 \
    --output_path ${fullpath} \
    --start_sequence {YOUR START SEQUENCE}
```

### Python, code

Below is an example of how to download NucleoBench and use it:
```python
"""Initialize the task."""
from nucleobench import models
# Design for a simple task: count the number of occurances of a particular substring.
# See `nucleobench.models.__init__` for a registry of tasks, or add your own.
model_obj = models.get_model('substring_count')

# Every task has some baseline, default arguments to initialize. We can use
# these to demonstrate, or modify them for custom behavior. We do both, to
# demonstrate.
model_init_args = model_obj.debug_init_args()
model_init_args['substring'] = 'ATGTC'
model_fn = model_obj(**model_init_args)

"""Initialize the designer."""
from nucleobench import optimizations
# Pick a design algorithm that attemps to solve the task. In this case,
# maximize the number of substrings.
opt_obj = optimizations.get_optimization('adabeam')
# Every task has some baseline, default arguments to initialize. We can use
# these to demonstrate, or modify them for custom behavior. We do both, to
# demonstrate.
opt_init_args = opt_obj.debug_init_args()
opt_init_args['model_fn'] = model_fn
opt_init_args['start_sequence'] = 'A' * 100
designer = opt_obj(**opt_init_args)

"""Run the designer and show the results."""
designer.run(n_steps=100)
ret = designer.get_samples(1)
ret_score = model_fn(ret)
print(f'Final score: {ret_score[0]}')
print(f'Final sequence: {ret[0]}')
```

## Citation

Please cite the following publication when referencing NucleoBench or AdaBeam:

```
@inproceedings{nucleobench,
  author    = {Joel Shor and Erik Strand and Cory Y. McLean},
  title     = {{NucleoBench: A Large-Scale Benchmark of Neural Nucleic Acid Design Algorithms}},
  booktitle = {GenBio ICML 2025},
  year = {2025},
  publisher = {PMLR},
  url = {https://www.biorxiv.org/content/10.1101/2025.06.20.660785},
  doi = {10.1101/2025.06.20.660785},
}
```