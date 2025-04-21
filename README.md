# NucleoBench

This is the initial repo for an upcoming paper, `NucleoBench: A Large-Scale Benchmark of Neural Nucleic Acid Design Algorithms`.

This repo is covered by the MIT license.

This repo is intended to be used in a few ways:

1. Reproducing the results from our paper.
1. Running the NucleoBench sequence designers on custom problems.
1. Using our new designer, AdaBeam, on a custom problem.

To do these, you can clone this repo, use the Docker image (for the benchmark), or use the PyPi package for our designers.

## Results

![Summary of results.](assets/images/results_summary.png)

## Installation & testing

Once this repo is cloned, you can make the conda/mamba/micromamba environment with:

```bash
conda env create -f environment.yml
conda activate nucleobench
```

To test that you've install NucleoBench, run all the unittests:

```bash
pytest nucleobench/
```

You can also run the integration tests, which require an internet connection:

```bash
pytest docker_entrypoint_test.py
```

Nucleic acid design benchmark.

## Running NucleoBench

See the folder `recipes` for examples of how to run the designer locally.

## Building a Docker image

To help deploy NucleoBench to the cloud, we've created a docker container. To build it yourself, see the top of `Dockerfile` for instructions. One way of creating a docker file is:

```bash
docker build -t nucleobench -f Dockerfile .
```