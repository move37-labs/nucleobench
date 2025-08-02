## NucleoBench

**A large-scale benchmark for modern nucleic acid sequence design algorithms (NucleoBench), and a new design algorithm that outperforms existing designers (AdaBeam).  Link to ICML GenBio 2025 workshop paper [here](https://www.biorxiv.org/content/10.1101/2025.06.20.660785).**

[comment]: <> (Consider an image here.)

This repo is intended to be used in a few ways:
1. Design a DNA sequence with selective expression in a cell-type (or any other target property in the benchmark, see list [here](#summary-of-tasks-in-nucleobench)), using the AdaBeam algorithm (or any of the ones listed [here](#summary-of-designers-in-nucleobench))
2. Design a DNA sequence with high binding affinity for a specific transcription factor (such as the ones listed [here](#summary-of-tasks-in-nucleobench)), using the AdaBeam algorithm (or any of the ones listed [here](#summary-of-designers-in-nucleobench))
1. Design a DNA or RNA sequence for a new task, using any designer (see tutorial [here](https://github.com/move37-labs/nucleobench/blob/documentation/recipes/colab/custom_task.ipynb))
1. Run a new design algorithm on NucleoBench tasks.


### Citation

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

## Contents

- [Quick Start](#quick_start)
  - [1 minute install w/ pip](#get-started-in-1-minute-pip-install)
  - [3  minute install w/ docker](#get-started-in-3-minutes-docker-image-pull)
  - [5 minute install w/ source](#get-started-in-5-minutes-git-clone)
- [Details](#details)
- [FAQ](#faq)

## Quick Start

NucleoBench is provided via **PyPi**, **Docker**, or **source**.

### Get started in 1 minute (pip install)

Install `nucleobench` on your terminal:
```bash
# Choose one.
pip install nucleobench  # optimizers and tasks
pip install nucleopt  # smaller, faster install for just optimizers
```

Then run in Python:
```python
# 1. Choose a model (task).
from nucleobench import models
model = models.get_model('substring_count')
model_init_args = model.debug_init_args()
model_init_args['substring'] = 'ATGTC'
model_fn = model_obj(**model_init_args)

# 2. Choose an optimizer.
from nucleobench import optimizations
opt_init_args = opt_obj.debug_init_args()
opt_init_args['model_fn'] = model_fn
opt_init_args['start_sequence'] = 'A' * 100
designer = opt_obj(**opt_init_args)

# 3. Run the designer and show the results.
designer.run(n_steps=100)
ret = designer.get_samples(1)
ret_score = model_fn(ret)
print(f'Final score: {ret_score[0]}')
print(f'Final sequence: {ret[0]}')
```

Output:
```bash
Step 99 current scores: [np.float64(508.0), np.float64(507.0), np.float64(506.0), np.float64(505.0), np.float64(504.0), np.float64(503.0), np.float64(503.0), np.float64(502.0), np.float64(502.0), np.float64(502.0)]
Final score: -508.0
Final sequence: AGATGTCATATATGATGTCATGTCATGTCGTCATGTCTGTCTCTCATGTATGTCATGTCTATGTCTGTCTATGTCTATGTCTATGTCATGTCTATGTCTC
```

This "recipe" can be found under [`recipes/python/adabeam_substring.py`](https://github.com/move37-labs/nucleobench/blob/main/recipes/python/adabeam_substringcount.py).

### Get started in 3 minutes (docker image pull)

Get the image:
```bash
docker image pull joelshor/nucleobench:latest
```

Output:
```bash
latest: Pulling from joelshor/nucleobench
Digest: sha256:602230b568c0f15acfa7a0b6723ffb16fab6f32c37ae5b88c71763fb722ab5c3
Status: Image is up to date for joelshor/nucleobench:latest
docker.io/joelshor/nucleobench:latest
```

Make a directory for output:
```bash
readonly output="./output/docker_recipe/adabeam_atac"
mkdir -p "${output}"
readonly fullpath="$(realpath $output)"
```

Then run it:
```bash
docker run \
    -v "${fullpath}":"${fullpath}" \
    joelshor/nucleobench:latest \
    --model substring_count \
        --substring 'ATGTC' \
    --optimization adabeam \
        --beam_size 2 \
        --n_rollouts_per_root 4 \
        --mutations_per_sequence 2 \
        --rng_seed 0 \
    --max_seconds 15 \
    --optimization_steps_per_output 5 \
    --proposals_per_round 2 \
    --output_path ${fullpath} \
    --start_sequence AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
```

Output:
```bash
Completed round 3441 (5 steps) took 0.00s. Avg 0.00s per step.
Proposals deposited at:
	/Users/joelshor/Desktop/docker_test/output/docker_recipe/adabeam_atac/adabeam_substring_count/20250731_194857/20250731_194912.pkl
```

This "recipe" can be found under [`recipes/docker/adabeam_atac.sh`](https://github.com/move37-labs/nucleobench/blob/main/recipes/docker/adabeam_atac.sh).

### Get started in 5 minutes (git clone)

```bash
git clone https://github.com/move37-labs/nucleobench.git
cd nucleobench
conda env create -f environment.yml
conda activate nucleobench
```

Now run the main entrypoint:

```bash
python -m docker_entrypoint \
    --model substring_count \
        --substring 'ATGTC' \
    --optimization adabeam \
        --beam_size 2 \
        --n_rollouts_per_root 4 \
        --mutations_per_sequence 2 \
        --rng_seed 0 \
    --max_seconds 15 \
    --optimization_steps_per_output 5 \
    --proposals_per_round 2 \
    --output_path ./output/python_recipe/adabeam_atac \
    --start_sequence AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
```

Output:
```bash
...
Completed round 3820 (5 steps) took 0.00s. Avg 0.00s per step.
  0%|                              | 3821/99999999 [00:14<109:01:33, 254.77it/s]
Proposals deposited at:
	./output/python_recipe/adabeam_atac/adabeam_substring_count/20250731_162119/20250731_162134.pkl
```

This "recipe" can be found under [`recipes/python/adabeam_atac.py`](https://github.com/move37-labs/nucleobench/blob/main/recipes/python/adabeam_atac.py).

## Details

**NucleoBench** is a large-scale comparison of modern sequence design algorithms across 16 biological tasks (such as transcription factor binding and gene expression) and 9 design algorithms. NucleoBench, compares design algorithms on the same tasks and start sequences across more than 400K experiments, allowing us to derive unique modeling insights on the importance of using gradient information, the role of randomness, scaling properties, and reasonable starting hyperparameters on new problems. We use these insights to present a novel hybrid design algorithm, **AdaBeam**, that outperforms existing algorithms on 11 of 16 tasks and demonstrates superior scaling properties on long sequences and large predictors. Our benchmark and algorithms are freely available online.

<img src="assets/images/results_summary.png" alt="results" style="width: 70%; max-width: 800px; height: auto; display: block; margin: auto;" />

### Comparison of nucleic acid design benchmarks

| NAME | YEAR | ALGOS | TASKS | SEQ. LENGTH (BP) | DESIGN BENCHMARK | LONG SEQS | LARGE MODELS | PAIRED START SEQS. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fitness Landscape Exploration Sandbox | 2020 | 4-6 | 9 | Most <100 | ✅ | ❌ | ❌ | ✅ |
| Computational Optimization of DNA Activity | 2024 | 3 | 3 | 200 | ✅ | ❌ | ❌ | ✅ |
| gRelu | 2024 | 2 | 5 | 500K (20 edit) | ❌ | ❌ | ✅ | ❌ |
| Linder et al repos | 2021 | 2 | 20 | <600 | ✅ | ❌ | ❌ | ❌ |
| NucleoBench (ours) | 2025 | 9 | 16 | 256-3K | ✅ | ✅ | ✅ | ✅ |

<small>Table: Nucleic acid design from sequence benchmarks. All benchmarks prior to NucleoBench are limited either in the range of tasks
they measure against, the range of optimizations they compare, or the complexity of the task.</small>

### Summary of tasks in NucleoBench

| TASK CATEGORY | MODEL | DESCRIPTION | NUM TASKS | SEQ LEN (BP) | SPEED (MS / EXAMPLE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cell-type specific cis-regulatory activity | Malinois | How DNA sequences control gene expression from the same DNA molecule. Cell types are: *precursor blood cells*, *liver cells*, *neuronal cells*. | 3 | 200 | 2 |
| Transcription factor binding | BPNet-lite | How likely a specific transcription factor (TF) will bind to a particular stretch of DNA. Specific TFs: *CTCF*, *E2F3*, *ELF4*, *GATA2*, *JUNB*, *MAX*, *MECOM*, *MYC*, *OTX1*, *RAD21*, *SOX6*| 11 | 3000 | 55 |
| Chromatin accessibility | BPNet-lite | Chromatin accessibility | 1 | 3000 | 260 |
| Selective gene expression | Enformer | Prediction of gene expression. We optimize for *maximal expression in muscle cells, minimal expression in liver cells*.| 1 | 196,608 / 256 * | 15,000 |

<small>*Input length is 200K, but only 256 bp are edited.</small>

### Summary of designers in NucleoBench

| Algo | Description | Gradient-based |
| :--- | :--- | :--- |
| Directed Evolution | Random mutations, track the best. | ❌ |
| Simulated Annealing | Greedy optimization with random jumps. | ❌ |
| [AdaLead](https://arxiv.org/abs/2010.02141) | Iterative combining and mutating of a population of sequences. | ❌ |
| [FastSeqProp](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04437-5) | Sampling and the straight-through estimator for maximal input. | ✅ |
| [Ledidi](https://www.biorxiv.org/content/10.1101/2020.05.21.109686v1) | Sampling and the gumbel softmax estimator for maximal input. | ✅ |
| --- |
| Ordered Beam | Greedy search, in fixed sequence order, with cache. | ❌ |
| Unordered Beam | Greedy search with cache. | ❌ |
| Gradient Evo | Directed Evolution, guided by model gradients. | ✅ |
| [AdaBeam (ours)](https://www.biorxiv.org/content/10.1101/2025.06.20.660785) | Hybrid of Unordered Beam and improved AdaLead. | ❌ |

<small>Table: Summary of designers in NucleoBench. Above the solid line are designers already found in the nucleic acid design literature.
Below the line are designers from the search literature not previously used to benchmark nucleic acid sequence design and hybrid
algorithms devised in this work.</small>

## FAQ

1. How can I add a new task to NucleoBench?
    A: Follow [this](https://github.com/move37-labs/nucleobench/blob/documentation/recipes/colab/custom_task.ipynb) colab notebook.