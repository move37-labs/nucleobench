import pickle
from pathlib import Path


def main():
    adalead_dir = Path(__file__).parent
    optimizations_dir = adalead_dir.parent
    nucleobench_dir = optimizations_dir.parent
    root_dir = nucleobench_dir.parent
    docker_test_dir = root_dir / "docker_entrypoint_test" / "simulated_annealing_malinois" / "simulated_annealing_malinois"
    assert docker_test_dir.exists()

    file = docker_test_dir / "20250206_232120" / "20250206_233120.pkl"

    data = pickle.load(file.open("rb"))
    for thing in data:
        print(thing["energies"].mean())


if __name__ == "__main__":
    main()
