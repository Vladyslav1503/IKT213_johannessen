import time
import tracemalloc
from typing import Literal
from dataset_processor import process_dataset

DATASET_PATH = "./datasets/Dataset_1/"
RESULTS_PATH = "./results/"

Methods = Literal["orb", "sift_flann", "sift_bf"]
AVAILABLE_METHODS: tuple[Methods, ...] = ("orb", "sift_flann", "sift_bf")


def main():
    results = {}

    print("\nStarting image matching benchmark...\n")

    for method in AVAILABLE_METHODS:
        print(f"Running {method.upper()} ...")

        tracemalloc.start()
        start_wall = time.time()
        start_cpu = time.process_time()

        # Process dataset and get accuracy
        accuracy = process_dataset(method, DATASET_PATH, f"{RESULTS_PATH}/{method}")

        elapsed_wall = time.time() - start_wall       # Wall-clock time
        elapsed_cpu = time.process_time() - start_cpu  # CPU time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[method] = {
            "accuracy": accuracy * 100,  # percentage
            "wall_time": elapsed_wall,
            "cpu_time": elapsed_cpu,
            "memory": peak / 1024 / 1024,  # MB
        }

        print(f"{method.upper()} completed in {elapsed_wall:.2f}s (CPU {elapsed_cpu:.2f}s) "
              f"| Peak Memory: {results[method]['memory']:.2f} MB | Accuracy: {accuracy*100:.2f}%\n")

    # --- Summary Table ---
    print("\n" + "=" * 90)
    print("RESOURCE & PERFORMANCE COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<25} {'ORB':<15} {'SIFT_FLANN':<15} {'SIFT_BF':<15}")
    print("-" * 90)
    print(f"{'Accuracy (%)':<25}"
          f"{results['orb']['accuracy']:<15.2f}"
          f"{results['sift_flann']['accuracy']:<15.2f}"
          f"{results['sift_bf']['accuracy']:<15.2f}")
    print(f"{'Execution Time (s)':<25}"
          f"{results['orb']['wall_time']:<15.2f}"
          f"{results['sift_flann']['wall_time']:<15.2f}"
          f"{results['sift_bf']['wall_time']:<15.2f}")
    print(f"{'CPU Time (s)':<25}"
          f"{results['orb']['cpu_time']:<15.2f}"
          f"{results['sift_flann']['cpu_time']:<15.2f}"
          f"{results['sift_bf']['cpu_time']:<15.2f}")
    print(f"{'Peak Memory (MB)':<25}"
          f"{results['orb']['memory']:<15.2f}"
          f"{results['sift_flann']['memory']:<15.2f}"
          f"{results['sift_bf']['memory']:<15.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
