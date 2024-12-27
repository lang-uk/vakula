#!/usr/bin/env python3
# cherevyk.py - compares two sets of scores and generates a histogram
# to show the impact of the matmul precision change on the scores

"""
Script for analyzing differences between two sets of model scores.

This module reads two sets of evaluation scores, calculates statistical metrics
of their differences, and generates a histogram visualization.
"""

import argparse
import json
import pathlib
from typing import Dict, List, Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
import smart_open
from tqdm import tqdm


def read_scores(file_path: pathlib.Path) -> Dict[str, float]:
    """Read scores from a jsonlines file.

    Args:
        file_path: Path to the input file.

    Returns:
        Dictionary mapping hashes to scores.
    """
    scores = {}
    with smart_open.open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {file_path.name}"):
            record = json.loads(line)
            scores[record["hash"]] = record["wmt23-cometkiwi-da-xxl_score"]
    return scores


def calculate_differences(
    scores1: Dict[str, float], scores2: Dict[str, float]
) -> Tuple[List[float], Set[str]]:
    """Calculate absolute differences between scores for matching hashes.

    Args:
        scores1: First set of scores.
        scores2: Second set of scores.

    Returns:
        List of absolute differences and set of skipped hashes.
    """
    differences = []
    skipped_hashes = set(scores1.keys()) ^ set(
        scores2.keys()
    )  # XOR to get non-matching

    common_hashes = set(scores1.keys()) & set(scores2.keys())
    for hash_value in tqdm(common_hashes, desc="Calculating differences"):
        diff = abs(scores1[hash_value] - scores2[hash_value])
        differences.append(diff)

    return differences, skipped_hashes


def calculate_statistics(differences: List[float]) -> Dict[str, float]:
    """Calculate statistical metrics for the differences.

    Args:
        differences: List of score differences.

    Returns:
        Dictionary of statistical metrics.
    """
    differences = np.array(differences)
    return {
        "min": float(np.min(differences)),
        "max": float(np.max(differences)),
        "mean": float(np.mean(differences)),
        "median": float(np.median(differences)),
        "variance": float(np.var(differences)),
        "std": float(np.std(differences)),
    }


def save_histograms(
    differences: List[float],
    output_full: pathlib.Path,
    output_filtered: pathlib.Path,
    bins: int = 50,
):
    """Generate and save two histograms of differences - full and without the largest bin.

    Args:
        differences: List of score differences.
        output_full: Path to save the full histogram.
        output_filtered: Path to save the filtered histogram.
        bins: Number of histogram bins.
    """
    # Generate full histogram first to get bin information
    plt.figure(figsize=(10, 6))
    hist_values, bin_edges, _ = plt.hist(differences, bins=bins, edgecolor="black")
    plt.title("Distribution of Score Differences (All Data)")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Count")
    plt.savefig(output_full)
    plt.close()

    # Find the largest bin
    max_bin_idx = np.argmax(hist_values)
    bin_start = bin_edges[max_bin_idx]
    bin_end = bin_edges[max_bin_idx + 1]

    # Filter out values from the largest bin
    filtered_diffs = [d for d in differences if d < bin_start or d >= bin_end]

    # Generate filtered histogram
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_diffs, bins=bins, edgecolor="black")
    plt.title("Distribution of Score Differences (Excluding Largest Bin)")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Count")
    plt.savefig(output_filtered)
    plt.close()


def main():
    """Main function to orchestrate score comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze differences between two sets of model scores"
    )
    parser.add_argument(
        "--input1",
        type=pathlib.Path,
        required=True,
        help="First input jsonlines file with scores",
    )
    parser.add_argument(
        "--input2",
        type=pathlib.Path,
        required=True,
        help="Second input jsonlines file with scores",
    )
    parser.add_argument(
        "--histogram-full",
        type=pathlib.Path,
        default="score_differences_full.png",
        help="Output file for full histogram (default: score_differences_full.png)",
    )
    parser.add_argument(
        "--histogram-filtered",
        type=pathlib.Path,
        default="score_differences_filtered.png",
        help="Output file for filtered histogram without largest bin (default: score_differences_filtered.png)",
    )
    parser.add_argument(
        "--bins", type=int, default=50, help="Number of histogram bins (default: 50)"
    )
    args = parser.parse_args()

    # Create output directory if needed
    args.histogram_full.parent.mkdir(parents=True, exist_ok=True)
    args.histogram_filtered.parent.mkdir(parents=True, exist_ok=True)

    # Read scores
    scores1 = read_scores(args.input1)
    scores2 = read_scores(args.input2)

    # Calculate differences
    differences, skipped_hashes = calculate_differences(scores1, scores2)

    # Calculate statistics
    stats = calculate_statistics(differences)

    # Generate histograms
    save_histograms(
        differences, args.histogram_full, args.histogram_filtered, args.bins
    )

    # Report results
    print("\nAnalysis complete:")
    print(f"Processed records: {len(differences):,}")
    print(f"Skipped records: {len(skipped_hashes):,}")
    print("\nStatistics:")
    for metric, value in stats.items():
        print(f"{metric}: {value:.6f}")
    print(f"\nHistograms saved to:")
    print(f"Full data: {args.histogram_full}")
    print(f"Without largest bin: {args.histogram_filtered}")


if __name__ == "__main__":
    main()
