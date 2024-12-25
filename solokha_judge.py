#!/usr/bin/env python3
# solokha.py - because who's better at judging than Vakula's mother, the witch Solokha?
# Character from Hohol's "Christmas Eve" (Ніч перед Різдвом)

"""
Script for parallel evaluation of translation quality using COMET models.

This module processes translation pairs from a jsonlines file, evaluates their
quality using specified COMET models, and outputs scores in a resumable manner
using multiple GPUs.
"""

import argparse
import json
import pathlib
from typing import Dict, Generator, List, Set
import torch
import torch.multiprocessing as mp
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import smart_open


def read_processed_hashes(output_file: pathlib.Path) -> Set[str]:
    """Read already processed translation hashes from output file.

    Args:
        output_file: Path to the output jsonlines file.

    Returns:
        Set of already processed translation hashes.
    """
    processed_hashes = set()
    if output_file.exists():
        with smart_open.open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_hashes.add(data["hash"])
    return processed_hashes


def read_translations(
    input_file: pathlib.Path,
    batch_size: int,
    src_lang_field: str,
    tgt_lang_field: str,
    processed_hashes: Set[str] = None,
) -> Generator[List[Dict[str, str]], None, None]:
    """Read translation pairs from input file in batches.

    Args:
        input_file: Path to the input jsonlines file.
        batch_size: Number of translations to yield at once.
        src_lang_field: Field name for source text in the input file.
        tgt_lang_field: Field name for target text in the input file.
        processed_hashes: Set of hashes to skip (for resuming).

    Yields:
        Batches of translation pairs.
    """
    current_batch = []
    with smart_open.open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # Skip if already processed
            if processed_hashes and data["hash"] in processed_hashes:
                continue

            pair = {
                "hash": data["hash"],
                "source": data[src_lang_field],
                "hypothesis": data[tgt_lang_field],
            }

            current_batch.append(pair)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if current_batch:  # Yield remaining items
            yield current_batch


def evaluate_batch(
    gpu_id: int, model_name: str, batch: List[Dict[str, str]], batch_size: int
) -> List[Dict[str, float]]:
    """Evaluate a batch of translations using COMET model.

    Args:
        gpu_id: GPU device ID to use.
        model_name: Name of the COMET model to use.
        batch: List of translation pairs to evaluate.
        batch_size: Batch size for model inference.

    Returns:
        List of dictionaries containing hashes and scores.
    """
    torch.cuda.set_device(gpu_id)
    model = load_from_checkpoint(model_name)
    model.eval()

    data = {
        "src": [pair["source"] for pair in batch],
        "mt": [pair["hypothesis"] for pair in batch],
    }

    scores = []
    for i in range(0, len(batch), batch_size):
        batch_data = {k: v[i : i + batch_size] for k, v in data.items()}

        with torch.no_grad():
            batch_scores = model.predict(
                batch_data, batch_size=batch_size, gpus=[gpu_id]
            )

        for pair, score in zip(batch[i : i + batch_size], batch_scores):
            scores.append(
                {
                    "hash": pair["hash"],
                    f"{model_name.split('/')[-1]}_score": float(score),
                }
            )

    return scores


def evaluate_worker(
    gpu_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    model_name: str,
    eval_batch_size: int,
):
    """Worker process for parallel evaluation.

    Args:
        gpu_id: GPU device ID to use.
        input_queue: Queue for receiving translation batches.
        output_queue: Queue for sending evaluation results.
        model_name: Name of the COMET model to use.
        eval_batch_size: Batch size for model inference.
    """
    try:
        while True:
            batch = input_queue.get()
            if batch is None:  # Poison pill
                break

            scores = evaluate_batch(gpu_id, model_name, batch, eval_batch_size)
            output_queue.put(scores)
    except Exception as e:
        output_queue.put(f"Error in worker {gpu_id}: {str(e)}")


def main():
    """Main function to orchestrate parallel translation quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate translation quality using COMET models in parallel"
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Input jsonlines file with translations",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output jsonlines file for scores",
    )
    parser.add_argument(
        "--model",
        choices=["Unbabel/wmt23-cometkiwi-da-xxl", "Unbabel/wmt23-cometkiwi-da-xl"],
        required=True,
        help="COMET model to use",
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", required=True, help="GPU device IDs to use"
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=3200,
        help="Batch size for reading translations",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=16, help="Batch size for model inference"
    )
    parser.add_argument(
        "--src-field",
        default="en",
        help="Field name for source text in input file (default: en)",
    )
    parser.add_argument(
        "--tgt-field",
        default="uk",
        help="Field name for target text in input file (default: uk)",
    )
    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Read processed hashes for resuming
    processed_hashes = read_processed_hashes(args.output)
    print(f"Found {len(processed_hashes):,} already processed translations")

    # Download model in the main process
    print(f"Downloading model {args.model}...")
    download_model(args.model)

    # Initialize multiprocessing queues
    mp.set_start_method("spawn", force=True)
    input_queue = mp.Queue(maxsize=len(args.gpus) * 2)
    output_queue = mp.Queue()

    # Start worker processes
    workers = []
    for gpu_id in args.gpus:
        worker = mp.Process(
            target=evaluate_worker,
            args=(gpu_id, input_queue, output_queue, args.model, args.eval_batch_size),
        )
        worker.start()
        workers.append(worker)

    # Process translations
    active_batches = 0
    translation_iterator = read_translations(
        args.input,
        args.read_batch_size,
        args.src_field,
        args.tgt_field,
        processed_hashes,
    )

    with tqdm(desc="Evaluating translations") as pbar:
        try:
            # Start initial batches
            for batch in translation_iterator:
                input_queue.put(batch)
                active_batches += 1
                if active_batches >= len(args.gpus) * 2:
                    break

            # Process results and add new batches
            while active_batches > 0:
                scores = output_queue.get()

                if isinstance(scores, str):  # Error message
                    raise RuntimeError(scores)

                # Write scores
                with smart_open.open(args.output, "a", encoding="utf-8") as f:
                    for score in scores:
                        f.write(json.dumps(score, ensure_ascii=False) + "\n")

                pbar.update(len(scores))
                active_batches -= 1

                # Add new batch if available
                try:
                    next_batch = next(translation_iterator)
                    input_queue.put(next_batch)
                    active_batches += 1
                except StopIteration:
                    pass

        finally:
            # Clean up workers
            for _ in workers:
                input_queue.put(None)
            for worker in workers:
                worker.join()


if __name__ == "__main__":
    main()
