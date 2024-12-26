#!/usr/bin/env python3
# solokha.py - because who's better at judging than Vakula's mother, the witch Solokha?
# Character from Hohol's "Christmas Eve" (Ніч перед Різдвом)

"""
Script for parallel evaluation of translation quality using COMET models.

This module processes translation pairs from a jsonlines file, evaluates their
quality using specified COMET models, and outputs scores using multiple GPUs.
"""

import argparse
import json
import pathlib
from multiprocessing import cpu_count
from typing import Dict, Generator, List, Set
import torch
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
                "src": data[src_lang_field],
                "mt": data[tgt_lang_field],
            }
            processed_hashes.add(data["hash"])

            current_batch.append(pair)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if current_batch:  # Yield remaining items
            yield current_batch


def evaluate_batch(
    model,
    model_name: str,
    batch: List[Dict[str, str]],
    gpus: int,
    eval_batch_size: int = 8,
) -> List[Dict[str, float]]:
    """Evaluate a batch of translations using COMET model.

    Args:
        model: Loaded COMET model instance.
        model_name: Name of the COMET model.
        batch: List of translation pairs to evaluate.
        eval_batch_size: Batch size for model inference.
        gpus: Number of GPUs to use.

    Returns:
        List of dictionaries containing hashes and scores.
    """

    with torch.no_grad():
        batch_scores = model.predict(
            batch, batch_size=eval_batch_size, gpus=gpus, num_workers=cpu_count()
        )

    return [
        {
            "hash": pair["hash"],
            f"{model_name.split('/')[-1]}_score": float(score),
        }
        for pair, score in zip(batch, batch_scores["scores"])
    ]


def main():
    """Main function to orchestrate translation quality evaluation."""
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
        choices=[
            "Unbabel/wmt23-cometkiwi-da-xxl",
            "Unbabel/wmt22-cometkiwi-da",
            "Unbabel/wmt23-cometkiwi-da-xl",
        ],
        required=True,
        help="COMET model to use",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPU devices to use (0 for auto)",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=3200,
        help="Batch size for reading translations (default: 3200)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size for model evaluation (default: 8)",
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

    # Download and prepare model
    print(f"Downloading model {args.model}...")
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)
    model = model.cuda()
    model.eval()

    # Process translations
    translation_iterator = read_translations(
        args.input,
        args.read_batch_size,
        args.src_field,
        args.tgt_field,
        processed_hashes,
    )

    with tqdm(desc="Evaluating translations") as pbar:
        try:
            for batch in translation_iterator:
                scores = evaluate_batch(
                    model=model,
                    model_name=args.model,
                    batch=batch,
                    eval_batch_size=args.eval_batch_size,
                    gpus=args.gpus,
                )

                # Write scores
                with smart_open.open(args.output, "a", encoding="utf-8") as f:
                    for score in scores:
                        f.write(json.dumps(score, ensure_ascii=False) + "\n")

                pbar.update(len(scores))

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise e


if __name__ == "__main__":
    main()
