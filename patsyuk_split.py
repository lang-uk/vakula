#!/usr/bin/env python3
# patsyuk.py - like Patsyuk who split dumplings with supernatural powers, 
# this script splits parallel corpora into smaller files
# Character from Hohol's "Christmas Eve" (Ніч перед Різдвом)

"""
Script for splitting combined parallel corpora into smaller files.

This module reads a combined corpus file and splits unique records into
multiple files of specified size, using configurable naming patterns.
"""

import argparse
import json
import pathlib
from typing import Dict, Generator, Set
import smart_open
from tqdm import tqdm


def read_unique_records(
    input_file: pathlib.Path,
    seen_hashes: Set[str] = None
) -> Generator[Dict[str, str], None, None]:
    """Read unique records from input file, tracking by hash.
    
    Args:
        input_file: Path to the input jsonlines file.
        seen_hashes: Optional set of hashes to track uniqueness.
        
    Yields:
        Unique translation records.
    """
    if seen_hashes is None:
        seen_hashes = set()
        
    with smart_open.open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading records"):
            record = json.loads(line)
            if record["hash"] not in seen_hashes:
                seen_hashes.add(record["hash"])
                yield record


def write_split_files(
    records: Generator[Dict[str, str], None, None],
    output_pattern: str,
    records_per_file: int,
    start_index: int = 0
) -> int:
    """Write records to multiple files of specified size.
    
    Args:
        records: Generator of translation records.
        output_pattern: Pattern for output filenames (must include {}).
        records_per_file: Number of records per output file.
        start_index: Starting index for file numbering.
        
    Returns:
        Number of files created.
    """
    current_file = None
    current_count = 0
    file_index = start_index
    total_records = 0
    
    try:
        for record in records:
            if current_file is None or current_count >= records_per_file:
                if current_file is not None:
                    current_file.close()
                
                output_path = output_pattern.format(file_index)
                current_file = smart_open.open(output_path, "w", encoding="utf-8")
                current_count = 0
                file_index += 1
            
            current_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            current_count += 1
            total_records += 1
            
    finally:
        if current_file is not None:
            current_file.close()
    
    return file_index - start_index, total_records


def main():
    """Main function to orchestrate corpus splitting."""
    parser = argparse.ArgumentParser(
        description="Split combined parallel corpus into smaller files"
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Input jsonlines file (combined corpus)"
    )
    parser.add_argument(
        "--output-pattern",
        required=True,
        help="Pattern for output files (must include {}), e.g. 'split_{}.jsonl.gz'"
    )
    parser.add_argument(
        "--records-per-file",
        type=int,
        default=100000,
        help="Number of records per output file (default: 100000)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for file numbering (default: 0)"
    )
    args = parser.parse_args()
    
    # Validate output pattern
    if "{}" not in args.output_pattern:
        raise ValueError("Output pattern must include {} for file numbering")
    
    # Create output directory if needed
    output_dir = pathlib.Path(args.output_pattern).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process records
    records = read_unique_records(args.input)
    num_files, total_records = write_split_files(
        records=records,
        output_pattern=args.output_pattern,
        records_per_file=args.records_per_file,
        start_index=args.start_index
    )
    
    print("\nProcessing complete:")
    print(f"Total records processed: {total_records:,}")
    print(f"Files created: {num_files}")
    

if __name__ == "__main__":
    main()