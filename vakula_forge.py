#!/usr/bin/env python3
# vakula_forge.py - because Vakula the Smith forges parallel texts just like he forged the slippers

"""
Script for downloading and processing parallel corpora from OPUS.

This module handles downloading TMX files from OPUS, parsing them to extract
sentence pairs, and storing the results in a compressed jsonlines format.
"""

import argparse
import json
import pathlib
import re
from hashlib import sha1
from typing import Dict, Generator, List, Set
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import smart_open
import xml.etree.ElementTree as ET
from tqdm import tqdm


def slugify(text: str) -> str:
    """Convert text into a URL slug.

    Args:
        text: Text to convert into a slug.

    Returns:
        A URL-friendly slug string.
    """
    # Replace slashes with underscores first
    text = text.strip("/").replace("/", "_")
    # Remove non-word characters and spaces
    text = re.sub(r"[^\w\s-]", "", text.lower())
    # Replace spaces and repeated dashes with single dash
    return re.sub(r"[-\s]+", "-", text).strip("-")


def get_tmx_links(url: str, source_lang: str, target_lang: str) -> List[Dict[str, str]]:
    """Extracts TMX file information from the OPUS webpage.

    Args:
        url: The URL of the OPUS corpus results page.
        source_lang: Source language code.
        target_lang: Target language code.

    Returns:
        A list of dictionaries containing TMX file information.

    Raises:
        requests.RequestException: If there's an error fetching the webpage.
        json.JSONDecodeError: If there's an error parsing the JSON data.
    """
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    next_data = soup.find("script", {"id": "__NEXT_DATA__"})

    if not next_data:
        raise ValueError("Could not find __NEXT_DATA__ script tag")

    data = json.loads(next_data.string)
    corpora = data["props"]["pageProps"]["corporaList"]["corpora"]

    tmx_links = []
    for corpus in corpora:
        if (
            corpus.get("source") == source_lang
            and corpus.get("target") == target_lang
            and corpus.get("url", "").endswith(".tmx.gz")
        ):
            tmx_links.append({"url": corpus["url"], "name": corpus.get("corpus", "")})

    return tmx_links


def download_file(url: str, output_dir: pathlib.Path, force: bool = False) -> str:
    """Downloads a file if it doesn't exist or if force download is enabled.

    Args:
        url: URL of the file to download.
        output_dir: Directory to save the downloaded file.
        force: Whether to force download even if file exists.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.RequestException: If there's an error downloading the file.
    """
    # Create slug from URL path
    url_parts = urlparse(url)
    slug = slugify(url_parts.path)
    filename = f"{slug}.tmx.gz"
    output_path = output_dir / filename

    if not output_path.exists() or force:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {filename}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

    return str(output_path)


def parse_xml_file(file_path: str) -> Generator[Dict[str, str], None, None]:
    """Parses an XML file to extract sentences in different languages using a stream parser.

    Args:
        file_path: The path to the XML file.

    Returns:
        An iterator over sentence pairs in different languages.
    """
    input_file = smart_open.open(file_path, "r", encoding="utf-8")
    context = ET.iterparse(input_file, events=("end",))

    for _, elem in context:
        if elem.tag == "tu":
            sentence_pair = {}
            for tuv in elem.findall("tuv"):
                lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                seg = tuv.find("seg")
                if lang and seg is not None and seg.text:
                    sentence_pair[lang] = seg.text.strip()

            if sentence_pair and len(sentence_pair) >= 2:
                yield sentence_pair
            elem.clear()


def calculate_hash(orig: str, trans: str) -> str:
    """Calculates a hash for a sentence pair.

    Args:
        orig: Original sentence.
        trans: Translated sentence.

    Returns:
        SHA1 hash of the normalized sentence pair.
    """
    return sha1(
        f"{orig.lower().strip()}:::{trans.lower().strip()}".encode("utf-8")
    ).hexdigest()


def process_tmx_file(
    file_path: pathlib.Path,
    output_file: str,
    source_name: str,
    source_lang: str,
    target_lang: str,
) -> tuple[int, Set[str]]:
    """Processes a TMX file and writes sentence pairs to output file.

    Args:
        file_path: Path to the TMX file.
        output_file: Path to the output jsonlines file.
        source_name: Name of the source corpus.
        source_lang: Source language code.
        target_lang: Target language code.

    Returns:
        Tuple containing count of processed sentences and set of unique hashes.
    """
    sentence_count = 0
    unique_hashes = set()

    with smart_open.open(output_file, "a") as f:
        for pair in tqdm(
            parse_xml_file(file_path), desc=f"Processing {source_name}", unit=" pairs"
        ):
            if source_lang in pair and target_lang in pair:
                hash_value = calculate_hash(pair[source_lang], pair[target_lang])
                unique_hashes.add(hash_value)

                output_record = {
                    source_lang: pair[source_lang],
                    target_lang: pair[target_lang],
                    "hash": hash_value,
                    "source": source_name,
                }

                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                sentence_count += 1

    return sentence_count, unique_hashes


def main():
    """Main function to orchestrate the download and processing of TMX files."""
    parser = argparse.ArgumentParser(description="Download and process OPUS TMX files")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download even if files exist",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="combined_corpora/processed_corpus.jsonl.gz",
        help="Output file path (default: processed_corpus.jsonl.gz)",
    )
    parser.add_argument(
        "--url",
        default="https://opus.nlpl.eu/results/en&uk/corpus-result-table",
        help="OPUS corpus result table URL",
    )
    parser.add_argument(
        "--source-lang", default="en", help="Source language code (default: en)"
    )
    parser.add_argument(
        "--target-lang", default="uk", help="Target language code (default: uk)"
    )
    args = parser.parse_args()

    # Create directories
    raw_data_dir = pathlib.Path("raw_data")
    raw_data_dir.mkdir(exist_ok=True)

    # Get TMX links
    try:
        tmx_info = get_tmx_links(args.url, args.source_lang, args.target_lang)
        print(f"Found {len(tmx_info)} TMX files to process")
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"Error getting TMX links: {e}")
        return

    # Download files
    downloaded_files = []
    for info in tmx_info:
        try:
            file_path = download_file(info["url"], raw_data_dir, args.force_download)
            downloaded_files.append((file_path, info["name"]))
        except requests.RequestException as e:
            print(f"Error downloading {info['url']}: {e}")

    # Process files
    total_sentences = 0
    all_unique_hashes = set()

    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.unlink(missing_ok=True)

    for file_path, source_name in downloaded_files:
        sentences, unique_hashes = process_tmx_file(
            file_path, args.output, source_name, args.source_lang, args.target_lang
        )
        total_sentences += sentences
        all_unique_hashes.update(unique_hashes)

    print("\nProcessing complete:")
    print(f"Total sentences processed: {total_sentences:,}")
    print(f"Unique sentence pairs: {len(all_unique_hashes):,}")


if __name__ == "__main__":
    main()
