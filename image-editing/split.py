import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any


def split_annotations(input_path: str, output_dir: str, num_parts: int) -> None:
    """
    Split annotations JSON file into K equal parts.

    Args:
        input_path: Path to the input annotations JSON file
        output_dir: Directory to save the split JSON files
        num_parts: Number of parts to split the annotations into
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(input_path, 'r') as f:
        annotations: List[Dict[str, Any]] = json.load(f)

    # Calculate split sizes
    total_annotations = len(annotations)
    base_size = math.floor(total_annotations / num_parts)
    remainder = total_annotations % num_parts

    # Split and save annotations
    start_idx = 0
    for i in range(num_parts):
        # Calculate size for this part (distribute remainder across first few parts)
        part_size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + part_size

        # Extract annotations for this part
        part_annotations = annotations[start_idx:end_idx]

        # Create output filename
        output_path = output_dir / f"annotations_part_{i + 1:02d}_of_{num_parts:02d}.json"

        # Save to file with pretty printing
        with open(output_path, 'w') as f:
            json.dump(part_annotations, f, indent=4)

        print(f"Saved part {i + 1} with {len(part_annotations)} annotations to {output_path}")

        # Update start index for next part
        start_idx = end_idx


def main() -> None:
    parser = argparse.ArgumentParser(description='Split annotations JSON file into K parts')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input annotations JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save split JSON files')
    parser.add_argument('--num_parts', type=int, required=True,
                        help='Number of parts to split the annotations into')

    args = parser.parse_args()

    split_annotations(
        input_path=args.input,
        output_dir=args.output_dir,
        num_parts=args.num_parts
    )


if __name__ == "__main__":
    main()