"""
Convert XDF files to JSON format.

This script reads all .xdf files in the data directory and converts them to .json files.
The JSON files will be saved in the same directory as the source XDF files.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyxdf


def numpy_to_python(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    else:
        return obj


def convert_xdf_to_json(xdf_file_path, output_path=None):
    """
    Convert a single XDF file to JSON format.

    Args:
        xdf_file_path: Path to the XDF file
        output_path: Optional output path for JSON file. If None, uses same directory as XDF file.

    Returns:
        Path to the created JSON file
    """
    xdf_path = Path(xdf_file_path)

    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")

    # Determine output path
    if output_path is None:
        json_path = xdf_path.with_suffix(".json")
    else:
        json_path = Path(output_path)

    print(f"Converting: {xdf_path}")
    print(f"        to: {json_path}")

    try:
        # Load XDF file
        streams, file_header = pyxdf.load_xdf(str(xdf_path))

        # Prepare data structure for JSON
        json_data = {"file_header": numpy_to_python(file_header), "streams": []}

        # Process each stream
        for stream in streams:
            # Get stream name to filter
            stream_name = ""
            if "info" in stream:
                info = stream["info"]
                stream_name = (
                    info.get("name", [""])[0]
                    if isinstance(info.get("name"), list)
                    else info.get("name", "")
                )

            # Only include LatencyMarkers and ExpMarkers streams
            if stream_name not in ["LatencyMarkers", "ExpMarkers"]:
                continue

            stream_data = {"info": {}, "time_series": [], "time_stamps": []}

            # Extract stream info
            if "info" in stream:
                info = stream["info"]
                # Convert info to a serializable format
                stream_data["info"] = {
                    "name": info.get("name", [""])[0]
                    if isinstance(info.get("name"), list)
                    else info.get("name", ""),
                    "type": info.get("type", [""])[0]
                    if isinstance(info.get("type"), list)
                    else info.get("type", ""),
                    "channel_count": info.get("channel_count", [""])[0]
                    if isinstance(info.get("channel_count"), list)
                    else info.get("channel_count", ""),
                    "nominal_srate": info.get("nominal_srate", [""])[0]
                    if isinstance(info.get("nominal_srate"), list)
                    else info.get("nominal_srate", ""),
                    "channel_format": info.get("channel_format", [""])[0]
                    if isinstance(info.get("channel_format"), list)
                    else info.get("channel_format", ""),
                    "stream_id": info.get("stream_id", [""])[0]
                    if isinstance(info.get("stream_id"), list)
                    else info.get("stream_id", ""),
                    "source_id": info.get("source_id", [""])[0]
                    if isinstance(info.get("source_id"), list)
                    else info.get("source_id", ""),
                    "created_at": info.get("created_at", [""])[0]
                    if isinstance(info.get("created_at"), list)
                    else info.get("created_at", ""),
                    "uid": info.get("uid", [""])[0]
                    if isinstance(info.get("uid"), list)
                    else info.get("uid", ""),
                    "session_id": info.get("session_id", [""])[0]
                    if isinstance(info.get("session_id"), list)
                    else info.get("session_id", ""),
                    "hostname": info.get("hostname", [""])[0]
                    if isinstance(info.get("hostname"), list)
                    else info.get("hostname", ""),
                    "desc": info.get("desc", {}),
                }

            # Extract time series data
            if "time_series" in stream:
                stream_data["time_series"] = numpy_to_python(stream["time_series"])

            # Extract timestamps
            if "time_stamps" in stream:
                stream_data["time_stamps"] = numpy_to_python(stream["time_stamps"])

            # Add footer if present
            if "footer" in stream:
                stream_data["footer"] = numpy_to_python(stream["footer"])

            json_data["streams"].append(stream_data)

        # Write JSON file
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        print(f"  Success: Created {json_path}")
        return json_path

    except Exception as e:
        print(f"  Error converting {xdf_path}: {e}")
        raise


def convert_all_xdf_in_directory(directory_path, recursive=True):
    """
    Convert all XDF files in a directory to JSON format.

    Args:
        directory_path: Path to the directory containing XDF files
        recursive: Whether to search recursively in subdirectories

    Returns:
        List of paths to created JSON files
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Find all XDF files
    if recursive:
        xdf_files = list(directory.rglob("*.xdf"))
    else:
        xdf_files = list(directory.glob("*.xdf"))

    if not xdf_files:
        print(f"No XDF files found in {directory}")
        return []

    print(f"Found {len(xdf_files)} XDF file(s) to convert")
    print("-" * 60)

    json_files = []
    success_count = 0
    error_count = 0

    for xdf_file in xdf_files:
        try:
            json_path = convert_xdf_to_json(xdf_file)
            json_files.append(json_path)
            success_count += 1
        except Exception as e:
            print(f"  Failed to convert {xdf_file}: {e}")
            error_count += 1
        print("-" * 60)

    # Print summary
    print("\nConversion complete:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Total: {len(xdf_files)}")

    return json_files


data_dir = Path("data")

if not data_dir.exists():
    print(f"Error: Data directory does not exist: {data_dir}")
    sys.exit(1)

try:
    convert_all_xdf_in_directory(data_dir, recursive=True)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
