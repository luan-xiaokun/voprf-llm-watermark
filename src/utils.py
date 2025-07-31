import json
from pathlib import Path
from collections.abc import Generator


def read_json(file_name: str) -> dict:
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_name: str, data: dict, indent: int | None = None) -> None:
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent)


def read_jsonlines(file_name: str) -> Generator[dict, None, None]:
    """Read a JSON Lines file and yield each JSON object."""
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def write_jsonlines(
    file_name: str, data: dict | list[dict], mode: str = "w", indent: int | None = None
) -> None:
    """Append JSON objects to a JSON Lines file."""
    if isinstance(data, dict):
        data = [data]
    with open(file_name, mode, encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, indent=indent) + "\n")
