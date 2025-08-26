import secrets
from pathlib import Path


def get_server_seed(seed_path: str | Path | None = None) -> bytes:
    if seed_path:
        seed_path = Path(seed_path).expanduser().resolve()
        if seed_path.exists():
            with open(seed_path) as f:
                return bytes.fromhex(f.read().strip())
        print(f"Warning: Seed file {seed_path} not found.")
    return secrets.token_bytes(64)
