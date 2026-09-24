"""Jeviathan entrypoint.

Run:  uvicorn jeviathan.main:app --host 0.0.0.0 --port 8100
Env:  JEVIATHAN_PROFILE (laptop-4050 | rtx5090), JEVIATHAN_BASE_URL, JEVIATHAN_MODEL
"""

from __future__ import annotations

import os

from .api.systemone import create_app

app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("JEVIATHAN_HOST", "0.0.0.0")
    port = int(os.environ.get("JEVIATHAN_PORT", "8100"))
    uvicorn.run("jeviathan.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
