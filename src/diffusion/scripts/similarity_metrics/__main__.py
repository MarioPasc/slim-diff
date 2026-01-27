"""Entry point for running similarity metrics as a module.

Usage:
    python -m src.diffusion.scripts.similarity_metrics --config config.yaml
    python -m src.diffusion.scripts.similarity_metrics compute-all --config config.yaml
"""

from .cli import main

if __name__ == "__main__":
    main()
