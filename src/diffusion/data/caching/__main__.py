"""Make the caching module directly runnable.

This allows running the cache builder with:
    python -m src.diffusion.data.caching ...
"""

from .cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
