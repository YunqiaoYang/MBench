# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Source locator abstractions for adapter I/O.

- Treats local directories and HuggingFace dataset URLs uniformly.
- Adapter only reads small text files (yaml / jsonl / json) through this layer.
- Large binaries (videos, npz) stay un-downloaded; metric plugins call
  ``materialize`` on demand.
- Lazy-imports ``huggingface_hub`` so the local code path stays dependency-free.
"""

from .sources import (
    HF_PREFIX,
    HfSourceLocator,
    LocalSourceLocator,
    SourceLocator,
    locator_for,
    materialize,
)

__all__ = [
    "HF_PREFIX",
    "HfSourceLocator",
    "LocalSourceLocator",
    "SourceLocator",
    "locator_for",
    "materialize",
]
