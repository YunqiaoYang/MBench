# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Dataset adapter package.

- Converts external data sources into list[EvaluationItem].
- Keeps source-specific parsing outside the pipeline.
- Provides the manifest adapter as the first built-in implementation.
"""
