# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Metric result aggregation package.

- Contains plugins that summarize ItemResult records.
- Keeps aggregation policies independent from metric implementations.
- Provides the mean aggregator as the default built-in implementation.
"""
