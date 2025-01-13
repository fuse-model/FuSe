#!/bin/bash
uv sync --extra gpu
uv pip install -e ../octo_digit --no-deps
uv pip install -e ../bridge_with_digit/widowx_envs