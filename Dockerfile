# storecam on CPU. Docker on macOS cannot reach the Apple GPU, so this image is the portable,
# slower path; on a Mac, `uv run storecam run` uses the GPU directly (README).
FROM python:3.12-slim

# OpenCV's wheel needs these shared libraries at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# Everything from the lock file except PyTorch and its CUDA libraries (several GB this CPU image
# would never use), then the CPU build of the same locked PyTorch versions; `uv pip check`
# fails the build if anything the lock needs is missing.
RUN uv export --frozen --no-dev --no-hashes --no-emit-project \
      | grep -v -E '^(torch|torchvision|triton|cuda-|nvidia-(cu|nccl|nvjitlink|nvshmem|nvtx))' > /tmp/req.txt \
    && uv venv .venv \
    && uv pip install --python .venv/bin/python --no-deps -r /tmp/req.txt \
    && uv pip install --python .venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/cpu \
       torch==2.15.0 torchvision==0.29.2 \
    && uv pip install --python .venv/bin/python --no-deps . \
    && uv pip check --python .venv/bin/python
ENV PATH="/app/.venv/bin:$PATH"
COPY config.yaml ./

ENTRYPOINT ["storecam"]
CMD ["run", "config.yaml"]
