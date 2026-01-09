FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml README.md ./
COPY src ./src

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy the rest of the application
COPY . .

# Copy and set up wait-for-it script
COPY scripts/wait-for-it.sh /usr/local/bin/wait-for-it.sh
RUN chmod +x /usr/local/bin/wait-for-it.sh

# Default command
CMD ["uv", "run", "python", "-c", "print('Generator ready')"]
