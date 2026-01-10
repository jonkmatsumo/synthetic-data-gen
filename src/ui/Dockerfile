# Lightweight Python base for Streamlit dashboard
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit>=1.37.0 \
    pandas>=2.2.0 \
    sqlalchemy>=2.0.0 \
    psycopg2-binary>=2.9.0 \
    requests>=2.31.0 \
    plotly>=5.22.0 \
    mlflow>=3.8.0 \
    boto3>=1.34.0

# Copy only the UI application
COPY src/ui/ /app/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
