FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.8.1 psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
