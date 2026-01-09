#!/bin/bash
# wait-for-it.sh: Wait for a service to be available

set -e

HOST="${1:-$POSTGRES_HOST}"
PORT="${2:-$POSTGRES_PORT}"
TIMEOUT="${3:-30}"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Usage: wait-for-it.sh <host> <port> [timeout]"
    echo "Or set POSTGRES_HOST and POSTGRES_PORT environment variables"
    exit 1
fi

echo "Waiting for $HOST:$PORT to be available..."

start_time=$(date +%s)
while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "Timeout waiting for $HOST:$PORT after ${TIMEOUT}s"
        exit 1
    fi

    echo "Waiting... (${elapsed}s elapsed)"
    sleep 1
done

echo "$HOST:$PORT is available!"
exec "${@:4}"
