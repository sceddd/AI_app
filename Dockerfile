FROM ubuntu:latest
LABEL authors="victor-ho"

ENTRYPOINT ["top", "-b"]

FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Ensure the log directory exists
RUN mkdir -p ./logs

# Make port 8000 (Django), 6379 (Redis), 8080 (TorchServe) available to the world outside this container
EXPOSE 8000 6379 8080

ENV DJANGO_SETTINGS_MODULE=AI_Backend.settings
ENV PYTHONUNBUFFERED 1