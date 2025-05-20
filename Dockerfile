FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN pip install --upgrade pip \
 && pip install poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
 && poetry install --no-root

COPY . /app

# 👇 Solución al problema de imports
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["/bin/bash"]
