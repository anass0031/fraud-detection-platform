FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Shared ML package (feature pipeline, model registry, explanations) from the "ml" build context.
# Dependencies first (better layer caching), then the package code.
COPY --from=ml pyproject.toml /ml/pyproject.toml
RUN mkdir /ml/fraudml && touch /ml/fraudml/__init__.py \
    && pip install -e "/ml[serve]"
COPY --from=ml fraudml /ml/fraudml

WORKDIR /app

COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install -e ".[dev]"

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
