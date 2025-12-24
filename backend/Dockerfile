FROM mambaorg/micromamba:latest
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

# install conda env
COPY backend/backend_env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean -a -y

# copy rest of code
COPY backend /app

ENV PROPERTY_PREDICTION_DIR=/app/property_prediction
ENV MODELS_DIR=/app/property_prediction/models
ENV SHARED_DIR=/app/shared
EXPOSE 8000

CMD ["micromamba", "run", "-n", "base", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]