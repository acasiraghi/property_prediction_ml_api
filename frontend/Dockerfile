FROM mambaorg/micromamba:latest
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

# install conda env
COPY frontend/frontend_env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean -a -y

# copy rest of code
COPY frontend /app
ENV SHARED_DIR=/app/shared
EXPOSE 3000
CMD ["micromamba", "run", "-n", "base", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--port", "3000"]