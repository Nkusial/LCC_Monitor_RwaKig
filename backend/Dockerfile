FROM mambaorg/micromamba:1.5.10

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.backend.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER backend ./backend
COPY --chown=$MAMBA_USER:$MAMBA_USER configs ./configs

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["micromamba", "run", "-n", "base", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
