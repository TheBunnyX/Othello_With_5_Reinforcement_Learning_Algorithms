FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Tuned for a 4-core CPU (i5-6500, no hyper-threading) with 4 GB RAM shared with Dokploy:
# one torch thread per worker avoids oversubscribing the cores; WEB_CONCURRENCY sets the workers.
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WEB_CONCURRENCY=3
WORKDIR /app

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY agents.py app.py env.py search.py vec_env.py ./
COPY web ./web
COPY weights ./weights

# The public site only plays games: no training monitor, no training endpoints.
ENV ENABLE_TRAINING_MONITOR=false MINIMAX_DEPTH=3 ALPHABETA_DEPTH=5
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/config')"

# --preload loads torch once and shares it between workers (copy-on-write), saving RAM.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--preload", "--threads", "4", "--timeout", "120"]
