"""gunicorn settings for the container: everything is logged to stdout/stderr so it
shows up in the Dokploy "Logs" tab (and `docker logs`)."""
import glob
import os

bind = "0.0.0.0:8002"
preload_app = True
threads = 4
timeout = 120

accesslog = "-"        # one line per request: client, path, status, time
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
access_log_format = '%(h)s "%(r)s" %(s)s %(b)sB %(L)ss'


def when_ready(server):
    """Print a startup summary once the app is loaded."""
    weights = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("weights/*.pt"))
    server.log.info("Othello RL starting | workers=%s threads=%s", server.cfg.workers, threads)
    server.log.info("Models found (%d): %s", len(weights), ", ".join(weights) or "NONE - check weights/")
    server.log.info("MINIMAX_DEPTH=%s ALPHABETA_DEPTH=%s ENABLE_TRAINING_MONITOR=%s",
                    os.environ.get("MINIMAX_DEPTH", "3"), os.environ.get("ALPHABETA_DEPTH", "5"),
                    os.environ.get("ENABLE_TRAINING_MONITOR", "true"))


def post_worker_init(worker):
    worker.log.info("Worker %s ready", worker.pid)
