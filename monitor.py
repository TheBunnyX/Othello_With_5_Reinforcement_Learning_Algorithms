"""Training monitor GUI: opens the dashboard in your browser.

    python monitor.py                 # http://127.0.0.1:8002/monitor
    python monitor.py --port 8002

Start / stop training from the page, or run training.py in another terminal:
the dashboard reads logs/<algo>_status.json and logs/<algo>.jsonl either way.
"""
from app import make_parser, monitor_enabled, serve

if __name__ == "__main__":
    p = make_parser()
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = p.parse_args()
    if not monitor_enabled():
        raise SystemExit("Training Monitor is disabled: set ENABLE_TRAINING_MONITOR=true in .env")
    serve(args, open_path=None if args.no_browser else "/monitor")
