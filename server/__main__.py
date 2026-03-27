"""Entry point: python -m server [--host] [--port] [--no-open]"""

import argparse
from server.app import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-Omni unified server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port (default: 8000)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, auto_open=not args.no_open)


if __name__ == "__main__":
    main()
