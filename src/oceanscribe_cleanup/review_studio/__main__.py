"""Run the explicitly authorized local review studio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import ReviewStore


def main():
    parser = argparse.ArgumentParser(description="Lokales OceanScribe Review-Studio")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-root", type=Path, default=Path("data/review-studio/current"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Nur für isolierte technische Fixture-Tests; keine echten Daten importieren",
    )
    args = parser.parse_args()
    store = ReviewStore(args.data_root)
    if not args.no_bootstrap:
        from .bootstrap import initialize_studio

        initialize_studio(store, Path.cwd())
    if args.prepare_only:
        print(json.dumps(store.dashboard(), ensure_ascii=False, indent=2))
        return
    if args.check:
        from .asr import LocalASRRunner

        print(
            json.dumps(
                {
                    "workspace": str(store.root),
                    "asr": LocalASRRunner.discover().status(),
                    "dashboard": store.dashboard(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    from .server import serve

    server = serve(store, host="127.0.0.1", port=args.port)
    print(f"Review-Studio: http://127.0.0.1:{server.server_address[1]}", flush=True)
    print(f"Lokale Daten: {store.root}; Stoppen mit Ctrl+C", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
