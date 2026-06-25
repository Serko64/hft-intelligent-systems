import os
import sys
import webview

from f1_rl.desktop.api import Api

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DIST_INDEX = os.path.join(ROOT, "web", "dist", "index.html")


def main():
    if os.path.exists(DIST_INDEX):
        url = DIST_INDEX
    else:
        sys.exit(
            f"Gebautes UI nicht gefunden unter {DIST_INDEX}.\n"
            f"Erst `npm run build` in web/ ausführen."
        )

    webview.create_window(
        "F1 RL Simulator",
        url=url,
        js_api=Api(),
        width=1480,
        height=900,
        min_size=(1024, 700),
    )

    webview.start()


if __name__ == "__main__":
    main()