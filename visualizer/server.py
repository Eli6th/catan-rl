from pathlib import Path
from catan_service import create_app

DIST_DIR = Path(__file__).parent / "dist"
LOG_DIR = Path(__file__).parent.parent / "logs"

app = create_app(frontend_dir=DIST_DIR, log_dir=LOG_DIR)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
