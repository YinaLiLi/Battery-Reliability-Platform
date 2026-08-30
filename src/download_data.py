from pathlib import Path
import requests


URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DATA_DIR / "nasa_battery_data.zip"


def download_file(url: str, output_path: Path):
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Downloaded to: {output_path}")


if __name__ == "__main__":
    download_file(URL, OUTPUT_FILE)