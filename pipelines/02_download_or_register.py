"""Placeholder stage for future raw-data download or asset registration."""

from common import ensure_output_dirs, load_config


def main() -> None:
    """Confirm the raw-data path exists until download logic is added."""
    config = load_config()
    ensure_output_dirs(config)
    print(f"Download/register stage ready. Raw data path: {config['paths']['raw']}")


if __name__ == "__main__":
    main()
