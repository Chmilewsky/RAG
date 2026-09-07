import fire
import sys
from src.cli import CLI


def main() -> None:
    """Launch the Fire CLI interface"""
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
