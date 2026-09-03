import fire
import sys
import traceback
from pathlib import Path
try:
    from src.cli import CLI
except Exception as e:
    last_call = traceback.extract_tb(e.__traceback__)[-1]
    file_name = Path(last_call.filename).name
    print(
        f"Erreur: ({e.__class__.__name__})"
        f" in {file_name}:{last_call.lineno} [{last_call.name}()]")
    print(f"Detail : {e}")
    sys.exit(1)


def main() -> None:
    """Launch the Fire CLI interface and
      handle execution errors with traceback location details."""
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        last_call = traceback.extract_tb(e.__traceback__)[-1]
        file_name = Path(last_call.filename).name

        print(
            f"Erreur: ({e.__class__.__name__})"
            f" in {file_name}:{last_call.lineno} [{last_call.name}()]")
        print(f"Detail : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
