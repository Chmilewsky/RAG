import fire
import sys
try:
    from src.index import CLI
except Exception:
    print("import error")
    sys.exit(1)


def main():
    try:
        fire.Fire(CLI)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
