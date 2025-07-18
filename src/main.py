"""Entry point for the shopping assistant."""

from .dialogue_system import start_chat


def main() -> None:
    """Launch the interactive shopping assistant."""
    start_chat()


if __name__ == "__main__":
    main()
