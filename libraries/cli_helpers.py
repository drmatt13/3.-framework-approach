import argparse


def lower_token(value: str) -> str:
    return str(value).strip().lower()


def parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")
