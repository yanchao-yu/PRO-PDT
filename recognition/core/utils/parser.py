import argparse

from recognition.core.utils.model_loader import (
    ModelLoader,
    ModelRegistry,
)


def parse_value(value: str):
    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    if value.lower() == "none":
        return None

    try:
        return int(value)

    except ValueError:
        pass

    try:
        return float(value)

    except ValueError:
        return value


def parse_overrides(values):
    overrides = {}

    for item in values:

        if "=" not in item:
            raise ValueError(
                f"Invalid override: {item}. "
                f"Expected key=value"
            )

        key, value = item.split(
            "=",
            1
        )

        overrides[key] = parse_value(
            value
        )

    return overrides


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            model.value
            for model in ModelRegistry
        ],
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    overrides = {}

    if args.depth is not None:
        overrides["depth"] = args.depth

    if args.heads is not None:
        overrides["heads"] = args.heads

    if args.dropout is not None:
        overrides["dropout"] = args.dropout

    model_name = ModelRegistry(
        args.model
    )

    model = ModelLoader.load(
        model_name=model_name,
        overrides=overrides,
    )
    print(
        f"Loaded model: {model_name.value}"
    )

    print(
        f"Overrides: {overrides}"
    )
