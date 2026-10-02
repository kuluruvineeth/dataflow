import logging
import sys

import cloudpickle
import fsspec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with fsspec.open(sys.argv[1], "rb") as file:
        executor = cloudpickle.load(file)
    executor.run()


if __name__ == "__main__":
    main()
