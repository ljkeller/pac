import logging

from runner import run_jobs


def main():
    # TODO: set root level file logger for entire run?
    logging.basicConfig(level=logging.INFO)

    run_jobs()


if __name__ == "__main__":
    main()
