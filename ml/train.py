import logging

from runner import run_jobs


def main():
    logging.basicConfig(level=logging.INFO)

    run_jobs()


if __name__ == "__main__":
    main()
