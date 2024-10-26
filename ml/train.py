"""Train all jobs in the jobs directory"""

import logging
import os
from pathlib import Path

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_log_level = getattr(logging, log_level, logging.WARNING)

# Establish root logger rules before importing other modules that leverage logging
logging.basicConfig(
    level=numeric_log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from runner import TrainingJob, get_job_files  # noqa: E402


def run_jobs(jobs_path=Path("./jobs")):
    """Run all jobs in the jobs directory"""

    for job_path in get_job_files(jobs_path):
        # TODO: Move log into tmpdir?

        # Start job-specific logging
        file_handler = logging.FileHandler(f"./{job_path.name}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        try:
            root_logger.info(f"Starting job {job_path.name}.")
            with TrainingJob(job_path) as job:
                job.train()
            root_logger.info(f"Finished job {job_path.name}.")
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()


def main():
    run_jobs()


if __name__ == "__main__":
    main()
