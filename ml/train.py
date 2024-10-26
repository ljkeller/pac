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


def process_training_batch(jobs_path=Path("./jobs")):
    """Run all jobs in the jobs directory"""

    root_logger = logging.getLogger()
    for job_path in get_job_files(jobs_path):
        file_handler = None
        try:
            root_logger.info(f"Starting job {job_path.name}.")
            with TrainingJob(job_path) as training_job:
                # Each training job is allocated its own log file, to be stored
                # in the job's archive
                print(f"{training_job.temp_dir_name}/{job_path.name}.log")
                file_handler = logging.FileHandler(
                    f"{training_job.temp_dir_name}/{job_path.name}.log"
                )
                root_logger.addHandler(file_handler)

                training_job.process()
            root_logger.info(f"Finished job {job_path.name}.")
        finally:
            if file_handler:
                root_logger.removeHandler(file_handler)
                file_handler.close()


def main():
    process_training_batch()


if __name__ == "__main__":
    main()
