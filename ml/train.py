"""Train all jobs in the jobs directory"""

import logging
import os
import time
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
from tqdm import tqdm  # noqa: E402


def process_training_batch(jobs_path=Path("./jobs")):
    """Run all jobs in the jobs directory"""
    start_time = time.time()
    root_logger = logging.getLogger()

    jobs = get_job_files(jobs_path)
    jobs_len = len(jobs)
    root_logger.info(f"Found {jobs_len} jobs to process.")
    for idx, job_path in enumerate(tqdm(jobs, desc="Jobs processed", colour="cyan")):
        root_logger.info(f"Processing job {job_path.name}")
        file_handler = None
        try:
            root_logger.info(f"Starting job {job_path.name}.")
            with TrainingJob(job_path) as training_job:
                # Each training job is allocated its own log file, to be stored
                # in the job's archive
                file_handler = logging.FileHandler(
                    f"{training_job.results_dir}/{job_path.stem}.log"
                )
                file_handler.setLevel(numeric_log_level)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                root_logger.addHandler(file_handler)

                training_job.process()
            root_logger.info(f"Finished job {job_path.name}.")
        except Exception as e:
            root_logger.error(f"Error procesging job {job_path.name}.")
            root_logger.error(e)
        finally:
            if file_handler:
                root_logger.removeHandler(file_handler)
                file_handler.close()
        root_logger.info(
            f"Processed job {job_path.name}. {idx + 1}/{jobs_len} jobs processed."
        )
        root_logger.info(f"Elapsed time: {time.time() - start_time}")


def main():
    process_training_batch()
    logging.getLogger().info("All jobs processed.")


if __name__ == "__main__":
    main()
