import threading
import time

import schedule
from loguru import logger

from ..config import Config


class SchedulerCoordinator:
    """Coordinates scheduled execution and handles concurrency."""

    def __init__(self, processor):
        self.document_processor = processor
        self.run_lock = threading.Lock()
        self.stop_event = threading.Event()

    def run_scheduled_job(self):
        """Execute the main processing job if not already running."""
        if self.stop_event.is_set():
            return

        acquired = self.run_lock.acquire(blocking=False)
        if not acquired:
            logger.warning(
                "Previous run still in progress; skipping this schedule tick."
            )
            return

        try:
            if not self.stop_event.is_set():
                self.document_processor.run_main_process()
        finally:
            self.run_lock.release()

    def run_initial_process(self):
        """Run initial processing if possible."""
        if self.stop_event.is_set():
            logger.info("Shutdown requested before run; skipping main().")
            return

        if self.run_lock.acquire(blocking=False):
            try:
                self.document_processor.run_main_process()
            finally:
                self.run_lock.release()
        else:
            logger.warning("Importer already running at startup; initial run skipped.")

    def start_scheduler(self):
        """Start the scheduled execution loop."""
        schedule.every(Config.SCHEDULE_TIME).minutes.do(self.run_scheduled_job)

        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    def request_stop(self):
        """Request a graceful shutdown."""
        self.stop_event.set()
