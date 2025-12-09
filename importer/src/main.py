import contextlib
import signal
import sys
from pathlib import Path

from loguru import logger

from .config import Config
from .processing import DocumentProcessor
from .services import SchedulerCoordinator


def setup_logging():
    """Configure application logging."""
    with contextlib.suppress(Exception):
        Path(Config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    _ = logger.add(
        Config.LOG_FILE,
        rotation="10 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


def setup_signal_handlers(scheduler_coordinator: SchedulerCoordinator):
    """Setup signal handlers for graceful shutdown."""

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}; requesting shutdown...")
        scheduler_coordinator.request_stop()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, handle_signal)


def main():
    """Main entry point - runs initial processing then starts scheduler."""
    setup_logging()

    # Initialize components
    document_processor = DocumentProcessor()
    scheduler_coordinator = SchedulerCoordinator(document_processor)

    # Setup signal handling
    setup_signal_handlers(scheduler_coordinator)

    try:
        # Run initial processing then start scheduler
        scheduler_coordinator.run_initial_process()
        scheduler_coordinator.start_scheduler()
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Importer stopped.")


if __name__ == "__main__":
    main()
