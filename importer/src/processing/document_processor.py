import hashlib
from typing import Optional

from loguru import logger

from ..config import Config
from ..models import DocumentWork
from ..utils import TextUtils
from ..connectors import Neo4jMemoryConnector, PaperlessConnector
from .agent_orchestrator import AgentOrchestrator
from .state_manager import StateManager
from ..services import ServiceBootstrapper


class DocumentProcessor:
    """Main ETL pipeline for processing documents."""

    def __init__(self):
        self.neo4j_connector = Neo4jMemoryConnector()
        self.agent_orchestrator = AgentOrchestrator(self.neo4j_connector)
        self.paperless_connector = PaperlessConnector()
        self.state_manager = StateManager()
        self.bootstrapper = ServiceBootstrapper()

    def prepare_document_work(self, document: dict) -> Optional[DocumentWork]:
        """Prepare a document work unit with validation and chunking."""
        try:
            doc_id = int(document.get("id", 0))
        except Exception:
            return None

        logger.info(f"[doc] consider id={doc_id} force={Config.FORCE_REPROCESS}")

        if not self.state_manager.should_process_document(doc_id):
            logger.info(
                f"[doc] skip id={doc_id} last_id={self.state_manager.state.get('last_id', 0)}"
            )
            return None

        try:
            detailed = self.paperless_connector.get_document(doc_id)
        except Exception as exc:
            logger.error(f"[doc] detail fetch failed id={doc_id}: {exc}")
            return None

        text = self.paperless_connector.extract_text(detailed)

        if not text:
            # Try to fall back from metadata
            fallback = " ".join(
                [
                    str(detailed.get("title") or ""),
                    str(detailed.get("notes") or ""),
                    str(detailed.get("original_filename") or ""),
                    str(detailed.get("created") or ""),
                ]
            ).strip()

            if fallback:
                logger.info(
                    f"[doc] empty OCR content; using metadata fallback id={doc_id}"
                )
            text = fallback

        if not text:
            logger.info(f"[doc] no usable text id={doc_id}; advancing state")
            self.state_manager.advance_last_id(doc_id)
            return None

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if not self.state_manager.is_document_changed(doc_id, text_hash):
            logger.info(f"[doc] unchanged hash; skip id={doc_id}")
            self.state_manager.advance_last_id(doc_id)
            return None

        chunks = TextUtils.chunk_text(text)
        logger.info(
            f"[doc] prepared id={doc_id} chunks={len(chunks)} first_len={len(chunks[0]) if chunks else 0}"
        )

        source_url = str(detailed.get("download_url") or "")
        return DocumentWork(
            doc_id=doc_id,
            source_url=source_url,
            chunks=chunks,
            text_hash=text_hash,
            doc=detailed,
        )

    def run_downstream_steps(self, work: DocumentWork):
        """Execute downstream processing steps for document work."""
        total_chunks = len(work.chunks)
        successful_chunks = 0
        failed_chunks = 0

        logger.info(f"[doc] processing {total_chunks} chunks for doc_id={work.doc_id}")

        for chunk_index, chunk_text in enumerate(work.chunks):
            source_id = str(work.doc_id)
            chunk_id = f"c{chunk_index + 1}"

            logger.info(
                f"[doc] processing chunk {chunk_index + 1}/{total_chunks} for doc_id={work.doc_id}"
            )

            try:
                self.agent_orchestrator.process_chunk(
                    source_id, chunk_id, work.source_url, chunk_text
                )
                successful_chunks += 1
                logger.info(
                    f"[doc] chunk {chunk_index + 1}/{total_chunks} completed successfully for doc_id={work.doc_id}"
                )
            except Exception as exc:
                failed_chunks += 1
                logger.error(
                    f"[doc] chunk {chunk_index + 1}/{total_chunks} failed for doc_id={work.doc_id}: {exc}",
                    exc_info=True,
                )
                # Continue processing other chunks even if one fails

            # Optional Obsidian export
            if Config.OBSIDIAN_EXPORT:
                try:
                    self.paperless_connector.write_obsidian_note(
                        work.doc, chunk_index, chunk_text
                    )
                    logger.debug(
                        f"[doc] obsidian export completed for chunk {chunk_index + 1} of doc_id={work.doc_id}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[doc] obsidian export failed for chunk {chunk_index + 1} of doc_id={work.doc_id}: {exc}"
                    )

        # Summary logging
        logger.info(
            f"[doc] chunk processing completed for doc_id={work.doc_id}: {successful_chunks}/{total_chunks} successful, {failed_chunks} failed"
        )

        if failed_chunks > 0:
            logger.warning(
                f"[doc] {failed_chunks} chunks failed processing for doc_id={work.doc_id} - some observations may be missing"
            )

    def finalize_document(self, work: DocumentWork):
        """Finalize document processing and update state."""
        self.state_manager.update_document_state(work.doc_id, work.text_hash)

    def process_document(self, document: dict):
        """Process a single document through the complete pipeline."""
        work = self.prepare_document_work(document)
        if not work:
            return

        self.run_downstream_steps(work)
        self.finalize_document(work)

    def run_main_process(self):
        """Run the main document processing loop."""
        try:
            self.bootstrapper.bootstrap_all_services()
            processed = 0

            for document in self.paperless_connector.iter_documents():
                # Check for shutdown signal from global state would go here if needed
                try:
                    self.process_document(document)
                    processed += 1
                except Exception as exc:
                    logger.error(f"Failed to process document: {exc}")

            logger.info(f"[run] completed; processed_docs={processed}")

        except Exception as exc:
            logger.critical(f"Fatal error in main process: {exc}")
            raise
