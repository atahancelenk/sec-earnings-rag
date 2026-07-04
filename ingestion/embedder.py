# ingestion/embedder.py

import os
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from chunker import Chunk

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, local, no API key
EMBEDDING_DIM   = 384                   # ← Pinecone index must match this
PINECONE_BATCH  = 100


class Embedder:
    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model   = SentenceTransformer(EMBEDDING_MODEL)
        self.pc      = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index   = self.pc.Index(os.getenv("PINECONE_INDEX"))

    def embed_and_upsert(self, chunks: list[Chunk]) -> int:
        logger.info(f"Embedding {len(chunks)} chunks locally...")
        total_upserted = 0

        # SentenceTransformer handles batching internally
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
        ).tolist()
        

        # Build and upsert vectors in Pinecone batches
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id":       chunk.chunk_id,
                "values":   embedding,
                "metadata": self._build_metadata(chunk),
            })

        for i in range(0, len(vectors), PINECONE_BATCH):
            batch = vectors[i : i + PINECONE_BATCH]
            self.index.upsert(vectors=batch)
            total_upserted += len(batch)
            logger.info(f"  Upserted {min(i + PINECONE_BATCH, len(vectors))}/{len(vectors)}")

        logger.info(f"Done — {total_upserted} vectors in Pinecone")
        return total_upserted

    @staticmethod
    def _build_metadata(chunk: Chunk) -> dict:
        meta = {
            "ticker":      chunk.ticker,
            "company_name": chunk.company_name,
            "form_type":   chunk.form_type,
            "fiscal_year": chunk.fiscal_year,
            "section":     chunk.section,
            "chunk_type":  chunk.chunk_type,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "char_count":  chunk.char_count,
            "text":        chunk.text[:1000],
        }
        if chunk.quarter is not None:
            meta["quarter"] = chunk.quarter
        if chunk.table_index is not None:
            meta["table_index"] = chunk.table_index
        return meta