# ingestion/chunker.py

import re
import logging
import tiktoken
from dataclasses import dataclass, field
from typing import Optional
from pdf_parser import ParsedFiling, ParsedSection

logger = logging.getLogger(__name__)

# Tokenizer — must match your embedding model
# text-embedding-3-small uses the same tokenizer as GPT-4
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Chunking hyperparameters
PROSE_CHUNK_TOKENS   = 512
PROSE_OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS     = 40    # discard chunks smaller than this
TABLE_MAX_TOKENS     = 1200  # tables beyond this get split by row groups


@dataclass
class Chunk:
    """
    A single chunk ready for embedding and Pinecone upsert.
    Every field here becomes either the vector payload or Pinecone metadata.
    """
    # Identity
    chunk_id:    str         # globally unique: ticker_formtype_year_section_idx
    text:        str         # the actual text that gets embedded

    # Document provenance — used for Pinecone metadata filtering
    ticker:      str
    company_name: str        # filled in later from yfinance metadata
    form_type:   str         # "10-K" or "10-Q"
    fiscal_year: int
    quarter:     Optional[int]
    section:     str         # "mda", "risk_factors", "financial_statements", etc.
    chunk_type:  str         # "prose" or "table"
    chunk_index: int         # position within the section

    # Quality signals
    token_count: int
    char_count:  int

    # Table-specific (None for prose chunks)
    table_index: Optional[int] = field(default=None)


class Chunker:
    """
    Splits ParsedFiling objects into Chunks ready for embedding.

    Two paths:
      - Prose sections  → overlapping sliding window (512t / 50t overlap)
      - Table content   → one chunk per table, split by row groups if too large
    """

    def __init__(
        self,
        prose_chunk_tokens:   int = PROSE_CHUNK_TOKENS,
        prose_overlap_tokens: int = PROSE_OVERLAP_TOKENS,
        min_chunk_tokens:     int = MIN_CHUNK_TOKENS,
        table_max_tokens:     int = TABLE_MAX_TOKENS,
    ):
        self.prose_chunk_tokens   = prose_chunk_tokens
        self.prose_overlap_tokens = prose_overlap_tokens
        self.min_chunk_tokens     = min_chunk_tokens
        self.table_max_tokens     = table_max_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_filing(self, filing: ParsedFiling) -> list[Chunk]:
        """
        Chunk all sections of a parsed filing.
        Returns a flat list of Chunks across all sections.
        """
        all_chunks = []

        for section in filing.sections:
            # Path 1: prose text
            prose_chunks = self._chunk_prose(section)
            all_chunks.extend(prose_chunks)

            # Path 2: tables embedded in this section
            table_chunks = self._chunk_tables(section)
            all_chunks.extend(table_chunks)

        logger.info(
            f"{filing.ticker} {filing.form_type} {filing.fiscal_year}: "
            f"{len(all_chunks)} chunks "
            f"({sum(1 for c in all_chunks if c.chunk_type == 'prose')} prose, "
            f"{sum(1 for c in all_chunks if c.chunk_type == 'table')} table)"
        )

        return all_chunks

    def chunk_filings(self, filings: list[ParsedFiling]) -> list[Chunk]:
        """Chunk a batch of filings."""
        all_chunks = []
        for filing in filings:
            all_chunks.extend(self.chunk_filing(filing))
        return all_chunks

    # ── Path 1: prose chunking ────────────────────────────────────────────────

    def _chunk_prose(self, section: ParsedSection) -> list[Chunk]:
        """
        Sliding window over the prose text of a section.

        Strategy:
          1. Tokenize the full section text
          2. Slide a window of `prose_chunk_tokens` tokens
          3. Step forward by (prose_chunk_tokens - prose_overlap_tokens) each time
          4. Decode each window back to text
          5. Prepend a context prefix so every chunk is self-contained
        """
        if not section.text.strip():
            return []

        tokens = TOKENIZER.encode(section.text)
        if len(tokens) < self.min_chunk_tokens:
            return []

        step   = self.prose_chunk_tokens - self.prose_overlap_tokens
        chunks = []
        idx    = 0
        window_start = 0

        while window_start < len(tokens):
            window_end   = min(window_start + self.prose_chunk_tokens, len(tokens))
            window_tokens = tokens[window_start:window_end]
            window_text   = TOKENIZER.decode(window_tokens)

            # Prepend a context header — this makes each chunk self-contained
            # so the LLM doesn't need surrounding chunks to answer questions
            context_prefix = self._build_prose_prefix(section, idx)
            full_text = f"{context_prefix}\n\n{window_text}"

            chunk = Chunk(
                chunk_id    = self._make_id(section, "prose", idx),
                text        = full_text,
                ticker      = section.ticker,
                company_name = "",   # filled later from yfinance store
                form_type   = section.form_type,
                fiscal_year = section.fiscal_year,
                quarter     = section.quarter,
                section     = section.section,
                chunk_type  = "prose",
                chunk_index = idx,
                token_count = len(TOKENIZER.encode(full_text)),
                char_count  = len(full_text),
            )
            chunks.append(chunk)

            # Stop if we've covered everything
            if window_end == len(tokens):
                break

            window_start += step
            idx          += 1

        return chunks

    # ── Path 2: table chunking ────────────────────────────────────────────────

    def _chunk_tables(self, section: ParsedSection) -> list[Chunk]:
        """
        One chunk per table.
        If a table exceeds table_max_tokens, split it into row groups
        while always keeping the header row attached to each group.
        """
        chunks = []

        for table_idx, table_text in enumerate(section.tables):
            table_tokens = TOKENIZER.encode(table_text)

            if len(table_tokens) <= self.table_max_tokens:
                # Table fits in one chunk — ideal case
                sub_chunks = [table_text]
            else:
                # Split by row groups, preserving the header
                sub_chunks = self._split_table_by_rows(table_text)

            for sub_idx, sub_text in enumerate(sub_chunks):
                prefix    = self._build_table_prefix(section, table_idx, sub_idx, len(sub_chunks))
                full_text = f"{prefix}\n\n{sub_text}"
                token_count = len(TOKENIZER.encode(full_text))

                if token_count < self.min_chunk_tokens:
                    continue

                chunk_idx = table_idx * 100 + sub_idx   # unique within section
                chunks.append(Chunk(
                    chunk_id     = self._make_id(section, "table", chunk_idx),
                    text         = full_text,
                    ticker       = section.ticker,
                    company_name = "",
                    form_type    = section.form_type,
                    fiscal_year  = section.fiscal_year,
                    quarter      = section.quarter,
                    section      = section.section,
                    chunk_type   = "table",
                    chunk_index  = chunk_idx,
                    token_count  = token_count,
                    char_count   = len(full_text),
                    table_index  = table_idx,
                ))

        return chunks

    def _split_table_by_rows(self, table_text: str) -> list[str]:
        """
        Split an oversized table into row groups, each with the header row prepended.

        The pipe-delimited format from FilingParser makes this straightforward:
        row 0 is always the header, subsequent rows are data rows.
        """
        rows = table_text.strip().split("\n")
        if len(rows) < 2:
            return [table_text]

        header = rows[0]
        data_rows = rows[1:]

        # How many data rows fit per chunk alongside the header?
        header_tokens = len(TOKENIZER.encode(header))
        budget = self.table_max_tokens - header_tokens - 10   # 10 token buffer
        groups = []
        current_rows = []
        current_tokens = 0

        for row in data_rows:
            row_tokens = len(TOKENIZER.encode(row))
            if current_tokens + row_tokens > budget and current_rows:
                groups.append(header + "\n" + "\n".join(current_rows))
                current_rows   = [row]
                current_tokens = row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        if current_rows:
            groups.append(header + "\n" + "\n".join(current_rows))

        return groups if groups else [table_text]

    # ── Context prefixes ──────────────────────────────────────────────────────

    @staticmethod
    def _build_prose_prefix(section: ParsedSection, chunk_index: int) -> str:
        """
        A short header prepended to every prose chunk.
        Makes each chunk self-contained — the LLM and retriever both
        benefit from knowing what document and section this came from
        without needing to look at surrounding chunks.
        """
        quarter_str = f"Q{section.quarter}" if section.quarter else "Annual"
        section_readable = section.section.replace("_", " ").title()
        return (
            f"[{section.ticker} | {section.form_type} | "
            f"FY{section.fiscal_year} {quarter_str} | "
            f"Section: {section_readable} | Chunk {chunk_index}]"
        )

    @staticmethod
    def _build_table_prefix(
        section: ParsedSection,
        table_index: int,
        sub_index: int,
        total_parts: int,
    ) -> str:
        quarter_str = f"Q{section.quarter}" if section.quarter else "Annual"
        section_readable = section.section.replace("_", " ").title()
        part_str = f" (part {sub_index + 1}/{total_parts})" if total_parts > 1 else ""
        return (
            f"[{section.ticker} | {section.form_type} | "
            f"FY{section.fiscal_year} {quarter_str} | "
            f"Section: {section_readable} | "
            f"Table {table_index}{part_str}]"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(section: ParsedSection, chunk_type: str, idx: int) -> str:
        quarter_str = f"Q{section.quarter}" if section.quarter else "annual"
        return (
            f"{section.ticker}_{section.form_type}_{section.fiscal_year}_"
            f"{quarter_str}_{section.section}_{chunk_type}_{idx}"
        ).lower().replace("-", "_")