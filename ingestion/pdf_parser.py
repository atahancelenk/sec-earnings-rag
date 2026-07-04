import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

# Section headings found in most 10-K and 10-Q filings
KNOWN_SECTIONS = [
    "business",
    "risk factors",
    "properties",
    "legal proceedings",
    "management",          # catches "MD&A" and its variants
    "quantitative and qualitative",
    "financial statements",
    "notes to financial",
    "controls and procedures",
    "other information",
]


@dataclass
class ParsedSection:
    """A single logical section extracted from a filing."""
    ticker:      str
    form_type:   str
    fiscal_year: int
    quarter:     Optional[int]
    section:     str          # normalized section name
    text:        str          # raw cleaned text
    char_count:  int
    tables:      list[str]    # extracted table text blocks


@dataclass
class ParsedFiling:
    """All sections extracted from a single filing document."""
    ticker:      str
    form_type:   str
    fiscal_year: int
    quarter:     Optional[int]
    local_path:  Path
    sections:    list[ParsedSection]

    @property
    def total_chars(self) -> int:
        return sum(s.char_count for s in self.sections)


class FilingParser:
    """
    Parses SEC EDGAR .htm filing documents into structured sections.

    Handles both prose sections (MD&A, Risk Factors) and tabular data
    (financial statements) with separate extraction paths.
    """

    def parse(self, record) -> Optional[ParsedFiling]:
        """
        Parse a downloaded FilingRecord into a ParsedFiling.
        `record` is an EDGARDownloader.FilingRecord with local_path set.
        """
        if not record.local_path or not record.local_path.exists():
            logger.error(f"File not found: {record.local_path}")
            return None

        raw_html = record.local_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw_html, "lxml")

        # Remove noise: scripts, styles, hidden elements
        self._strip_noise(soup)

        sections = self._extract_sections(soup, record)
        tables   = self._extract_tables(soup)

        # Attach tables to the most relevant section
        sections = self._assign_tables(sections, tables)

        filing = ParsedFiling(
            ticker=record.ticker,
            form_type=record.form_type,
            fiscal_year=record.fiscal_year,
            quarter=record.quarter,
            local_path=record.local_path,
            sections=sections,
        )

        logger.info(
            f"Parsed {record.ticker} {record.form_type} {record.fiscal_year}: "
            f"{len(sections)} sections, {filing.total_chars:,} chars"
        )
        return filing

    # ── Section extraction ────────────────────────────────────────────────────

    def _extract_sections(self, soup: BeautifulSoup, record) -> list[ParsedSection]:
        """
        Walk the document and split on section headings.
        Headings in EDGAR filings are almost always bold text or heading tags
        that match known section names.
        """
        sections = []
        current_section = "preamble"
        current_text    = []

        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "div"]):
            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue

            # Check if this element looks like a section heading
            heading = self._classify_heading(elem, text)
            if heading:
                # Save the previous section if it has content
                if current_text:
                    sections.append(self._make_section(
                        record, current_section,
                        " ".join(current_text)
                    ))
                current_section = heading
                current_text    = []
            else:
                current_text.append(text)

        # Don't forget the last section
        if current_text:
            sections.append(self._make_section(
                record, current_section,
                " ".join(current_text)
            ))

        # Drop sections that are too short to be meaningful
        sections = [s for s in sections if s.char_count > 200]
        return sections

    def _classify_heading(self, elem, text: str) -> Optional[str]:
        """
        Return a normalized section name if this element is a section heading,
        otherwise return None.

        EDGAR headings are identified by:
        1. Being a heading tag (h1–h4)
        2. Being a bold/strong element whose text matches a known section
        3. Being short (< 120 chars) and ALL CAPS or Title Case
        """
        text_lower = text.lower().strip()

        # Must be short to be a heading
        if len(text) > 120:
            return None

        is_heading_tag = elem.name in ("h1", "h2", "h3", "h4")
        is_bold = bool(elem.find(["b", "strong"])) or elem.name in ("b", "strong")
        looks_like_heading = text.isupper() or (
            text.istitle() and len(text.split()) <= 10
        )

        if not (is_heading_tag or is_bold or looks_like_heading):
            return None

        # Match against known section names
        for known in KNOWN_SECTIONS:
            if known in text_lower:
                return self._normalize_section_name(text_lower)

        return None

    @staticmethod
    def _normalize_section_name(raw: str) -> str:
        """Collapse variations of section names into canonical forms."""
        raw = raw.lower().strip()
        if "risk" in raw:
            return "risk_factors"
        if "management" in raw or "md&a" in raw or "discussion" in raw:
            return "mda"
        if "financial statement" in raw:
            return "financial_statements"
        if "notes to" in raw:
            return "notes_to_financials"
        if "business" in raw:
            return "business"
        if "quantitative" in raw:
            return "market_risk"
        if "controls" in raw:
            return "controls"
        return re.sub(r"\s+", "_", raw[:40])

    @staticmethod
    def _make_section(record, section_name: str, text: str) -> ParsedSection:
        cleaned = FilingParser._clean_text(text)
        return ParsedSection(
            ticker=record.ticker,
            form_type=record.form_type,
            fiscal_year=record.fiscal_year,
            quarter=record.quarter,
            section=section_name,
            text=cleaned,
            char_count=len(cleaned),
            tables=[],
        )

    # ── Table extraction ──────────────────────────────────────────────────────

    def _extract_tables(self, soup: BeautifulSoup) -> list[str]:
        """
        Extract all HTML tables as plain-text blocks.
        Each table becomes a single string with pipe-delimited rows —
        this is intentional: it keeps the table as one indivisible chunk
        rather than fragmenting it across chunk boundaries.
        """
        tables = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [
                    td.get_text(separator=" ", strip=True)
                    for td in tr.find_all(["td", "th"])
                ]
                # Skip rows that are all empty (common in EDGAR formatting)
                if any(c for c in cells):
                    rows.append(" | ".join(cells))

            table_text = "\n".join(rows)
            # Only keep tables with real content (min 3 rows, min 100 chars)
            if len(rows) >= 3 and len(table_text) > 100:
                tables.append(table_text)

        return tables

    def _assign_tables(
        self,
        sections: list[ParsedSection],
        tables: list[str],
    ) -> list[ParsedSection]:
        """
        Heuristically attach each table to the most relevant section.
        Financial statement tables go to financial_statements;
        everything else goes to the nearest prose section.
        """
        financial_keywords = {"revenue", "income", "loss", "assets", "liabilities",
                               "equity", "cash", "earnings", "per share", "diluted"}

        for table_text in tables:
            text_lower = table_text.lower()
            is_financial = any(kw in text_lower for kw in financial_keywords)

            if is_financial:
                target = next(
                    (s for s in sections if s.section == "financial_statements"),
                    sections[0] if sections else None,
                )
            else:
                target = sections[0] if sections else None

            if target:
                target.tables.append(table_text)

        return sections

    # ── Text cleaning ─────────────────────────────────────────────────────────

    @staticmethod
    def _strip_noise(soup: BeautifulSoup) -> None:
        """Remove elements that add no textual value."""
        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()
        # Remove hidden elements
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none")):
            tag.decompose()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and remove common EDGAR boilerplate noise."""
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove page break markers
        text = re.sub(r"-\s*\d+\s*-", " ", text)
        # Remove repeated dots (table of contents leaders)
        text = re.sub(r"\.{3,}", " ", text)
        # Remove form numbers inline (e.g. "Form 10-K" appearing mid-sentence)
        text = re.sub(r"\bForm\s+10-[KQ][/A]?\b", "", text)
        return text.strip()