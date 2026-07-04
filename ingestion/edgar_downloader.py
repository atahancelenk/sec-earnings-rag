import os
import time
import json
import logging
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://data.sec.gov"
ARCHIVES_URL = "https://www.sec.gov"
SUBMISSIONS_URL = f"{BASE_URL}/submissions/CIK{{cik}}.json"

# SEC requires a descriptive User-Agent — use your real email or you'll get blocked
HEADERS = {
    "User-Agent": "SecEarningsRAG/1.0 your-email@example.com",
    "Accept-Encoding": "gzip, deflate",
}

REQUEST_DELAY = 0.12   # ~8 req/s — safely under the 10/s limit
MAX_RETRIES   = 3


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class FilingRecord:
    """Represents a single SEC filing with its metadata."""
    ticker:           str
    cik:              str
    form_type:        str        # "10-K" or "10-Q"
    accession_number: str        # e.g. "0000320193-23-000077"
    filed_date:       str        # "2023-11-03"
    fiscal_year:      int
    quarter:          Optional[int]   # None for 10-K, 1–4 for 10-Q
    primary_doc:      str = ""
    local_path:       Optional[Path] = field(default=None)


# ── Core downloader ───────────────────────────────────────────────────────────

class EDGARDownloader:
    """
    Downloads 10-K and 10-Q filings from SEC EDGAR.

    Usage:
        downloader = EDGARDownloader(output_dir="data/raw")
        records = downloader.download_filings("AAPL", form_types=["10-K", "10-Q"], limit=5)
    """

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── Public API ────────────────────────────────────────────────────────────

    def download_filings(
        self,
        ticker: str,
        form_types: list[str] = ["10-K", "10-Q"],
        limit: int = 5,
    ) -> list[FilingRecord]:
        """
        Download the most recent `limit` filings of each form type for a ticker.
        Returns a list of FilingRecords with local_path populated on success.
        """
        cik = self._resolve_cik(ticker)
        if not cik:
            logger.error(f"Could not resolve CIK for ticker {ticker}")
            return []

        logger.info(f"{ticker} → CIK {cik}")

        all_records = []
        for form_type in form_types:
            filings = self._get_filing_list(ticker, cik, form_type, limit)
            logger.info(f"Found {len(filings)} {form_type} filings for {ticker}")

            for record in filings:
                path = self._download_filing(record, cik)
                if path:
                    record.local_path = path
                    all_records.append(record)
                time.sleep(REQUEST_DELAY)

        return all_records

    # ── Step 1: ticker → CIK ─────────────────────────────────────────────────

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """
        Convert a ticker symbol to a zero-padded 10-digit CIK.
        EDGAR's company_tickers.json maps every listed ticker to its CIK.
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get_json(url)
        if not data:
            return None

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry["ticker"] == ticker_upper:
                # CIK must be zero-padded to 10 digits for the submissions URL
                return str(entry["cik_str"]).zfill(10)

        logger.warning(f"Ticker {ticker} not found in EDGAR company list")
        return None

    # ── Step 2: CIK → filing list ─────────────────────────────────────────────

    def _get_filing_list(
        self,
        ticker: str,
        cik: str,
        form_type: str,
        limit: int,
    ) -> list[FilingRecord]:
        """
        Pull the submission history for a CIK and extract matching filings.
        """
        url = SUBMISSIONS_URL.format(cik=cik)
        data = self._get_json(url)
        if not data:
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        accessions  = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        records = []
        for form, accession, date, primary_doc in zip(forms, accessions, filed_dates, primary_docs):
            if form != form_type:
                continue

            fiscal_year, quarter = self._parse_fiscal_period(date, form_type)

            records.append(FilingRecord(
                ticker=ticker.upper(),
                cik=cik,
                form_type=form_type,
                accession_number=accession,
                filed_date=date,
                fiscal_year=fiscal_year,
                quarter=quarter,
                primary_doc=primary_doc,
            ))

            if len(records) >= limit:
                break

        return records

    # ── Step 3: accession → document URL → download ───────────────────────────

    def _download_filing(self, record: FilingRecord, cik: str) -> Optional[Path]:
        if not record.primary_doc:
            logger.warning(f"No primary document specified for {record.accession_number}")
            return None
        

        accession_clean = record.accession_number.replace("-", "")
        cik_short = cik.lstrip("0")

        doc_url = (
            f"{ARCHIVES_URL}/Archives/edgar/data/{cik_short}"
            f"/{accession_clean}/{record.primary_doc}"
        )

        # Build a clean local filename
        quarter_str = f"Q{record.quarter}" if record.quarter else "annual"
        filename = f"{record.ticker}_{record.form_type}_{record.fiscal_year}_{quarter_str}.htm"
        local_path = self.output_dir / record.ticker / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded (idempotent re-runs)
        if local_path.exists():
            logger.info(f"Already exists, skipping: {local_path}")
            return local_path

        return self._download_file(doc_url, local_path)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get_json(self, url: str) -> Optional[dict]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                time.sleep(REQUEST_DELAY)
                return resp.json()
            except requests.HTTPError as e:
                if resp.status_code == 429:
                    wait = 2 ** attempt   # exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP {resp.status_code} for {url}: {e}")
                    return None
            except Exception as e:
                logger.error(f"Request failed for {url}: {e}")
                return None
        return None

    def _download_file(self, url: str, local_path: Path) -> Optional[Path]:
        try:
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    # ── Fiscal period helper ──────────────────────────────────────────────────

    @staticmethod
    def _parse_fiscal_period(filed_date: str, form_type: str) -> tuple[int, Optional[int]]:
        """
        Infer fiscal year and quarter from the filing date.
        This is an approximation — the filing date lags the period end by ~45 days.
        """
        from datetime import datetime
        dt = datetime.strptime(filed_date, "%Y-%m-%d")

        if form_type == "10-K":
            # Annual report — fiscal year is typically the year before the filing date
            fiscal_year = dt.year if dt.month > 3 else dt.year - 1
            return fiscal_year, None
        else:
            # Quarterly — map filing month to approximate quarter
            month_to_quarter = {
                1: 1, 2: 1, 3: 1,
                4: 2, 5: 2, 6: 2,
                7: 3, 8: 3, 9: 3,
                10: 4, 11: 4, 12: 4,
            }
            quarter = month_to_quarter[dt.month]
            return dt.year, quarter