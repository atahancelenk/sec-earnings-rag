import logging
from edgar_downloader import EDGARDownloader
from pdf_parser import FilingParser
from chunker import Chunker
from embedder import Embedder

logging.basicConfig(level=logging.INFO)

TICKERS = ["AAPL", "MSFT", "NVDA"]

def main():
    downloader = EDGARDownloader(output_dir="data/raw")
    parser     = FilingParser()
    chunker    = Chunker()
    embedder   = Embedder()

    all_chunks = []

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}")
        print(f"{'='*50}")

        # Download most recent 2 x 10-K and 4 x 10-Q
        records = downloader.download_filings(
            ticker,
            form_types=["10-K", "10-Q"],
            limit=3,
        )

        for record in records:
            if not record.local_path:
                continue
                
            parsed = parser.parse(record)
            if not parsed:
                continue

            chunks = chunker.chunk_filing(parsed)
            all_chunks.extend(chunks)

            # Quick quality report per filing
            prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
            table_chunks = [c for c in chunks if c.chunk_type == "table"]
            avg_tokens   = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0

            print(f"{record.form_type} {record.fiscal_year}"
                  f" {len(prose_chunks)} prose + {len(table_chunks)} table chunks"
                  f" (avg {avg_tokens:.0f} tokens")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding and upserting to Pinecone...")

    total = embedder.embed_and_upsert(all_chunks)
    print(f"Done — {total} vectors upserted to Pinecone")
    
if __name__ == "__main__":
    main()