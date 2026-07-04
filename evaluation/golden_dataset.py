"""
Golden evaluation dataset for the SEC Earnings RAG system.

Each entry is a question with a ground-truth answer derived by manually
reading the actual filing. This is what RAGAS compares retrieval and
generation against — it must be accurate or the eval scores are meaningless.

Guidelines for writing good entries:
  - Pull exact figures from the filing yourself (don't trust the LLM's answer)
  - Include a mix of simple lookups and harder synthesis questions
  - Cover multiple tickers, form types, and sections
  - ground_truth should be a complete, factual sentence — not just a number
"""

from dataclasses import dataclass


@dataclass
class GoldenExample:
    question:     str
    ground_truth: str          # the factually correct answer, written by you
    ticker:       str | None = None
    form_type:    str | None = None
    fiscal_year:  int | None = None


GOLDEN_DATASET: list[GoldenExample] = [
    # -------------------- Apple (10-K) --------------------
    GoldenExample(
        question="What was Apple's total net sales in fiscal year 2024?",
        ground_truth="Apple's total net sales in fiscal year 2024 were $391.03 billion.",
        ticker="AAPL", form_type="10-K", fiscal_year=2024,
    ),
    GoldenExample(
        question="What was Apple's net income in fiscal year 2024?",
        ground_truth="Apple's net income for fiscal year 2024 was $93.73 billion.",
        ticker="AAPL", form_type="10-K", fiscal_year=2024,
    ),
    
    # -------------------- Microsoft (10-K) --------------------
    GoldenExample(
        question="What was Microsoft's total revenue in fiscal year 2024?",
        ground_truth="Microsoft's total revenue in fiscal year 2024 was $245.1 billion.",
        ticker="MSFT", form_type="10-K", fiscal_year=2024,
    ),
    GoldenExample(
        question="What AI-related risks did Microsoft identify in its annual report?",
        ground_truth="Microsoft highlighted risks regarding the rapidly evolving AI regulatory landscape, intellectual property disputes, intense competition, and potential ethical concerns.",
        ticker="MSFT", form_type="10-K", fiscal_year=2024,
    ),

    # -------------------- NVIDIA (10-K) --------------------
    GoldenExample(
        question="What was NVIDIA's total revenue in fiscal year 2024?",
        ground_truth="NVIDIA's total revenue in fiscal year 2024 was $60.92 billion.",
        ticker="NVDA", form_type="10-K", fiscal_year=2024,
    ),
    GoldenExample(
        question="Which business segment generated the highest revenue for NVIDIA in fiscal year 2024?",
        ground_truth="The Data Center segment generated the highest revenue, driven by generative AI demand.",
        ticker="NVDA", form_type="10-K", fiscal_year=2024,
    ),

    # -------------------- DÜZELTİLENLER: 10-Q Soruları (Belirli Çeyreklere Göre) --------------------
    # Apple için elimizde Q3 2025, Q1 2026, Q2 2026 var.
    GoldenExample(
        question="What was Apple's total revenue in Q1 of fiscal year 2026?",
        ground_truth="For Q1 FY2026, Apple reported total net sales of approximately $120.5 billion. [Mock Test Data]",
        ticker="AAPL", form_type="10-Q", fiscal_year=2026,
        # quarter="Q1" # Eğer veri modelin destekliyorsa eklenebilir.
    ),
    
    # Microsoft için elimizde Q4 2025, Q1 2026, Q2 2026 var.
    GoldenExample(
        question="Why did Microsoft's operating expenses change during Q2 of fiscal year 2026?",
        ground_truth="Operating expenses increased primarily due to continued heavy capital expenditures in AI infrastructure and data center expansion. [Mock Test Data]",
        ticker="MSFT", form_type="10-Q", fiscal_year=2026,
    ),

    # NVIDIA için elimizde Q3 2025, Q4 2025, Q2 2026 var.
    GoldenExample(
        question="How did NVIDIA's gross margin perform in Q4 of fiscal year 2025?",
        ground_truth="NVIDIA's gross margin expanded significantly, maintaining levels above 75% due to the massive scale and pricing power of their Data Center GPU sales. [Mock Test Data]",
        ticker="NVDA", form_type="10-Q", fiscal_year=2025,
    ),

    # -------------------- Cross-document reasoning --------------------
    GoldenExample(
        question="Which company reported the highest total revenue in fiscal year 2024: Apple, Microsoft, or NVIDIA?",
        ground_truth="Apple reported the highest total revenue at $391.0 billion, compared to Microsoft's $245.1 billion and NVIDIA's $60.9 billion.",
        ticker=None, form_type="10-K", fiscal_year=2024,
    ),

    # -------------------- YENİ EKLENEN 5 SORU --------------------
    
    # YENİ 1: Yıllar Arası Karşılaştırma (Trend Analizi)
    GoldenExample(
        question="How did Apple's total net sales change from fiscal year 2023 to fiscal year 2025?",
        ground_truth="To answer this, the system must retrieve both AAPL_10-K_2023 and AAPL_10-K_2025 to calculate the percentage or absolute difference in total net sales.",
        ticker="AAPL", form_type="10-K", fiscal_year=None, # Sistem iki farklı yıla bakmalı
    ),

    # YENİ 2: Yeni 10-K Belgelerinden Spesifik Veri Çekimi
    GoldenExample(
        question="What was the total revenue for Microsoft's Intelligent Cloud segment in fiscal year 2025?",
        ground_truth="Microsoft's Intelligent Cloud revenue for FY2025 reached approximately $120.8 billion, reflecting sustained Azure growth. [Mock Test Data]",
        ticker="MSFT", form_type="10-K", fiscal_year=2025,
    ),

    # YENİ 3: Belgeler Arası 10-Q Kıyaslaması (Aynı Çeyrek)
    GoldenExample(
        question="Based on their respective Q2 2026 10-Q filings, did Microsoft or Apple report higher operating expenses?",
        ground_truth="The system must extract operating expenses from MSFT_10-Q_2026_Q2 and AAPL_10-Q_2026_Q2 to compare and state which is higher.",
        ticker=None, form_type="10-Q", fiscal_year=2026,
    ),

    # YENİ 4: Risk Faktörleri (Derinlemesine Okuma)
    GoldenExample(
        question="Did NVIDIA mention export controls to China as a material risk factor in its fiscal year 2025 10-K?",
        ground_truth="Yes, NVIDIA explicitly mentioned U.S. government export controls to China and other targeted regions as a significant material risk affecting its Data Center revenue. [Mock Test Data]",
        ticker="NVDA", form_type="10-K", fiscal_year=2025,
    ),

    # YENİ 5: "Should Fail" (Olmayan Belgeyi Sorgulama - Sistemin Halüsinasyon Testi)
    GoldenExample(
        question="What was Apple's net income in Q4 of fiscal year 2025?",
        ground_truth="The requested information is unavailable because the AAPL_10-Q_2025_Q4 filing is not included in the indexed dataset.",
        ticker="AAPL", form_type="10-Q", fiscal_year=2025,
    ),

    # -------------------- Eski Should Fail'ler --------------------
    GoldenExample(
        question="What was Tesla's total revenue in fiscal year 2024?",
        ground_truth="The requested information is unavailable because Tesla filings are not included in the indexed SEC dataset.",
        ticker="TSLA", form_type="10-K", fiscal_year=2024,
    ),
]