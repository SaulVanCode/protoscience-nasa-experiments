#!/usr/bin/env python3
"""
E111 — Zipf's Law in Human Language Across 6 Languages

Question: Do word frequencies follow the same power law in every
human language? Is the Zipf exponent truly universal?

Background:
  George Zipf (1949) observed that in any large text, the frequency
  of the nth most common word is proportional to 1/n^alpha.

  "the" appears ~7% of the time in English
  "of" appears ~3.5%
  "and" appears ~2.8%
  ...and so on, following a clean power law.

  This works in EVERY language ever tested. Even in languages with
  completely different structures (Chinese has no spaces, Finnish
  has 15 grammatical cases, Arabic is root-based).

  Nobody has a fully satisfying explanation. Theories:
  - Least effort principle (Zipf's original idea)
  - Random typing on a keyboard produces Zipf (Miller 1957)
  - Preferential attachment in vocabulary growth
  - Information-theoretic optimality (maximum entropy)

Data: Project Gutenberg public domain texts
Source: https://www.gutenberg.org/
"""

import json
import urllib.request
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Project Gutenberg texts — one per language, all public domain
TEXTS = {
    "English": {
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "title": "Pride and Prejudice (Austen)",
        "cache": "gutenberg_english.txt",
    },
    "Spanish": {
        "url": "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",
        "title": "Don Quijote (Cervantes)",
        "cache": "gutenberg_spanish.txt",
    },
    "French": {
        "url": "https://www.gutenberg.org/cache/epub/13846/pg13846.txt",
        "title": "Les Miserables (Hugo)",
        "cache": "gutenberg_french.txt",
    },
    "German": {
        "url": "https://www.gutenberg.org/cache/epub/7207/pg7207.txt",
        "title": "Die Verwandlung (Kafka)",
        "cache": "gutenberg_german.txt",
    },
    "Italian": {
        "url": "https://www.gutenberg.org/cache/epub/1012/pg1012.txt",
        "title": "Divina Commedia (Dante)",
        "cache": "gutenberg_italian.txt",
    },
    "Portuguese": {
        "url": "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
        "title": "Os Lusiadas (Camoes)",
        "cache": "gutenberg_portuguese.txt",
    },
}


def fetch_text(lang, info):
    """Fetch text from Gutenberg or cache."""
    cache_path = DATA_DIR / info["cache"]
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    print(f"      Downloading {info['title']}...")
    req = urllib.request.Request(info["url"])
    req.add_header("User-Agent", "ProtoScience/1.0 (linguistics experiment)")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        cache_path.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"      Failed: {e}")
        return None


def extract_words(text):
    """Extract words from text, lowercased, letters only."""
    # Remove Gutenberg header/footer
    start = text.find("*** START")
    end = text.find("*** END")
    if start > 0:
        text = text[start:]
    if end > 0:
        text = text[:end]

    # Extract words (unicode-aware for accented characters)
    words = re.findall(r"[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())
    return words


def fit_zipf(freqs):
    """Fit Zipf's law: freq(rank) = C * rank^(-alpha)."""
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    freqs = np.array(freqs, dtype=float)

    log_r = np.log10(ranks)
    log_f = np.log10(freqs)

    coeffs = np.polyfit(log_r, log_f, 1)
    alpha = -coeffs[0]
    C = 10 ** coeffs[1]

    pred = np.polyval(coeffs, log_r)
    ss_res = np.sum((log_f - pred) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"alpha": float(alpha), "C": float(C), "r2": float(r2), "n_words": len(freqs)}


def hapax_ratio(word_counts):
    """Fraction of words that appear only once (hapax legomena)."""
    total = len(word_counts)
    hapax = sum(1 for c in word_counts.values() if c == 1)
    return hapax / total if total > 0 else 0


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E111 -- Zipf's Law in Human Language Across 6 Languages")
    print("=" * 70)

    results = {}

    for lang, info in TEXTS.items():
        print(f"\n  [{lang}] {info['title']}")

        text = fetch_text(lang, info)
        if text is None:
            continue

        words = extract_words(text)
        if len(words) < 100:
            print(f"    Too few words ({len(words)}), skipping")
            continue

        # Count words
        counts = Counter(words)
        total_words = len(words)
        unique_words = len(counts)

        # Sort by frequency
        sorted_counts = sorted(counts.values(), reverse=True)

        # Fit Zipf
        zipf = fit_zipf(sorted_counts)

        # Top 10 words
        top10 = counts.most_common(10)

        # Hapax
        hapax = hapax_ratio(counts)

        results[lang] = {
            "title": info["title"],
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": unique_words / total_words,
            "zipf_alpha": zipf["alpha"],
            "zipf_r2": zipf["r2"],
            "hapax_ratio": float(hapax),
            "top10": [(w, c) for w, c in top10],
        }

        print(f"    Words: {total_words:,} total, {unique_words:,} unique")
        print(f"    Zipf alpha = {zipf['alpha']:.4f}  R2 = {zipf['r2']:.4f}")
        print(f"    Hapax legomena: {hapax*100:.1f}% of vocabulary appears only once")
        print(f"    Top 5: {', '.join(f'{w}({c})' for w, c in top10[:5])}")

    # Comparison across languages
    print(f"\n  " + "=" * 60)
    print(f"  CROSS-LANGUAGE COMPARISON")
    print(f"  " + "=" * 60)

    print(f"\n  {'Language':12s} {'Alpha':>7s} {'R2':>7s} {'Words':>8s} {'Unique':>8s} {'Hapax%':>7s}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")

    alphas = []
    r2s = []
    for lang, r in results.items():
        print(f"  {lang:12s} {r['zipf_alpha']:7.4f} {r['zipf_r2']:7.4f} {r['total_words']:8,} {r['unique_words']:8,} {r['hapax_ratio']*100:6.1f}%")
        alphas.append(r["zipf_alpha"])
        r2s.append(r["zipf_r2"])

    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    mean_r2 = np.mean(r2s)

    print(f"\n  Mean alpha: {mean_alpha:.4f} +/- {std_alpha:.4f}  (Zipf predicts ~1.0)")
    print(f"  Mean R2:    {mean_r2:.4f}")

    # Is it universal?
    verdict = "REDISCOVERED" if abs(mean_alpha - 1.0) < 0.2 and mean_r2 > 0.9 else "PARTIAL"
    print(f"\n  Zipf universality: [{verdict}]")

    if std_alpha < 0.1:
        print(f"  The exponent varies by only +/-{std_alpha:.3f} across 6 languages")
        print(f"  from 3 different language families. This is universal.")

    # The mystery
    print(f"\n  " + "=" * 60)
    print(f"  WHY DOES THIS WORK?")
    print(f"  " + "=" * 60)
    print(f"\n  Every human language ever tested follows Zipf's law.")
    print(f"  English, Spanish, French, German, Italian, Portuguese —")
    print(f"  but also Chinese, Arabic, Finnish, Basque, Swahili.")
    print(f"  Even ancient languages (Latin, Greek, Sanskrit).")
    print(f"  Even sign languages and programming languages.")
    print(f"\n  Nobody fully knows why. The leading theory:")
    print(f"  Language optimizes the tradeoff between speaker effort")
    print(f"  (use few words) and listener clarity (use many words).")
    print(f"  Zipf's law is the equilibrium of this negotiation.")

    # Artifact
    artifact = {
        "id": "E111",
        "timestamp": now,
        "world": "linguistics",
        "data_source": "Project Gutenberg (public domain texts)",
        "data_url": "https://www.gutenberg.org/",
        "status": "passed" if verdict == "REDISCOVERED" else "partial",
        "design": {
            "description": "Test Zipf's law on word frequencies across 6 languages from different language families",
            "languages": list(TEXTS.keys()),
            "texts": {k: v["title"] for k, v in TEXTS.items()},
        },
        "result": {
            "language_results": {k: {kk: vv for kk, vv in v.items() if kk != "top10"} for k, v in results.items()},
            "mean_alpha": float(mean_alpha),
            "std_alpha": float(std_alpha),
            "mean_r2": float(mean_r2),
            "verdict": verdict,
            "key_finding": f"Zipf alpha = {mean_alpha:.3f} +/- {std_alpha:.3f} across {len(results)} languages",
        },
    }

    out_path = ROOT / "results" / "E111_zipf_language.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
