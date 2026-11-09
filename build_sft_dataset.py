"""
Offline Telkomsel Product Catalog -> SFT Dataset Builder
==========================================================
100% local. No API calls, no external network access, no data leaves this
machine. Uses regex pattern-matching to pull structured facts out of the
catalog's free-text fields, then fills a bank of hand-written bilingual
question templates to produce closed-book instruction/output pairs.

Usage:
    python build_sft_dataset.py input.xlsx output.jsonl

Requires: pandas, beautifulsoup4, openpyxl
    pip install pandas beautifulsoup4 openpyxl --break-system-packages
"""

import sys
import json
import re
import random
from bs4 import BeautifulSoup
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# 1. CLEANING
# ---------------------------------------------------------------------------

def clean_html(text):
    """Strip HTML tags, return plain text."""
    if pd.isna(text) or text is None:
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ").strip()

def html_list_items(text):
    """Extract <li> items if present, else fall back to cleaned plain text
    split on sentence-ending periods."""
    if pd.isna(text) or text is None:
        return []
    soup = BeautifulSoup(str(text), "html.parser")
    items = [li.get_text().strip() for li in soup.find_all("li")]
    if items:
        return [i for i in items if i]
    plain = clean_html(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain) if s.strip()]


# ---------------------------------------------------------------------------
# 2. RULE-BASED FACT EXTRACTION
# Add / edit these regex patterns to match phrasings that actually appear in
# your full catalog -- this is the main thing you'll want to extend once you
# run this over more than one row and see what other T&C phrasings show up.
# ---------------------------------------------------------------------------

EXTRACTORS = [
    # (fact_key, regex, value_group_or_None, fact_type)
    ("duration_days", r"berlaku\s+selama\s+(\d+)\s*hari", 1, "number"),
    ("call_quota_minutes", r"kuota\s+(\d+)\s*menit", 1, "number"),
    ("data_quota_gb", r"kuota\s+(?:internet\s+)?(\d+(?:[.,]\d+)?)\s*GB", 1, "number"),
    ("data_quota_mb", r"kuota\s+(?:internet\s+)?(\d+)\s*MB", 1, "number"),
    ("sms_quota", r"(\d+)\s*SMS", 1, "number"),
    ("max_purchase_per_day", r"maksimum\s+(\d+)\s*kali\s+dalam\s+satu\s+hari", 1, "number"),
    ("ppn_included", r"(harga\s+sudah\s+termasuk\s+PPN)", None, "bool"),
    ("adn_excluded", r"(tidak\s+berlaku\s+untuk\s+penggunaan\s+telepon\s+ke\s+nomor\s+ADN)", None, "bool"),
]

def extract_facts(row):
    """Returns a dict of fact_key -> value extracted from this product row."""
    long_desc = clean_html(row.get("long_description_bahasa_indonesia_ml4", ""))
    tnc_items = html_list_items(row.get("terms_and_conditions_ml4", ""))
    blob = long_desc + " " + " ".join(tnc_items)

    facts = {}
    for key, pattern, group, ftype in EXTRACTORS:
        m = re.search(pattern, blob, re.IGNORECASE)
        if m:
            facts[key] = m.group(group) if group else True

    # Always-available structured fields (no regex needed, straight from columns)
    facts["product_id"] = row.get("product_offer_id_ml4")
    facts["name_id"] = row.get("commercial_name_bahasa_indonesia_ml4")
    facts["name_en"] = row.get("commercial_name_english_ml4")
    facts["short_desc"] = row.get("short_description_bahasa_indonesia_ml4")
    start_date = row.get("available_start_date_ml4")
    facts["start_date"] = str(start_date)[:10] if pd.notna(start_date) else None

    # Keep raw cleaned text too, for fallback/summary facts and verification
    facts["_long_desc_raw"] = long_desc
    facts["_tnc_raw_items"] = tnc_items
    return facts


# ---------------------------------------------------------------------------
# 3. TEMPLATE BANKS (bilingual, per fact type)
# Each entry maps a fact_key to a list of (instruction_template, answer_template)
# pairs. {name} / {value} get substituted. Add more phrasings here any time --
# more entries = more paraphrase diversity = better extraction per the research.
# ---------------------------------------------------------------------------

TEMPLATES = {
    "duration_days": [
        ("Berapa lama masa berlaku paket {name}?", "Paket {name} berlaku selama {value} hari."),
        ("Paket {name} itu aktifnya berapa hari?", "Paket ini aktif selama {value} hari sejak diaktifkan."),
        ("Masa aktif paket {name} berapa lama?", "Masa aktif paket {name} adalah {value} hari."),
        ("Kalau saya beli {name}, sampai kapan bisa dipakai?", "Paket {name} berlaku selama {value} hari sejak pembelian."),
        ("Berapa hari sih masa aktif {name}?", "Masa aktifnya {value} hari."),
    ],
    "call_quota_minutes": [
        ("Berapa kuota menit telepon dari paket {name}?", "Paket {name} memberikan kuota {value} menit telepon ke semua operator."),
        ("{name} itu dapat berapa menit?", "Kuotanya {value} menit telepon ke semua operator."),
        ("Isi paket {name} apa saja?", "Isinya adalah kuota {value} menit telepon ke semua operator."),
        ("Apakah paket ini memberi {value} menit nelpon ke semua operator?", "Ya, paket {name} memberikan {value} menit telepon ke semua operator."),
        ("Paket {name} kuota nelponnya berapa menit?", "Kuota nelponnya {value} menit ke semua operator."),
    ],
    "data_quota_gb": [
        ("Berapa kuota data dari paket {name}?", "Paket {name} memberikan kuota data sebesar {value} GB."),
        ("{name} dapat kuota internet berapa GB?", "Kuota internetnya {value} GB."),
        ("Paket {name} itu kuota internetnya berapa?", "Kuota internetnya sebesar {value} GB."),
    ],
    "data_quota_mb": [
        ("Berapa kuota data dari paket {name}?", "Paket {name} memberikan kuota data sebesar {value} MB."),
        ("{name} dapat kuota internet berapa MB?", "Kuota internetnya {value} MB."),
    ],
    "sms_quota": [
        ("Berapa kuota SMS dari paket {name}?", "Paket {name} memberikan kuota {value} SMS."),
        ("Apakah {name} dapat kuota SMS juga?", "Ya, paket ini memberikan kuota {value} SMS."),
    ],
    "max_purchase_per_day": [
        ("Berapa kali maksimal saya bisa membeli paket {name} dalam sehari?", "Paket {name} bisa dibeli maksimum {value} kali dalam satu hari."),
        ("Ada batas pembelian untuk paket {name}?", "Ada, maksimum {value} kali pembelian per hari."),
        ("Bolehkah beli {name} lebih dari {value} kali sehari?", "Tidak boleh, batas pembeliannya maksimum {value} kali dalam satu hari."),
        ("Berapa batas beli harian untuk paket {name}?", "Batas pembelian paket {name} adalah maksimum {value} kali dalam satu hari."),
    ],
    "ppn_included": [
        ("Apakah harga paket {name} sudah termasuk pajak?", "Ya, harga paket {name} sudah termasuk PPN."),
        ("Apakah saya perlu bayar pajak tambahan untuk {name}?", "Tidak perlu, harga paket {name} sudah termasuk PPN."),
        ("Harga {name} itu belum kena pajak ya?", "Sudah termasuk, harga paket {name} sudah termasuk PPN."),
    ],
    "adn_excluded": [
        ("Apakah paket {name} bisa dipakai untuk menelepon nomor ADN?", "Tidak. Paket {name} tidak berlaku untuk penggunaan telepon ke nomor ADN."),
        ("Ada batasan penggunaan untuk paket {name}?", "Ya, paket {name} tidak berlaku untuk telepon ke nomor ADN."),
        ("Bisa nggak nelpon nomor ADN pakai paket {name}?", "Tidak bisa, nomor ADN dikecualikan dari paket {name}."),
    ],
    "start_date": [
        ("Sejak kapan paket {name} tersedia?", "Paket {name} tersedia sejak {value}."),
        ("Kapan paket {name} mulai aktif dijual?", "Paket {name} mulai tersedia sejak tanggal {value}."),
    ],
}

# Bidirectional product-ID <-> name templates (defeats the reversal curse)
ID_TEMPLATES = [
    ("Apa product offer ID dari paket {name}?", "Product offer ID paket {name} adalah {id}."),
    ("ID produk {name} berapa?", "ID-nya adalah {id}."),
    ("Produk apa yang memiliki product offer ID {id}?", "Product offer ID {id} adalah paket {name}."),
    ("Paket apa itu ID {id}?", "ID {id} adalah paket {name}."),
]

# Whole-product summary templates (uses the raw cleaned description directly,
# no regex needed -- guarantees at least one solid fact per product even if
# no regex patterns matched)
SUMMARY_TEMPLATES = [
    ("Apa itu paket {name}?", "{name} adalah {short_desc_lower} {long_desc}"),
    ("Jelaskan paket {name} secara singkat.", "{long_desc}"),
    ("Ceritakan tentang paket {name}.", "{long_desc}"),
]

# Attribute types we might template for -- used to generate abstention
# examples for whichever ones AREN'T found in a given product's facts.
ABSTAINABLE = {
    "data_quota_gb": ("Apakah paket {name} memberikan kuota data internet?",
                       "Tidak ada informasi yang menyebutkan kuota data pada paket {name}."),
    "sms_quota": ("Apakah paket {name} memberikan kuota SMS?",
                  "Tidak ada informasi yang menyebutkan kuota SMS pada paket {name}."),
}


# ---------------------------------------------------------------------------
# 4. GENERATE INSTRUCTION/OUTPUT PAIRS FOR ONE PRODUCT
# ---------------------------------------------------------------------------

def generate_pairs_for_product(facts):
    pairs = []
    name = facts.get("name_id") or facts.get("name_en") or "produk ini"
    product_key = facts.get("product_id") or name  # used only for train/test splitting

    def fmt(template, **kw):
        return template.format(name=name, **kw)

    # Per-fact templated pairs
    for key, templates in TEMPLATES.items():
        if key not in facts or facts[key] in (None, False):
            continue
        value = facts[key]
        for instr_t, ans_t in templates:
            try:
                instr = fmt(instr_t, value=value)
                ans = fmt(ans_t, value=value)
            except (KeyError, IndexError):
                continue
            pairs.append({"instruction": instr, "output": ans})

    # Bidirectional ID <-> name pairs
    if facts.get("product_id"):
        for instr_t, ans_t in ID_TEMPLATES:
            pairs.append({
                "instruction": fmt(instr_t, id=facts["product_id"]),
                "output": fmt(ans_t, id=facts["product_id"]),
            })

    # Whole-product summary (guaranteed even with zero regex matches)
    if facts.get("_long_desc_raw"):
        short_desc_lower = (facts.get("short_desc") or "").lower()
        for instr_t, ans_t in SUMMARY_TEMPLATES:
            pairs.append({
                "instruction": fmt(instr_t),
                "output": fmt(ans_t,
                              long_desc=facts["_long_desc_raw"],
                              short_desc_lower=short_desc_lower),
            })

    # Abstention pairs for attributes NOT found on this product
    for key, (instr_t, ans_t) in ABSTAINABLE.items():
        if key not in facts:
            pairs.append({
                "instruction": fmt(instr_t),
                "output": fmt(ans_t),
            })

    for p in pairs:
        p["_product_key"] = product_key
    return pairs


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main(input_path, output_path, max_products=None, test_split=0.0, paraphrase_holdout=0.0):
    df = pd.read_excel(input_path)

    if max_products is not None:
        # Filter to rows that actually have a usable product name FIRST,
        # so max_products means "N real products", not "N raw rows"
        # (raw rows may include non-product entity types with blank names).
        name_cols = [c for c in ["commercial_name_bahasa_indonesia_ml4",
                                  "commercial_name_english_ml4"] if c in df.columns]
        has_name = df[name_cols].notna().any(axis=1) if name_cols else pd.Series(True, index=df.index)
        df = df[has_name].head(max_products)
        print(f"Limiting to first {max_products} products with a valid name "
              f"(out of {has_name.sum()} available in the file).")

    all_pairs = []
    skipped = 0

    for _, row in df.iterrows():
        facts = extract_facts(row)
        if not facts.get("name_id") and not facts.get("name_en"):
            skipped += 1
            continue
        pairs = generate_pairs_for_product(facts)
        all_pairs.extend(pairs)

    print(f"Products processed: {len(df) - skipped} (skipped {skipped} with no name)")
    print(f"Total SFT pairs generated: {len(all_pairs)}")

    if test_split > 0:
        # Split by WHOLE PRODUCT, not by row -- so no product's facts/paraphrases
        # leak across both sets. This tests generalization to unseen products,
        # not just memorization of phrasings already seen during training.
        product_keys = sorted(set(p["_product_key"] for p in all_pairs), key=str)
        random.shuffle(product_keys)
        n_test_products = max(1, round(len(product_keys) * test_split))
        test_products = set(product_keys[:n_test_products])
        train_products = set(product_keys[n_test_products:])

        train_pairs_full = [p for p in all_pairs if p["_product_key"] in train_products]
        test_pairs = [p for p in all_pairs if p["_product_key"] in test_products]
        random.shuffle(train_pairs_full)
        random.shuffle(test_pairs)

        # Second, independent holdout: within TRAIN products (already seen by the
        # model), pull out some of THEIR OWN paraphrase pairs into a separate eval
        # file. This isolates a different question from the whole-product test set:
        # "does it generalize to a phrasing of a fact it saw the fact for, just not
        # this exact question form?" -- vs. "does it generalize to a product it
        # never saw at all?". Held out per-product so every train product still
        # keeps at least 1 pair for actual training.
        seen_paraphrase_holdout_pairs = []
        train_pairs = []
        if paraphrase_holdout > 0:
            from collections import defaultdict
            by_product = defaultdict(list)
            for p in train_pairs_full:
                by_product[p["_product_key"]].append(p)
            for key, plist in by_product.items():
                random.shuffle(plist)
                n_hold = min(len(plist) - 1, round(len(plist) * paraphrase_holdout))  # keep >=1 for training
                n_hold = max(0, n_hold)
                seen_paraphrase_holdout_pairs.extend(plist[:n_hold])
                train_pairs.extend(plist[n_hold:])
            random.shuffle(train_pairs)
            random.shuffle(seen_paraphrase_holdout_pairs)
        else:
            train_pairs = train_pairs_full

        stem = output_path.rsplit(".", 1)[0]
        train_path = f"{stem}_train.jsonl"
        test_path = f"{stem}_test.jsonl"
        seen_paraphrase_path = f"{stem}_seen_paraphrase_test.jsonl"

        files_to_write = [(train_path, train_pairs), (test_path, test_pairs)]
        if paraphrase_holdout > 0:
            files_to_write.append((seen_paraphrase_path, seen_paraphrase_holdout_pairs))

        for path, pairs in files_to_write:
            with open(path, "w", encoding="utf-8") as f:
                for p in pairs:
                    clean = {"instruction": p["instruction"], "output": p["output"]}
                    f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        print(f"\nSplit by product: {len(train_products)} train products, "
              f"{len(test_products)} held-out test products")
        print(f"  Train:                {len(train_pairs)} pairs -> {train_path}")
        print(f"  Test (unseen products): {len(test_pairs)} pairs -> {test_path}")
        if paraphrase_holdout > 0:
            print(f"  Test (seen products, held-out phrasings): "
                  f"{len(seen_paraphrase_holdout_pairs)} pairs -> {seen_paraphrase_path}")
        print("\nSample train pairs:")
        for p in train_pairs[:3]:
            print(" -", {"instruction": p["instruction"], "output": p["output"]})
        print("\nSample test pairs (held-out products, unseen during training):")
        for p in test_pairs[:3]:
            print(" -", {"instruction": p["instruction"], "output": p["output"]})
        if paraphrase_holdout > 0 and seen_paraphrase_holdout_pairs:
            print("\nSample seen-product held-out-phrasing pairs "
                  "(product WAS trained on, this exact question form wasn't):")
            for p in seen_paraphrase_holdout_pairs[:3]:
                print(" -", {"instruction": p["instruction"], "output": p["output"]})
    else:
        random.shuffle(all_pairs)
        with open(output_path, "w", encoding="utf-8") as f:
            for p in all_pairs:
                clean = {"instruction": p["instruction"], "output": p["output"]}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        print(f"Written to: {output_path}")
        print("\nSample pairs:")
        for p in all_pairs[:5]:
            print(" -", {"instruction": p["instruction"], "output": p["output"]})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the catalog .xlsx file")
    parser.add_argument("output_path", help="Path to write the output .jsonl file")
    parser.add_argument("--max_products", type=int, default=None,
                         help="Only process the first N rows of the catalog "
                              "(useful for a quick small dataset / test run)")
    parser.add_argument("--test_split", type=float, default=0.0,
                         help="Fraction of PRODUCTS (not rows) to hold out entirely as a "
                              "test set, e.g. 0.1 for 10%%. Splits whole products so no "
                              "product's facts leak across train/test. Writes "
                              "<output>_train.jsonl and <output>_test.jsonl instead of "
                              "a single file when > 0.")
    parser.add_argument("--paraphrase_holdout", type=float, default=0.0,
                         help="Within TRAIN products only, fraction of each product's own "
                              "paraphrase pairs to hold out into a separate eval file "
                              "(<output>_seen_paraphrase_test.jsonl). Tests whether the "
                              "model generalizes to new phrasings of facts it WAS trained "
                              "on -- a different question from --test_split, which tests "
                              "generalization to products it never saw at all. Requires "
                              "--test_split > 0.")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.max_products, args.test_split, args.paraphrase_holdout)