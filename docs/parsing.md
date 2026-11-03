# Stage 1 — Document Parsing & Ingestion

## 1. Overview & Architecture

The parsing module (`app/parsing/`) implements Stage 1 of the AI Resume Analyzer pipeline. It ingests raw document bytes in-memory and outputs a normalized, structured `ParsedResume` model.

### Key Tenets
1. **Zero Disk Footprint & Strict Privacy**: Résumé bytes are processed strictly in memory (`io.BytesIO`). No temporary files are written to disk, and résumé text is never printed or logged.
2. **Format-Agnostic Ingestion**: Magic bytes are inspected first (`%PDF`, `PK\x03\x04`, `\x89PNG`, `\xff\xd8\xff`), followed by file extensions. PDF bytes with a `.docx` extension or DOCX bytes with a `.pdf` extension are correctly identified and parsed.
3. **Column-Aware Reading Order**: Rather than extracting text top-to-bottom across the entire page (which interleaves two-column layouts and destroys section boundaries), the parser clusters blocks by horizontal position (`x0`), identifies gutters, checks vertical overlap, and reconstructs natural column-by-column reading order.
4. **Table Structure Preservation**: Tables are detected via PyMuPDF's `find_tables()` (or python-docx table iteration), converting tabular data into structured line representations (`cell | cell`) while preserving logical position.
5. **OCR Fallback**: Image-only or scanned documents with low character counts trigger rasterization at 300 DPI, grayscale conversion, autocontrast enhancement, and Tesseract OCR extraction.
6. **Typed Exception Hierarchy**: Callers are protected from bare library exceptions. All parsing failures raise subclasses of `ParseError`.

```
Raw Bytes In-Memory (PDF / DOCX / Image)
                 │
                 ▼
      Format Detection (Magic Bytes First)
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
   [PDF]       [DOCX]     [Image]
     │           │           │
  PyMuPDF   python-docx   Tesseract
  (2-Col /   (Body-order     OCR
   Tables /   traversal)     (300 DPI)
   OCR fb)       │           │
     └───────────┼───────────┘
                 │
                 ▼
       clean_text() [NFKC, bullets, dehyphen]
                 │
                 ▼
       split_sections() [exact + rapidfuzz>=92]
                 │
                 ▼
       extract_contact_info() [email, phone, name]
                 │
                 ▼
     ParsedResume (Schema v1.1.0)
```

---

## 2. Configuration & Thresholds

All configuration parameters reside in `app/config.py` under the `Settings` class:

| Parameter | Default Value | Rationale |
|-----------|---------------|-----------|
| `max_file_bytes` | `10 * 1024 * 1024` (10 MB) | Prevents denial-of-service via massive payload uploads. |
| `max_pages` | `10` | Standard résumé limit; excess pages are ignored with a warning recorded in `ParsedResume.warnings`. |
| `min_chars_per_page` | `50` | Average character threshold under which a PDF is considered an image-only scan requiring OCR. |
| `ocr_dpi` | `300` | Rasterization resolution balancing Tesseract character recognition accuracy with CPU memory and processing time. |
| `rapidfuzz score_cutoff` | `92` | Fuzzy matching threshold for section heading synonyms, preventing over-eager matching of regular text. |
| Column Split Gutter | `50 pt` | Minimum horizontal gap between `x0` clusters required to partition a page into multi-column layout. |
| Column Overlap Ratio | `0.20` (20%) | Minimum vertical overlap between left and right block clusters required to confirm a two-column layout. |

---

## 3. Core Components

### 3.1. Text Cleaning & Normalization (`app/parsing/cleaner.py`)
- **Unicode NFKC**: Canonical decomposition followed by canonical composition, standardizing ligatures and fullwidth characters.
- **Zero-Width Stripping**: Removes invisible zero-width spaces (`\u200b`), joiners (`\u200c`, `\u200d`), BOM (`\ufeff`), and directional markers (`\u200e`, `\u200f`).
- **Bullet Normalization**: Maps diverse unicode bullets (`•`, `▪`, `◦`, `●`, `‣`) to standard markdown list item prefixes (`- `).
- **Lowercase-Lowercase De-hyphenation**: Joins words split across line breaks by hyphens (`([a-z])- *\n *([a-z])` -> `\1\2`). Prevents corrupting title compounds (e.g. `Pre-\nPost` or `go-to-market`).
- **Whitespace Collapsing**: Collapses redundant horizontal spaces within lines and limits inter-paragraph spacing to at most 1 blank line (max 2 consecutive newlines).

### 3.2. Canonical Section Splitting (`app/parsing/sections.py`)
Résumés are segmented into 11 canonical sections defined in `data/taxonomies/section_headings.csv`:
1. `summary`
2. `experience`
3. `education`
4. `skills`
5. `projects`
6. `certifications`
7. `awards`
8. `publications`
9. `languages`
10. `interests`
11. `references`

#### Heading Detection Rules:
- Line must be 4 words or fewer.
- Must not end with a trailing period.
- Normalizes trailing colons, case, whitespace, and `&` vs `and`.
- Matches exact heading synonyms first, followed by RapidFuzz ratio >= 92.
- Text before the first detected heading is stored under `"header"`.
- Duplicate headings are concatenated with newlines.

### 3.3. Contact Extraction (`app/parsing/contact.py`)
Extracts structured contact details:
- **Email**: Strict RFC-compliant email regex.
- **Phone**: Robust regex matching US standard (`555-0142`, `(555) 014-2345`) and international formats.
- **LinkedIn & GitHub**: Regex extracting profile URLs or handles.
- **Best-Effort Name**: Scans header lines for the first line consisting of 2-4 capitalized words with no digits, filtering out known heading stop words, degree credentials (e.g. `, BSN, RN`), and pipe delimiters.

### 3.4. Typed Exceptions (`app/parsing/errors.py`)
All parser errors inherit from `ParseError`:
- `UnsupportedFileTypeError`: Format not supported.
- `FileTooLargeError`: File size exceeds `settings.max_file_bytes`.
- `CorruptedFileError`: Truncated, invalid, or unreadable document stream.
- `PasswordProtectedError`: Encrypted PDF requiring a password.
- `EmptyDocumentError`: 0-byte file or 0 readable pages.

---

## 4. Evaluation Benchmark

The benchmark in `scripts/eval_parsing.py` evaluates all 5 synthetic fixtures against their `.expected.json` ground-truth sidecars:

| Fixture | Sections Prec / Rec | Contact Name / Email / Phone | Multi-Col | Tables | Images | OCR Used | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ds_clean.pdf` | 1.00 / 1.00 | PASS / PASS / PASS | False | False | False | False | ~125ms |
| `swe_two_column.pdf` | 1.00 / 1.00 | PASS / PASS / PASS | True | True | True | False | ~85ms |
| `accountant_table.pdf` | 1.00 / 1.00 | PASS / PASS / PASS | False | True | False | False | ~140ms |
| `chef_plain.docx` | 1.00 / 1.00 | PASS / PASS / PASS | False | False | False | False | ~15ms |
| `nurse_scanned.pdf` | 1.00 / 1.00 | PASS / PASS / PASS | False | False | True | True | ~2750ms |

### Key Validation Checks:
- **Nurse Scanned OCR Skill Recall**: 100% (5/5 expected skills detected in OCR text layer).
- **Swe Two-Column Interleaving**: Verified that experience text retains `Senior Backend Engineer` and `CloudScale Systems` without any sidebar interleaving (`Tech Stack`, `Academics`).
- **Edge-Case Resilience**: 8/8 in-memory edge cases cleanly raise typed errors or degrade with warnings without crashing.

---

## 5. Known Limitations & Edge Cases

1. **Skewed Scans & Warped Text**: Scans with heavy geometric distortion or excessive skew angles may degrade Tesseract OCR character accuracy without prior deskewing / perspective rectification.
2. **Complex Nested Tables**: Multi-level nested tables (tables inside table cells) flatten into sequential cell rows.
3. **Decorative Names & Non-Standard Punctuation**: Names containing numbers, unusual symbols, or single-name mononyms will not match the 2-4 capitalized words heuristic.
4. **Multilingual OCR**: Tesseract is configured for default English OCR (`eng`). Non-Latin scripts require installing additional Tesseract language packs.
