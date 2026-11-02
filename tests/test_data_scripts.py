"""test_data_scripts.py – Offline unit tests for data download and verification scripts.

Strictly offline: all tests run against local fake fixtures using pytest's `tmp_path`.
No network calls are made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.download_data import (
    calculate_sha256,
    load_manifest,
    save_manifest,
    parse_args,
)
from scripts.verify_data import (
    CheckResult,
    format_table,
    verify_livecareer,
    verify_ner_synthetic,
    verify_ner_real_skills,
    verify_job_resume_fit,
    verify_linkedin_sample,
    verify_onet,
    verify_manifest,
)


def test_calculate_sha256(tmp_path: Path) -> None:
    """Check that calculate_sha256 produces exact hex digest."""
    test_file = tmp_path / "test.txt"
    content = b"The quick brown fox jumps over the lazy dog"
    test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    actual = calculate_sha256(test_file)
    assert actual == expected


def test_manifest_save_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving and atomic loading of data_manifest.json."""
    fake_manifest_path = tmp_path / "docs" / "data_manifest.json"
    fake_docs_dir = tmp_path / "docs"

    monkeypatch.setattr("scripts.download_data.MANIFEST_PATH", fake_manifest_path)
    monkeypatch.setattr("scripts.download_data.DOCS_DIR", fake_docs_dir)

    payload = {
        "dataset_a": {
            "name": "Test Dataset",
            "files": [{"path": "data/raw/test.parquet", "rows": 100}],
        }
    }

    save_manifest(payload)
    assert fake_manifest_path.exists()

    loaded = load_manifest()
    assert loaded == payload


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test default CLI options."""
    monkeypatch.setattr("sys.argv", ["download_data.py"])
    args = parse_args()
    assert args.only is None
    assert args.force is False


def test_parse_args_with_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI parsing with --only and --force."""
    monkeypatch.setattr("sys.argv", ["download_data.py", "--only", "livecareer", "--force"])
    args = parse_args()
    assert args.only == "livecareer"
    assert args.force is True


def test_format_table() -> None:
    """Test ASCII table formatting."""
    results = [
        CheckResult("LiveCareer", "data/raw/livecareer/train.parquet", "2,484 rows", "2,484 rows", "PASS"),
        CheckResult("FakeDataset", "data/raw/fake/test.parquet", "100 rows", "90 rows", "FAIL"),
    ]
    table = format_table(results)
    assert "LiveCareer" in table
    assert "PASS" in table
    assert "FAIL" in table
    assert "+-" in table


def test_verify_livecareer_pass_and_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test verification logic on fake LiveCareer dataset."""
    fake_raw = tmp_path / "data" / "raw"
    monkeypatch.setattr("scripts.verify_data.DATA_RAW", fake_raw)

    # Missing file -> FAIL
    res_missing = verify_livecareer()
    assert res_missing[0].status == "FAIL"

    # Valid file -> PASS
    lc_dir = fake_raw / "livecareer"
    lc_dir.mkdir(parents=True)
    categories = [f"Cat_{i}" for i in range(24)]
    # Create 2,484 rows evenly distributed across 24 categories
    rows = []
    for i in range(2484):
        rows.append({
            "ID": i,
            "Resume_str": f"Resume text {i}",
            "Resume_html": f"<p>Resume {i}</p>",
            "Category": categories[i % 24],
        })
    df_valid = pd.DataFrame(rows)
    df_valid.to_parquet(lc_dir / "train.parquet", index=False)

    res_valid = verify_livecareer()
    assert res_valid[0].status == "PASS"

    # Corrupt row count -> FAIL
    df_invalid = df_valid.iloc[:100]
    df_invalid.to_parquet(lc_dir / "train.parquet", index=False)
    res_invalid = verify_livecareer()
    assert res_invalid[0].status == "FAIL"


def test_verify_ner_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test verification logic on fake NER synthetic dataset."""
    fake_raw = tmp_path / "data" / "raw"
    monkeypatch.setattr("scripts.verify_data.DATA_RAW", fake_raw)
    ner_dir = fake_raw / "ner_synthetic"
    ner_dir.mkdir(parents=True)

    for split, count in [("train.parquet", 9600), ("validation.parquet", 1200), ("test.parquet", 1200)]:
        df = pd.DataFrame({
            "id": list(range(count)),
            "text": ["dummy"] * count,
            "bio_labels": [["O"]] * count,
            "tokens": [["token"]] * count,
        })
        df.to_parquet(ner_dir / split, index=False)

    results = verify_ner_synthetic()
    assert len(results) == 3
    assert all(r.status == "PASS" for r in results)


def test_verify_linkedin_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LinkedIn sample verification rules (row count, min desc length, non-empty title)."""
    fake_raw = tmp_path / "data" / "raw"
    monkeypatch.setattr("scripts.verify_data.DATA_RAW", fake_raw)
    li_dir = fake_raw / "linkedin_sample"
    li_dir.mkdir(parents=True)

    # 5,000 rows with desc >= 200 chars and non-empty titles -> PASS
    df = pd.DataFrame({
        "title": ["Software Engineer"] * 5000,
        "description": ["A" * 250] * 5000,
        "skills_desc": ["Python"] * 5000,
        "location": ["Remote"] * 5000,
        "formatted_experience_level": ["Mid-Senior"] * 5000,
    })
    df.to_parquet(li_dir / "postings_5k.parquet", index=False)

    res = verify_linkedin_sample()
    assert res[0].status == "PASS"

    # Description too short (< 200) -> FAIL
    df_short = df.copy()
    df_short.loc[0, "description"] = "Too short"
    df_short.to_parquet(li_dir / "postings_5k.parquet", index=False)
    res_short = verify_linkedin_sample()
    assert res_short[0].status == "FAIL"


def test_verify_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test manifest verification."""
    manifest_file = tmp_path / "data_manifest.json"
    monkeypatch.setattr("scripts.verify_data.MANIFEST_PATH", manifest_file)

    # Missing -> FAIL
    res_missing = verify_manifest()
    assert res_missing.status == "FAIL"

    # Valid with >= 5 entries -> PASS
    manifest_file.write_text(json.dumps({f"ds_{i}": {} for i in range(5)}))
    res_valid = verify_manifest()
    assert res_valid.status == "PASS"
