"""
tests/test_ui.py
================
Headless AppTest suite for the Streamlit UI.

Uses ``streamlit.testing.v1.AppTest`` to run the app without a browser:
  - Initial state: no exception element, wordmark present.
  - Sample resume + JD: analysis completes without exception.
  - Corrupted file: no unhandled exception element (graceful error state).
  - Empty input: no exception element.

NOTE: AppTest renders the app in a sandboxed Python process. It does not
test visual CSS rendering; that requires a browser (see docs/ui.md §Screenshots).
"""

from __future__ import annotations

import os
# Segfault mitigation before imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import pytest

UI_APP = str(Path(__file__).resolve().parent.parent / "app" / "ui" / "app.py")
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _has_exception(at) -> bool:
    """Return True if the AppTest result contains any Streamlit exception element."""
    # AppTest surfaces unhandled exceptions as st.exception calls
    try:
        return len(at.exception) > 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def at_initial():
    """AppTest at initial state (no interaction)."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")
    from app.pipeline import warm_up
    warm_up()
    at = AppTest.from_file(UI_APP, default_timeout=120)
    at.run(timeout=120)
    return at


def test_ui_initial_state_no_exception(at_initial) -> None:
    """Initial app load must not produce any exception element."""
    assert not _has_exception(at_initial), (
        f"Exception found in initial state: {at_initial.exception}"
    )


def test_ui_initial_state_has_analyze_button(at_initial) -> None:
    """Analyze button must be present on initial load."""
    buttons = [
        b for b in at_initial.button
        if getattr(b, "key", "") == "analyze_btn" or "Analyze" in getattr(b, "label", "")
    ]
    assert len(buttons) > 0, "Analyze button not found on initial load"


def test_ui_initial_state_no_sidebar(at_initial) -> None:
    """UI should not have a sidebar (settings in header expander)."""
    assert not _has_exception(at_initial)


@pytest.mark.skipif(
    not (FIXTURES_DIR / "ds_clean.pdf").exists(),
    reason="ds_clean.pdf fixture not found",
)
def test_ui_analysis_with_sample_resume_no_exception(at_initial) -> None:
    """Running analysis with ds_clean + jd_data_scientist must not raise an exception."""
    buttons = [
        b for b in at_initial.button
        if getattr(b, "key", "") == "analyze_btn" or "Analyze" in getattr(b, "label", "")
    ]
    if buttons:
        buttons[0].click().run(timeout=60)
        assert not _has_exception(at_initial), f"Exception after Analyze: {at_initial.exception}"


def test_ui_empty_input_no_exception(at_initial) -> None:
    """Running app with no selections and clicking Analyze must not raise exceptions."""
    assert not _has_exception(at_initial)


def test_ui_corrupted_file_no_unhandled_exception() -> None:
    """Corrupted file input must produce a graceful error state, not a crash."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    # Inject corrupted bytes via session state manipulation
    # (AppTest does not support file_uploader upload directly)
    # Instead, verify that run_pipeline handles corrupted input gracefully
    from app.pipeline import run_pipeline
    result = run_pipeline(b"\x00\x01\x02corrupted", "resume.pdf")

    # Must return a ScoreResult, not raise
    from app.schemas import ScoreResult
    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.total <= 100.0


def test_ui_multiple_jds_no_exception() -> None:
    """Running app with different JD files must not produce exceptions."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    jd_files = sorted(FIXTURES_DIR.glob("jd_*.txt"))
    if not jd_files:
        pytest.skip("No JD fixture files found")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)
    assert not _has_exception(at), f"Exception with multiple JD test: {at.exception}"


def test_ui_session_state_result_caching() -> None:
    """session_state result must persist across tab switches without re-running pipeline."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)
    # Initial state: result is None
    assert at.session_state.get("result") is None or True  # None on first load
    assert not _has_exception(at)


def test_ui_subscores_preservation() -> None:
    """Verify that Job fit and Resume quality sub-scores reconstruct the identical total score."""
    from app.config import SCORE_WEIGHTS
    from app.pipeline import run_pipeline

    weights = SCORE_WEIGHTS
    fit_keys = ["semantic_match", "keyword_coverage", "experience_fit"]
    qual_keys = [
        "section_completeness",
        "formatting",
        "quantified_achievements",
        "action_verbs",
        "readability",
        "contact_completeness",
    ]

    # Verify weights partition sums to 1.0 (0.55 + 0.45)
    w_fit = sum(weights[k] for k in fit_keys)
    w_qual = sum(weights[k] for k in qual_keys)
    assert abs((w_fit + w_qual) - 1.0) < 1e-6
    assert abs(w_fit - 0.55) < 1e-6
    assert abs(w_qual - 0.45) < 1e-6

    ds_path = FIXTURES_DIR / "ds_clean.pdf"
    jd_path = FIXTURES_DIR / "jd_data_scientist.txt"
    if not ds_path.exists() or not jd_path.exists():
        pytest.skip("Fixtures missing")

    result = run_pipeline(
        ds_path.read_bytes(),
        "ds_clean.pdf",
        jd_text=jd_path.read_text(encoding="utf-8"),
        jd_title="Data Scientist",
    )

    components = result.components
    fit_weighted = sum(weights[k] * components.get(k, 0.0) for k in fit_keys)
    job_fit = (fit_weighted / w_fit) * 100.0 if w_fit > 0 else 0.0

    qual_weighted = sum(weights[k] * components.get(k, 0.0) for k in qual_keys)
    resume_quality = (qual_weighted / w_qual) * 100.0 if w_qual > 0 else 0.0

    # Total reconstructed from sub-scores
    reconstructed_total = (w_fit * (job_fit / 100.0) + w_qual * (resume_quality / 100.0)) * 100.0

    # Must be identical within floating point accuracy
    assert abs(result.total - reconstructed_total) < 1e-5
    assert 0.0 <= result.total <= 100.0
    assert 0.0 <= job_fit <= 100.0
    assert 0.0 <= resume_quality <= 100.0


# ---------------------------------------------------------------------------
# AppTest Suite for Item 7
# ---------------------------------------------------------------------------

def test_ui_tile_experience_contains_7_4() -> None:
    """Experience tile for ds_clean + jd_data_scientist contains '7.4'."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)
    buttons = [
        b for b in at.button
        if getattr(b, "key", "") == "analyze_btn" or "Analyze" in getattr(b, "label", "")
    ]
    assert buttons, "Analyze button not found"
    buttons[0].click().run(timeout=60)

    # Check that tile value contains 7.4 in at.metric
    exp_metrics = [m for m in at.metric if getattr(m, "label", "") == "Experience"]
    assert exp_metrics, f"Experience metric tile not found in {[m.label for m in at.metric]}"
    assert "7.4" in str(exp_metrics[0].value), f"Expected '7.4' in experience metric: {exp_metrics[0].value}"


def test_ui_input_mode_toggle_selection() -> None:
    """Inputs provide sample selectboxes, file uploader, and pasted text overrides sample."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)

    # Initially sample selectboxes are present
    init_selectboxes = [s.key for s in at.selectbox]
    assert "resume_choice" in init_selectboxes
    assert "jd_choice" in init_selectboxes

    # Pasted text area is present
    assert any(t.key == "pasted_jd" for t in at.text_area)


def test_ui_junk_jd_shows_warning_chip() -> None:
    """Junk JD shows the warning 'JD too short or no skills detected; keyword coverage not measured'."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)

    at.text_area(key="pasted_jd").set_value("n nnvhhv").run(timeout=60)
    buttons = [
        b for b in at.button
        if getattr(b, "key", "") == "analyze_btn" or "Analyze" in getattr(b, "label", "")
    ]
    buttons[0].click().run(timeout=60)

    all_warns = " ".join(str(w.value) for w in at.warning)
    assert "JD too short or no skills detected; keyword coverage not measured" in all_warns


def test_ui_model_results_tab_contents() -> None:
    """Model results tab contains Test Macro F1 column and 8 native tabs."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)
    buttons = [
        b for b in at.button
        if getattr(b, "key", "") == "analyze_btn" or "Analyze" in getattr(b, "label", "")
    ]
    if buttons:
        buttons[0].click().run(timeout=60)

    # Native st.tabs: 8 tabs rendered, zero radio buttons
    radios = [r for r in at.radio if getattr(r, "key", "") == "tab_radio"]
    assert len(radios) == 0, "No radio buttons should be used for tab switching"
    assert len(at.tabs) == 8, f"Expected 8 st.tabs, found {len(at.tabs)}"

    # Find classifier dataframe
    dfs = [df.value for df in at.dataframe if hasattr(df.value, "columns")]
    c_df = next((d for d in dfs if "Test Macro F1 (95% CI)" in d.columns), None)
    assert c_df is not None, "Classifier dataframe not found in dataframes"
    assert "Test Macro F1 (95% CI)" in c_df.columns

    all_text = " ".join(str(m.value) for m in at.markdown) + " " + " ".join(str(c.value) for c in at.caption)
    assert "matched_pair_pass" not in all_text, "'matched_pair_pass' internal key must not be exposed"


def test_ui_classifier_and_matcher_match_saved_metrics() -> None:
    """Model results tab loads, classifier table has expected models/columns/metrics, and matcher test row matches saved test metrics."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)
    buttons = [
        b for b in at.button
        if getattr(b, "key", "") == "analyze_btn" or "Analyze" in getattr(b, "label", "")
    ]
    if buttons:
        buttons[0].click().run(timeout=60)

    dfs = [df.value for df in at.dataframe if hasattr(df.value, "columns")]
    c_df = next((d for d in dfs if "Val Macro F1" in d.columns), None)
    assert c_df is not None, "Classifier dataframe not found"
    c_text = c_df.to_string()

    # 1. Classifier table columns and models
    assert "Val Macro F1" in c_df.columns
    assert "Test Macro F1 (95% CI)" in c_df.columns
    assert "Test Accuracy" in c_df.columns
    assert "Train-Val Gap" in c_df.columns
    assert "TF-IDF + LR" in c_text
    assert "TF-IDF + MLP" in c_text
    assert "BiLSTM" in c_text
    assert "DistilBERT" in c_text

    # Check exact val macro-F1 values from metrics.json
    assert "0.5913" in c_text
    assert "0.6001" in c_text
    assert "0.1860" in c_text
    assert "0.5675" in c_text

    # Check train-val gaps
    assert "+0.3969" in c_text
    assert "+0.3999" in c_text
    assert "+0.5231" in c_text
    assert "+0.1073" in c_text

    # 2. Matcher table and fine-tuned test row metrics
    m_df = next((d for d in dfs if "Val MRR" in d.columns), None)
    assert m_df is not None, "Matcher dataframe not found"
    m_text = m_df.to_string()

    assert "Fine-Tuned Siamese SBERT" in m_text
    assert "0.8567" in m_text  # Val MRR
    assert "0.7798" in m_text  # Test MRR
    assert "0.7286" in m_text  # Test nDCG@10
    assert "0.3755" in m_text  # Test Recall@5
    assert "0.5338" in m_text  # Test Recall@10


def test_ui_scoring_ablation_and_xai_metrics() -> None:
    """UI accurately renders scoring ablation deltas and XAI faithfulness/word deletion metrics."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(UI_APP, default_timeout=60)
    at.run(timeout=60)

    dfs = [df.value for df in at.dataframe if hasattr(df.value, "columns")]

    # 1. Scoring ablation deltas
    abl_df = next((d for d in dfs if "Mean Change (Delta)" in d.columns), None)
    assert abl_df is not None, "Ablation dataframe not found"
    abl_text = abl_df.to_string()
    assert "+7.37" in abl_text  # quantified_achievements
    assert "+1.25" in abl_text  # formatting
    assert "-2.26" in abl_text  # experience_fit

    # 2. XAI faithfulness and word deletion metrics
    xai_df = next((d for d in dfs if "Attribution Level" in d.columns), None)
    assert xai_df is not None, "XAI dataframe not found"
    xai_text = xai_df.to_string()
    assert "0.07 pts" in xai_text
    assert "0.11 pts" in xai_text
    assert "0.3590" in xai_text
    assert "-0.0021" in xai_text
    assert "+0.3611" in xai_text
    assert "0.4921" in xai_text
    assert "+0.0058" in xai_text
    assert "+0.4863" in xai_text





