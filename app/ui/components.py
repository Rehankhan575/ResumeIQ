"""
app/ui/components.py
====================
Pure presentation components for ResumeIQ dark glassmorphism interface.
All dynamic text is strictly sanitized using html.escape to prevent XSS.
"""

from __future__ import annotations

import html


def render_ambient_background() -> str:
    """Render the fixed ambient glowing blobs and noise overlay."""
    return """
<div class="ambient-canvas" aria-hidden="true">
  <div class="ambient-blob ambient-blob--teal"></div>
  <div class="ambient-blob ambient-blob--indigo"></div>
  <div class="ambient-blob ambient-blob--blue"></div>
</div>
<div class="ambient-noise" aria-hidden="true"></div>
"""


def render_hero(
    brand_name: str = "ResumeIQ",
    subtitle: str = "See how an applicant tracking system (ATS) scores your resume against a job description, and exactly what to fix.",
) -> str:
    """Render the hero brand bar, status indicator, and subtitle."""
    esc_brand = html.escape(brand_name)
    esc_subtitle = html.escape(subtitle)

    return f"""
<div class="hero-wrapper">
  <div class="hero-header-row">
    <div class="brand-badge">
      <div class="brand-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2DD4BF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
      </div>
      <span class="brand-name">{esc_brand}</span>
      <span class="brand-tag">ATS Engine</span>
    </div>
    <div class="hero-status-pill">
      <span class="hero-status-dot"></span>
      <span>System Active</span>
    </div>
  </div>
  <div class="hero-subtitle">{esc_subtitle}</div>
</div>
"""


def render_step_cards() -> str:
    """Render the 3 workflow step cards in a dark glass grid."""
    steps = [
        (
            "1. Choose",
            "Upload a resume and paste a job description, or try a sample.",
        ),
        (
            "2. Analyze",
            "We read the resume, match it to the job and score it out of 100.",
        ),
        (
            "3. Improve",
            "Fix the top issues, then run it again to see your score change.",
        ),
    ]

    cards_html = []
    for title, desc in steps:
        esc_title = html.escape(title)
        esc_desc = html.escape(desc)
        cards_html.append(f"""
  <div class="step-card">
    <div class="step-title">{esc_title}</div>
    <p class="step-desc">{esc_desc}</p>
  </div>""")

    joined_cards = "\n".join(cards_html)
    return f"""<div class="steps-grid">\n{joined_cards}\n</div>"""

