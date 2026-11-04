"""
app/ui/theme.py
===============
Design tokens, ambient background keyframes, and CSS styling for the
ResumeIQ dark glassmorphism SaaS interface.

Reference styles:
- Obsidian background (#07080B)
- Frosted glass cards with multi-directional specular borders
- Soft radial color blobs (teal, indigo, sky blue) with GPU-accelerated drift
- Inner bevel highlights and deep elevation shadows
- High-contrast typography and WCAG AAA/AA compliance
"""

from __future__ import annotations

# Design Tokens
CANVAS_BG = "#07080B"
SURFACE_GLASS_BG = "linear-gradient(145deg, rgba(255, 255, 255, 0.055) 0%, rgba(255, 255, 255, 0.015) 100%)"
SURFACE_CARD_BG = "rgba(15, 17, 23, 0.7)"
BORDER_GLASS = "1px solid rgba(255, 255, 255, 0.12)"
BORDER_GLASS_LIGHT = "1px solid rgba(255, 255, 255, 0.2)"
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"
ACCENT_TEAL = "#2DD4BF"
ACCENT_INDIGO = "#6366F1"
ACCENT_SKY = "#38BDF8"

DARK_GLASS_CSS = """
/* ==========================================================================
   ResumeIQ Dark Glassmorphism Design System
   ========================================================================== */

/* Root & Base */
html, body {
  background-color: #07080B !important;
  color: #F8FAFC !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
  font-size: 16px !important;
  margin: 0 !important;
  padding: 0 !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

[data-testid="stAppViewContainer"] {
  background-color: #07080B !important;
  color: #F8FAFC !important;
  overflow-x: hidden !important;
}

[data-testid="stMain"] {
  background: transparent !important;
}

/* Page container */
[data-testid="block-container"] {
  max-width: 1140px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 3.5rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
  position: relative !important;
  z-index: 1 !important;
}

/* Hide header toolbar, footer, and main menu */
#MainMenu, footer, [data-testid="stHeader"] {
  display: none !important;
}

/* Base typography */
p, span, label, div {
  color: #F8FAFC;
}

/* Strict font floor >= 14px for all UI text nodes */
small, [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span,
[data-testid="stMetricLabel"] p, [data-testid="stMetricDelta"] span {
  font-size: 14px !important;
  color: #94A3B8 !important;
}

/* Card titles */
.card-title {
  font-size: 18px !important;
  font-weight: 600 !important;
  color: #F8FAFC !important;
  margin-bottom: 0.5rem !important;
  letter-spacing: -0.01em !important;
}

/* Ambient Background Blobs & Noise */
.ambient-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}

.ambient-blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(120px);
  will-change: transform, opacity;
  pointer-events: none;
}

.ambient-blob--teal {
  top: -100px;
  left: 4%;
  width: 660px;
  height: 660px;
  background: radial-gradient(circle, rgba(45, 212, 191, 0.28) 0%, rgba(45, 212, 191, 0) 70%);
  filter: blur(100px);
  animation: driftTeal 22s ease-in-out infinite alternate;
}

.ambient-blob--indigo {
  top: 30px;
  right: 4%;
  width: 700px;
  height: 700px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.26) 0%, rgba(99, 102, 241, 0) 70%);
  filter: blur(110px);
  animation: driftIndigo 26s ease-in-out infinite alternate;
}

.ambient-blob--blue {
  top: 420px;
  left: 25%;
  width: 620px;
  height: 620px;
  background: radial-gradient(circle, rgba(56, 189, 248, 0.22) 0%, rgba(56, 189, 248, 0) 70%);
  filter: blur(110px);
  animation: driftBlue 24s ease-in-out infinite alternate;
}

.ambient-noise {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none;
  z-index: 0;
  opacity: 0.035;
  mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
}

/* Keyframe animations: transform and opacity ONLY */
@keyframes driftTeal {
  0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.15; }
  50% { transform: translate3d(40px, 30px, 0) scale(1.08); opacity: 0.20; }
  100% { transform: translate3d(-30px, 50px, 0) scale(0.96); opacity: 0.16; }
}

@keyframes driftIndigo {
  0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.14; }
  50% { transform: translate3d(-50px, 40px, 0) scale(1.1); opacity: 0.19; }
  100% { transform: translate3d(30px, -30px, 0) scale(0.95); opacity: 0.15; }
}

@keyframes driftBlue {
  0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.11; }
  50% { transform: translate3d(35px, -40px, 0) scale(1.09); opacity: 0.16; }
  100% { transform: translate3d(-40px, 25px, 0) scale(0.97); opacity: 0.12; }
}

/* Reduced motion accessibility */
@media (prefers-reduced-motion: reduce) {
  .ambient-blob {
    animation: none !important;
  }
  .glass, .glass-step, .step-card {
    transition: none !important;
  }
}

/* ==========================================================================
   Glass Classes
   ========================================================================== */

.glass {
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.055) 0%, rgba(255, 255, 255, 0.015) 100%);
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.11);
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.20), 0 24px 48px -12px rgba(0, 0, 0, 0.72);
  border-radius: 20px;
  position: relative;
  overflow: hidden;
}

/* Top-left soft glow for glass surfaces (Image B style) */
.glass::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 160px;
  height: 160px;
  background: radial-gradient(circle at 0% 0%, rgba(45, 212, 191, 0.16), transparent 70%);
  pointer-events: none;
  border-radius: 20px 0 0 0;
}

/* Glass fallback for browsers without backdrop-filter support */
@supports not (backdrop-filter: blur(1px)) {
  .glass, .step-card, [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 17, 23, 0.94) !important;
  }
}

/* Streamlit Container border=True override */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.055) 0%, rgba(255, 255, 255, 0.015) 100%) !important;
  backdrop-filter: blur(24px) saturate(160%) !important;
  -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
  border: 1px solid rgba(255, 255, 255, 0.13) !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.28), 0 20px 44px -10px rgba(0, 0, 0, 0.75) !important;
  padding: 1.15rem !important;
  position: relative !important;
}

/* Hero Section */
.hero-wrapper {
  margin-bottom: 1.25rem;
}

.hero-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.brand-badge {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.brand-icon {
  width: 32px;
  height: 32px;
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(45, 212, 191, 0.3) 0%, rgba(99, 102, 241, 0.3) 100%);
  border: 1px solid rgba(255, 255, 255, 0.25);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 14px rgba(45, 212, 191, 0.25);
}

.brand-name {
  font-size: 24px;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #FFFFFF;
}

.brand-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.14);
  font-size: 14px;
  font-weight: 500;
  color: #94A3B8;
}

.hero-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border-radius: 9999px;
  background: rgba(45, 212, 191, 0.09);
  border: 1px solid rgba(45, 212, 191, 0.32);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15), 0 0 12px -2px rgba(45, 212, 191, 0.2);
  font-size: 14px;
  font-weight: 600;
  color: #2DD4BF;
}

.hero-status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #2DD4BF;
  box-shadow: 0 0 8px #2DD4BF;
}

.hero-subtitle {
  font-size: 16px !important;
  line-height: 1.55 !important;
  color: #CBD5E1 !important;
  margin: 0 0 1rem 0 !important;
  max-width: 860px;
}

/* 3 Step Cards Grid */
.steps-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-bottom: 1.25rem;
}

@media (max-width: 768px) {
  .steps-grid {
    grid-template-columns: 1fr;
  }
}

.step-card {
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.055) 0%, rgba(255, 255, 255, 0.015) 100%);
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.13);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.28), 0 16px 36px -10px rgba(0, 0, 0, 0.7);
  transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
  position: relative;
  overflow: hidden;
}

.step-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100px;
  height: 100px;
  background: radial-gradient(circle at 0% 0%, rgba(45, 212, 191, 0.15) 0%, transparent 70%);
  pointer-events: none;
  border-radius: 16px 0 0 0;
}

.step-card:hover {
  transform: translateY(-2px);
  border-color: rgba(45, 212, 191, 0.35);
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.35), 0 20px 40px -10px rgba(0, 0, 0, 0.8), 0 0 20px -6px rgba(45, 212, 191, 0.25);
}

.step-title {
  font-size: 15px;
  font-weight: 700;
  color: #FFFFFF;
  margin-bottom: 0.25rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

.step-desc {
  font-size: 14px;
  line-height: 1.45;
  color: #94A3B8;
  margin: 0;
}

/* Form Inputs, Selectboxes, Dropzones Dark Glass Override */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: rgba(13, 15, 21, 0.75) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  color: #F8FAFC !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
  font-size: 14px !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: #2DD4BF !important;
  box-shadow: 0 0 0 1px #2DD4BF, inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stSelectbox"] > div > div {
  background: rgba(13, 15, 21, 0.75) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  color: #F8FAFC !important;
  border-radius: 12px !important;
  font-size: 14px !important;
}

[data-testid="stFileUploader"] section {
  background: rgba(13, 15, 21, 0.5) !important;
  border: 1px dashed rgba(255, 255, 255, 0.18) !important;
  border-radius: 14px !important;
  padding: 1rem !important;
}

/* Primary Button */
[data-testid="stButton"] button[kind="primary"],
[data-testid="stButton"] button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #2DD4BF 0%, #0D9488 100%) !important;
  color: #07080B !important;
  font-weight: 650 !important;
  font-size: 15px !important;
  border: 1px solid rgba(255, 255, 255, 0.35) !important;
  box-shadow: 0 4px 20px -2px rgba(45, 212, 191, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
  border-radius: 12px !important;
  padding: 0.65rem 1.5rem !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px -2px rgba(45, 212, 191, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
}

/* Info and Alert Banners */
[data-testid="stAlert"] {
  background: rgba(15, 23, 42, 0.75) !important;
  border: 1px solid rgba(56, 189, 248, 0.25) !important;
  border-radius: 14px !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
}

/* Sidebar Dark Glass */
[data-testid="stSidebar"] {
  background-color: #0A0C11 !important;
  border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
"""
