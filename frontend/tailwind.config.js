/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#0A0A0B",
        surface: {
          DEFAULT: "#111113",
          card: "#161618",
          subtle: "#1C1C1F",
        },
        border: {
          DEFAULT: "rgba(255, 255, 255, 0.08)",
          strong: "rgba(255, 255, 255, 0.14)",
          focus: "#D9C7A0",
        },
        content: {
          primary: "#F5F5F4",
          secondary: "#A1A1A6",
          muted: "#6B6B72",
        },
        sand: {
          DEFAULT: "#D9C7A0",
          dim: "rgba(217, 199, 160, 0.12)",
        },
        semantic: {
          good: "#7FB28A",
          "good-dim": "rgba(127, 178, 138, 0.12)",
          warn: "#D4A95A",
          "warn-dim": "rgba(212, 169, 90, 0.12)",
          bad: "#C97B7B",
          "bad-dim": "rgba(201, 123, 123, 0.12)",
        },
      },
      fontFamily: {
        sans: ["'Geist Sans'", "'Geist'", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["'Geist Mono'", "monospace"],
        serif: ["'Instrument Serif'", "serif"],
      },
      letterSpacing: {
        heading: "-0.02em",
      },
      borderRadius: {
        sm: "8px",
        md: "12px",
        lg: "16px",
        xl: "24px",
        full: "9999px",
      },
      boxShadow: {
        glass: "inset 0 1px 0 0 rgba(255, 255, 255, 0.08), 0 24px 48px -12px rgba(0, 0, 0, 0.7)",
      },
    },
  },
  plugins: [],
}

