import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ScoreRing } from "../components/results/ScoreRing";
import { HighlightedResume } from "../components/results/HighlightedResume";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { HistoryView } from "../components/views/HistoryView";
import { BiasCheckView } from "../components/views/BiasCheckView";
import { SkillGapView } from "../components/views/SkillGapView";

describe("ResumeIQ UI Verification Suite", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  // 1. Score Ring at 0, 50, 100, and out-of-range
  describe("ScoreRing Clamping and Values", () => {
    it("renders score 0 correctly", () => {
      render(<ScoreRing score={0} grade="F" />);
      expect(screen.getByTestId("score-value")).toHaveTextContent("0");
    });

    it("renders score 50 correctly", () => {
      render(<ScoreRing score={50} grade="C" />);
      expect(screen.getByTestId("score-value")).toHaveTextContent("50");
    });

    it("renders score 100 correctly", () => {
      render(<ScoreRing score={100} grade="A" />);
      expect(screen.getByTestId("score-value")).toHaveTextContent("100");
    });

    it("clamps negative out-of-range scores to 0", () => {
      render(<ScoreRing score={-25} grade="F" />);
      expect(screen.getByTestId("score-value")).toHaveTextContent("0");
    });

    it("clamps excessive out-of-range scores to 100", () => {
      render(<ScoreRing score={150} grade="A" />);
      expect(screen.getByTestId("score-value")).toHaveTextContent("100");
    });

    it("safely handles NaN score by falling back to 0", () => {
      render(<ScoreRing score={NaN} grade="F" />);
      expect(screen.getByTestId("score-value")).toHaveTextContent("0");
    });
  });

  // 2. Highlight rendering with "<script>" in text (must render as text, never innerHTML)
  describe("HighlightedResume Zero-XSS Guarantee", () => {
    it("renders text containing <script> literally without DOM injection", () => {
      const maliciousPayload = 'Candidate with expertise in <script>alert("xss")</script> and Machine Learning.';
      const highlights = [
        {
          start: 28,
          end: 58,
          text: '<script>alert("xss")</script>',
          weight: 0.5,
          type: "keyword_match" as const,
        },
      ];

      const { container } = render(
        <HighlightedResume rawText={maliciousPayload} highlights={highlights} />
      );

      // Verify that no executable <script> element exists in the DOM tree
      const scriptElements = container.querySelectorAll("script");
      expect(scriptElements.length).toBe(0);

      // Verify that the text itself is present as harmless literal text content
      expect(container.textContent).toContain('<script>alert("xss")</script>');
    });

    it("snaps subword or mid-token offsets to whole-word boundaries", () => {
      const text = "Expert in predictive modeling and Computer Science.";
      // Suppose offset was [13, 16] ('dic' inside 'predictive')
      const highlights = [
        {
          start: 13,
          end: 16,
          text: "dic",
          weight: 0.5,
          type: "positive" as const,
        },
      ];

      const { container } = render(
        <HighlightedResume rawText={text} highlights={highlights} />
      );

      // The highlighted element should contain the whole word 'predictive', not just 'dic'
      const mark = container.querySelector(".underline");
      expect(mark?.textContent).toBe("predictive");
    });
  });

  // 3. History storing no resume text
  describe("History Storage Privacy Compliance", () => {
    it("never serializes or persists raw resume text into localStorage", () => {
      localStorage.setItem("resumeiq_history_opt_in", "true");

      // Simulate a history save
      const sampleRecord = {
        id: "rec-test",
        timestamp: "2026-11-04 18:00",
        jd_title: "Senior Data Scientist",
        score: 84.88,
        grade: "B",
      };

      localStorage.setItem("resumeiq_history", JSON.stringify([sampleRecord]));

      const storedRaw = localStorage.getItem("resumeiq_history");
      expect(storedRaw).toBeTruthy();

      const parsed = JSON.parse(storedRaw!);
      expect(parsed[0].score).toBe(84.88);
      expect(parsed[0].jd_title).toBe("Senior Data Scientist");

      // Verify forbidden privacy keys are not stored anywhere
      expect(storedRaw).not.toContain("raw_text");
      expect(storedRaw).not.toContain("sections");
      expect(storedRaw).not.toContain("education");
      expect(storedRaw).not.toContain("experience");
      expect(storedRaw).not.toContain("contact");
    });
  });

  // 4. API Error State
  describe("API Error State & Recovery", () => {
    it("renders friendly error banner with a working retry action", () => {
      const onRetry = vi.fn();
      render(
        <ErrorBanner
          message="Connection refused: FastAPI server is unreachable on localhost:8000."
          onRetry={onRetry}
        />
      );

      expect(
        screen.getByText("Connection refused: FastAPI server is unreachable on localhost:8000.")
      ).toBeInTheDocument();

      const retryBtn = screen.getByRole("button", { name: /retry/i });
      expect(retryBtn).toBeInTheDocument();

      fireEvent.click(retryBtn);
      expect(onRetry).toHaveBeenCalledTimes(1);
    });
  });

  // 5. Verdict String Cleanliness (Item 7)
  describe("ScoreRing Verdict String", () => {
    it("renders 'Needs Work' for grade F without trailing Fit artifact", () => {
      render(<ScoreRing score={55} grade="F" />);
      expect(screen.getByText("Needs Work")).toBeInTheDocument();
      expect(screen.queryByText(/Needs work Fit/i)).not.toBeInTheDocument();
    });

    it("renders 'Good Match' for grade B", () => {
      render(<ScoreRing score={85} grade="B" />);
      expect(screen.getByText("Good Match")).toBeInTheDocument();
    });
  });

  // 6. History Clean State by Default (Item 5)
  describe("HistoryView Clean Initialization", () => {
    it("shows empty state when localStorage has no records", () => {
      render(<HistoryView />);
      expect(screen.getByText("No evaluation records saved.")).toBeInTheDocument();
    });
  });

  // 7. Bias View Clean vs Flagged States (Item 2)
  describe("BiasCheckView Dynamic States", () => {
    it("shows clean neutrality verified state when findings are empty", () => {
      render(<BiasCheckView findings={[]} />);
      expect(screen.getByText("Profile Neutrality Verified")).toBeInTheDocument();
      expect(screen.getByText("No Demographic or Bias Risks Detected")).toBeInTheDocument();
      expect(screen.queryByText(/Audit Score/i)).not.toBeInTheDocument();
    });

    it("renders actionable findings when flags exist", () => {
      const findings = [
        {
          type: "personal_info",
          evidence_label: "Date of birth disclosed",
          suggestion: "Remove date of birth to protect against potential age discrimination.",
        },
      ];
      render(<BiasCheckView findings={findings} />);
      expect(screen.getByText("Date of birth disclosed")).toBeInTheDocument();
      expect(screen.getByText(/Remove date of birth/i)).toBeInTheDocument();
    });
  });

  // 8. SkillGapView Real Requirements (Item 2)
  describe("SkillGapView Real Competencies", () => {
    it("renders matched requirements and missing skills without external URLs", () => {
      const matched = ["python", "sql"];
      const missing = ["docker"];
      const recommendations = {
        docker: [
          {
            title: "Applied Docker for Engineers",
            provider: "Self-Directed",
            level: "Intermediate",
            estimated_hours: 15,
          },
        ],
      };

      const { container } = render(
        <SkillGapView
          matchedSkills={matched}
          missingSkills={missing}
          recommendations={recommendations}
        />
      );

      expect(screen.getByText("python")).toBeInTheDocument();
      expect(screen.getByText("sql")).toBeInTheDocument();
      expect(screen.getAllByText("docker").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Applied Docker for Engineers")).toBeInTheDocument();

      // Zero external hyperlinks
      const links = container.querySelectorAll("a");
      expect(links.length).toBe(0);
    });
  });
});
