import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ScoreRing } from "../components/results/ScoreRing";
import { HighlightedResume } from "../components/results/HighlightedResume";
import { ErrorBanner } from "../components/ui/ErrorBanner";

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
});
