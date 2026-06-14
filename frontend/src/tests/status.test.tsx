import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import StatusPage from "../app/status/page";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// Mock next/link
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    className,
  }: {
    children: React.ReactNode;
    href: string;
    className?: string;
  }) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}));

// Set env vars at module scope
process.env.NEXT_PUBLIC_API_URL = "http://test-api:8000";

function createMockResponse(data: Record<string, unknown>, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(data),
  };
}

describe("StatusPage — HealthState union", () => {
  afterEach(() => {
    mockFetch.mockReset();
  });

  it("renders loading state on mount with Checking text", async () => {
    mockFetch.mockImplementationOnce(
      () =>
        new Promise(() => {
          /* never resolves */
        })
    );
    render(<StatusPage />);
    const checking = screen.getAllByText("Checking...");
    expect(checking.length).toBeGreaterThanOrEqual(1);
  });

  it("transitions to ok state with mocked latency", async () => {
    mockFetch.mockImplementationOnce(async () => {
      await new Promise((r) => setTimeout(r, 50));
      return createMockResponse({
        status: "healthy",
        model_loaded: true,
        model_path: "models/churn_model.pkl",
      });
    });

    render(<StatusPage />);

    await waitFor(() => {
      expect(screen.getByText("Connected")).not.toBeNull();
    });

    expect(screen.getByText(/Status: healthy/)).not.toBeNull();
    expect(screen.getByText("Loaded")).not.toBeNull();
  });

  it("transitions to error state on HTTP failure", async () => {
    mockFetch.mockImplementationOnce(async () => {
      return createMockResponse({ detail: "Not found" }, 404);
    });

    render(<StatusPage />);

    await waitFor(() => {
      expect(screen.getByText("Offline")).not.toBeNull();
    });

    expect(screen.getByText(/HTTP 404/)).not.toBeNull();
  });

  it("transitions to error state on network failure", async () => {
    mockFetch.mockImplementationOnce(async () => {
      throw new Error("Network error");
    });

    render(<StatusPage />);

    await waitFor(() => {
      expect(screen.getByText("Offline")).not.toBeNull();
    });

    expect(screen.getByText(/Network error/)).not.toBeNull();
  });

  it("shows model offline when model_loaded is false", async () => {
    mockFetch.mockImplementationOnce(async () => {
      return createMockResponse({
        status: "healthy",
        model_loaded: false,
        model_path: "unknown",
      });
    });

    render(<StatusPage />);

    await waitFor(() => {
      const offline = screen.getAllByText("Offline");
      expect(offline.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("displays build timestamp fallback in test", () => {
    mockFetch.mockImplementationOnce(
      () =>
        new Promise(() => {
          /* never resolves */
        })
    );
    render(<StatusPage />);
    // Module-level BUILD_TIME constant evaluates to "development" in test env
    expect(screen.getByText("development")).not.toBeNull();
  });

  it("polls /health every 15s and updates state", async () => {
    let callCount = 0;

    mockFetch.mockImplementation(async () => {
      callCount++;
      return createMockResponse({
        status: "healthy",
        model_loaded: true,
        model_path: "models/churn_model.pkl",
      });
    });

    render(<StatusPage />);

    await waitFor(() => expect(callCount).toBe(1));

    const beforeSecondCall = callCount;
    await waitFor(
      () => expect(callCount).toBeGreaterThan(beforeSecondCall),
      { timeout: 20000 }
    );
  }, 30000);

  it("does not update state after unmount", async () => {
    let resolvePromise: (_v: unknown) => void = () => {};
    const fetchPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    mockFetch.mockImplementationOnce(async () => {
      return fetchPromise.then(() =>
        createMockResponse({
          status: "healthy",
          model_loaded: true,
          model_path: "models/churn_model.pkl",
        })
      );
    });

    const { unmount } = render(<StatusPage />);
    unmount();

    await act(async () => {
      resolvePromise(true);
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("shows request timed out message on AbortError", async () => {
    const abortError = new Error("The operation was aborted");
    abortError.name = "AbortError";
    mockFetch.mockImplementationOnce(async () => {
      throw abortError;
    });

    render(<StatusPage />);

    await waitFor(() => {
      expect(screen.getByText("Offline")).not.toBeNull();
    });

    expect(screen.getByText(/timed out/)).not.toBeNull();
  });
});
