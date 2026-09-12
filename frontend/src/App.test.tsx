import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "./App";
import { DataProvider } from "./data/store";
import { makeDataset } from "./test/fixtures";

function renderApp() {
  const data = makeDataset();
  return render(
    <DataProvider loader={() => Promise.resolve(data)}>
      <App />
    </DataProvider>,
  );
}

describe("App", () => {
  it("shows a loading state, then the KPI bar once data resolves", async () => {
    renderApp();
    expect(screen.getByText(/Loading model outputs/i)).toBeInTheDocument();
    // KPI value from the fixture (total_customers formatted with a separator).
    expect(await screen.findByText("8,000")).toBeInTheDocument();
    expect(screen.getByText(/Top-20% value share/i)).toBeInTheDocument();
  });

  it("defaults to the Segmentation tab and shows a segment row", async () => {
    renderApp();
    await screen.findByText("8,000");
    const seg = screen.getByRole("tab", { name: "Segmentation" });
    expect(seg).toHaveAttribute("aria-selected", "true");
    // Segment table renders the top-value segment.
    expect(screen.getAllByText("Segment 4").length).toBeGreaterThan(0);
  });

  it("switches to the Attribution tab and surfaces the zero-effect callout", async () => {
    renderApp();
    await screen.findByText("8,000");
    await userEvent.click(
      screen.getByRole("tab", { name: "Attribution" }),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("tab", { name: "Attribution" }),
      ).toHaveAttribute("aria-selected", "true"),
    );
    // The planted zero-effect channel is named in the incrementality callout.
    expect(screen.getAllByText(/retargeting/i).length).toBeGreaterThan(0);
  });

  it("switches to the CLV tab and shows the model comparison", async () => {
    renderApp();
    await screen.findByText("8,000");
    await userEvent.click(
      screen.getByRole("tab", { name: /Customer Lifetime Value/i }),
    );
    // Appears in both the "best model" KPI sub-label and the comparison table.
    const hits = await screen.findAllByText("BG/NBD + Gamma-Gamma");
    expect(hits.length).toBeGreaterThan(0);
  });

  it("renders an error state when the loader rejects", async () => {
    render(
      <DataProvider loader={() => Promise.reject(new Error("boom"))}>
        <App />
      </DataProvider>,
    );
    expect(await screen.findByText(/Could not load the data: boom/i)).toBeInTheDocument();
  });
});
