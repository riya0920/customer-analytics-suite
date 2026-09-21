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
    expect(await screen.findByText("4,899")).toBeInTheDocument();
    expect(screen.getByText(/Top-20% value share/i)).toBeInTheDocument();
  });

  it("says the data is real and only the ad journeys are simulated", async () => {
    renderApp();
    await screen.findByText("4,899");
    expect(screen.getByText(/UCI Online Retail II, Dec 2009/)).toBeInTheDocument();
    expect(screen.getByText(/ad journeys are simulated/i)).toBeInTheDocument();
  });

  it("shows money in pounds, never dollars", async () => {
    const { container } = renderApp();
    await screen.findByText("4,899");
    // Value at stake = sum of segment clv_total in the fixture.
    expect(screen.getByText("£2,798,983")).toBeInTheDocument();
    for (const tab of ["Customer Lifetime Value", "Attribution", "Segmentation"]) {
      await userEvent.click(screen.getByRole("tab", { name: tab }));
      expect(container.textContent).not.toMatch(/\$\d/);
    }
  });

  it("defaults to the Segmentation tab and shows a segment row", async () => {
    renderApp();
    await screen.findByText("4,899");
    const seg = screen.getByRole("tab", { name: "Segmentation" });
    expect(seg).toHaveAttribute("aria-selected", "true");
    // Segment table renders the top-value segment.
    expect(screen.getAllByText("Segment 4").length).toBeGreaterThan(0);
  });

  it("names the small wholesale segment from the data, not hard-coded text", async () => {
    const { container } = renderApp();
    await screen.findByText("4,899");
    const text = container.textContent ?? "";
    expect(text).toMatch(/Segment 3 is only 34 customers \(0\.7%\)/);
    expect(text).toMatch(/21\.3% of value/);
  });

  it("explains k=5 by forward separation and does not claim it is most stable", async () => {
    const { container } = renderApp();
    await screen.findByText("4,899");
    const text = container.textContent ?? "";
    expect(text).toMatch(/0\.066 at k=5/);
    expect(text).toMatch(/peaks at 0\.071 at k=7/);
    expect(text).toMatch(/Stability is not the reason for k=5/);
    expect(text).not.toMatch(/stability drops/i);
  });

  it("switches to the Attribution tab and surfaces the zero-effect callout", async () => {
    const { container } = renderApp();
    await screen.findByText("4,899");
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
    // Budget callout is computed: best-MAE and best-budget methods differ.
    const text = container.textContent ?? "";
    expect(text).toMatch(/Shapley gives the best budget, losing only 2\.9%/);
    expect(text).toMatch(/Markov removal does worst: it loses 31\.8%/);
    expect(text).toMatch(/puts £9,123 into retargeting/);
  });

  it("switches to the CLV tab and shows the model comparison", async () => {
    renderApp();
    await screen.findByText("4,899");
    await userEvent.click(
      screen.getByRole("tab", { name: /Customer Lifetime Value/i }),
    );
    // Appears in the "best model" KPI sub-label, the table and the callout.
    const hits = await screen.findAllByText("BG/NBD + Gamma-Gamma");
    expect(hits.length).toBeGreaterThan(0);
    expect(screen.getByText("MAE (£)")).toBeInTheDocument();
  });
});
