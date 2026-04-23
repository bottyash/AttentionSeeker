import { describe, it, expect, vi, beforeAll } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import AttentionHeatmap from "../components/AttentionHeatmap";

// Build a minimal mock attention payload: 2 tokens, 6 layers, 12 heads
const N = 2;
const row = Array(N).fill(1 / N); // uniform attention, sums to 1
const head = Array(12).fill([row, row]);
const mockAttentions = Array(6).fill(head);

vi.mock("../store/useVizStore", () => ({
    default: () => ({
        data: {
            data: {
                tokens: ["[CLS]", "hi"],
                attentions: mockAttentions,
            },
        },
        selectedLayer: 0,
        selectedHead: 0,
        setSelectedLayer: vi.fn(),
        setSelectedHead: vi.fn(),
    }),
}));

describe("AttentionHeatmap", () => {
    it("renders layer selector tabs (6 layers)", () => {
        render(<AttentionHeatmap />);
        // Layer tabs: L1 … L6
        for (let i = 1; i <= 6; i++) {
            expect(screen.getByText(`L${i}`)).toBeInTheDocument();
        }
    });

    it("renders head selector tabs (12 heads)", () => {
        render(<AttentionHeatmap />);
        for (let i = 1; i <= 12; i++) {
            expect(screen.getByText(`H${i}`)).toBeInTheDocument();
        }
    });

    it("renders section title", () => {
        render(<AttentionHeatmap />);
        expect(screen.getByText(/attention weights/i)).toBeInTheDocument();
    });
});
