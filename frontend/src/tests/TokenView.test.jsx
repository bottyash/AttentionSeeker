import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TokenView from "../components/TokenView";

// Stub InfoCard so framer-motion doesn't complicate the test environment
vi.mock("../components/InfoCard", () => ({
    default: () => null,
}));

// Top-level store mock
vi.mock("../store/useVizStore", () => ({
    default: () => ({
        data: {
            data: {
                tokens: ["[CLS]", "hello", "world", "[SEP]", "extra"],
                input_ids: [101, 7592, 2088, 102, 4805],
                token_embeds: Array(5).fill(Array(384).fill(0.1)),
            },
        },
    }),
}));

describe("TokenView", () => {
    it("renders the correct number of token pills", () => {
        render(<TokenView />);
        // getAllByText handles cases where token text appears in multiple DOM nodes
        expect(screen.getAllByText("[CLS]").length).toBeGreaterThan(0);
        expect(screen.getAllByText("hello").length).toBeGreaterThan(0);
        expect(screen.getAllByText("world").length).toBeGreaterThan(0);
        expect(screen.getAllByText("[SEP]").length).toBeGreaterThan(0);
        expect(screen.getAllByText("extra").length).toBeGreaterThan(0);
    });

    it("shows token count label", () => {
        render(<TokenView />);
        expect(screen.getByText(/5 tokens/i)).toBeInTheDocument();
    });
});
