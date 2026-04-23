import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import StepStepper from "../components/StepStepper";

// Provide a store with data present (so navigation is enabled)
const mockStore = {
    currentStep: 0,
    setCurrentStep: vi.fn(),
    nextStep: vi.fn(),
    prevStep: vi.fn(),
    data: { data: {} }, // truthy — enables navigation
};

vi.mock("../store/useVizStore", () => ({
    default: () => mockStore,
}));

describe("StepStepper", () => {
    it("renders 9 step pills", () => {
        render(<StepStepper />);
        // Step numbers 1–9 should all be visible inside pills
        for (let i = 1; i <= 9; i++) {
            expect(screen.getByLabelText(new RegExp(`step ${i}`, "i"))).toBeInTheDocument();
        }
    });

    it("clicking Next calls nextStep", () => {
        render(<StepStepper />);
        fireEvent.click(screen.getByText(/next/i));
        expect(mockStore.nextStep).toHaveBeenCalledTimes(1);
    });

    it("Back button is disabled at step 0", () => {
        render(<StepStepper />);
        const backBtn = screen.getByText(/back/i);
        expect(backBtn).toBeDisabled();
    });

    it("pressing ArrowRight calls nextStep", () => {
        render(<StepStepper />);
        const stepper = screen.getByRole("group") ?? document.querySelector(".stepper-shell");
        // Fire keydown on the stepper container
        fireEvent.keyDown(document.querySelector(".stepper-shell"), { key: "ArrowRight" });
        // nextStep was already called once in the click test; total should now be 2
        expect(mockStore.nextStep).toHaveBeenCalledTimes(2);
    });

    it("step counter shows '1 / 9' at step 0", () => {
        render(<StepStepper />);
        expect(screen.getByText("1 / 9")).toBeInTheDocument();
    });
});
