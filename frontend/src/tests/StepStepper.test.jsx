import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import StepStepper from "../components/StepStepper";

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
        for (let i = 1; i <= 9; i++) {
            expect(screen.getByLabelText(new RegExp(`step ${i}`, "i"))).toBeInTheDocument();
        }
    });

    it("clicking Next calls nextStep", () => {
        mockStore.nextStep.mockClear();
        render(<StepStepper />);
        fireEvent.click(screen.getByText(/next/i));
        expect(mockStore.nextStep).toHaveBeenCalledTimes(1);
    });

    it("Back button is disabled at step 0", () => {
        render(<StepStepper />);
        expect(screen.getByText(/back/i)).toBeDisabled();
    });

    it("pressing ArrowRight on stepper shell calls nextStep", () => {
        mockStore.nextStep.mockClear();
        const { container } = render(<StepStepper />);
        const stepperEl = container.querySelector(".stepper-shell");
        expect(stepperEl).not.toBeNull();
        fireEvent.keyDown(stepperEl, { key: "ArrowRight" });
        expect(mockStore.nextStep).toHaveBeenCalledTimes(1);
    });

    it("step counter shows '1 / 9' at step 0", () => {
        render(<StepStepper />);
        expect(screen.getByText("1 / 9")).toBeInTheDocument();
    });
});
