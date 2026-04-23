import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import useVizStore from "../store/useVizStore";

const STEPS = [
    { label: "Input" },
    { label: "Tokenize" },
    { label: "Token Embed" },
    { label: "Pos Embed" },
    { label: "Layers" },
    { label: "Attention" },
    { label: "Pool" },
    { label: "Normalize" },
    { label: "Similarity" },
];

export default function StepStepper() {
    const { currentStep, setCurrentStep, nextStep, prevStep, data } = useVizStore();
    const stepperRef = useRef(null);

    const canNavigate = !!data;

    useEffect(() => {
        const el = stepperRef.current;
        if (!el) return;
        const handler = (e) => {
            if (e.key === "ArrowRight") nextStep();
            if (e.key === "ArrowLeft") prevStep();
        };
        el.addEventListener("keydown", handler);
        return () => el.removeEventListener("keydown", handler);
    }, [nextStep, prevStep]);

    return (
        <div className="stepper-shell" ref={stepperRef} tabIndex={0}>
            {/* Step pills */}
            <div className="stepper-pills">
                {STEPS.map((s, i) => (
                    <button
                        key={i}
                        className={`step-pill ${i === currentStep ? "active" : ""} ${i < currentStep ? "done" : ""}`}
                        onClick={() => canNavigate && setCurrentStep(i)}
                        disabled={!canNavigate && i !== 0}
                        aria-label={`Step ${i + 1}: ${s.label}`}
                        title={s.label}
                    >
                        <span className="step-number">{i + 1}</span>
                        <span className="step-label">{s.label}</span>
                    </button>
                ))}
            </div>

            {/* Progress bar */}
            <div className="stepper-progress-track">
                <motion.div
                    className="stepper-progress-fill"
                    animate={{ width: `${(currentStep / (STEPS.length - 1)) * 100}%` }}
                    transition={{ type: "spring", stiffness: 200, damping: 30 }}
                />
            </div>

            {/* Navigation buttons */}
            <div className="stepper-nav">
                <button className="nav-btn" onClick={prevStep} disabled={currentStep === 0}>
                    ← Back
                </button>
                <span className="step-counter">
                    {currentStep + 1} / {STEPS.length}
                </span>
                <button
                    className="nav-btn primary"
                    onClick={nextStep}
                    disabled={currentStep === STEPS.length - 1 || (!canNavigate && currentStep === 0)}
                >
                    Next →
                </button>
            </div>
        </div>
    );
}
