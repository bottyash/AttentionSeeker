/**
 * InfoCard.jsx — expandable ⓘ explanation card
 *
 * Usage: <InfoCard title="Why does this matter?">Plain-language explanation…</InfoCard>
 * Renders a collapsed ⓘ chip by default; click to expand into a styled card.
 */

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function InfoCard({ title = "What's happening here?", children }) {
    const [open, setOpen] = useState(false);

    return (
        <div className="info-card-wrap">
            <button
                className="info-trigger"
                onClick={() => setOpen(!open)}
                aria-expanded={open}
                aria-label={open ? "Hide explanation" : "Show explanation"}
            >
                <span className="info-icon">ⓘ</span>
                <span className="info-trigger-label">{open ? "Hide" : title}</span>
            </button>

            <AnimatePresence>
                {open && (
                    <motion.div
                        className="info-card"
                        role="note"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.22, ease: "easeInOut" }}
                        style={{ overflow: "hidden" }}
                    >
                        <div className="info-card-inner">
                            {children}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
