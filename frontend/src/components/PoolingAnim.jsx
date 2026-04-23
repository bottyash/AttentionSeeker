import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import * as d3 from "d3";
import useVizStore from "../store/useVizStore";

export default function PoolingAnim() {
    const { data } = useVizStore();
    const svgRef = useRef(null);

    if (!data) return <div className="placeholder">Run encode first to see pooling.</div>;

    const { tokens, layer_outputs, pooled_embed } = data.data;
    const lastLayerKey = `layer_${Object.keys(layer_outputs).length - 1}`;
    const lastLayer = layer_outputs[lastLayerKey];

    return (
        <div className="pooling-anim">
            <h2 className="section-title">Mean Pooling</h2>
            <p className="section-desc">
                All {tokens.length} token vectors at the final layer are averaged dimension-by-dimension into one sentence vector.
                Watch them collapse ↓
            </p>

            <div className="pool-rows">
                {tokens.map((tok, i) => (
                    <motion.div
                        key={i}
                        className="pool-row"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.06 }}
                    >
                        <span className="pool-token-label">{tok}</span>
                        <div className="pool-bar-track">
                            {lastLayer?.[i]?.slice(0, 64).map((v, j) => (
                                <div
                                    key={j}
                                    className="pool-bar-cell"
                                    style={{
                                        backgroundColor: v >= 0 ? `rgba(45,212,191,${Math.min(Math.abs(v), 1)})` : `rgba(251,113,133,${Math.min(Math.abs(v), 1)})`,
                                    }}
                                />
                            ))}
                        </div>
                    </motion.div>
                ))}

                {/* Arrow */}
                <motion.div
                    className="pool-arrow"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: tokens.length * 0.06 + 0.2 }}
                >
                    ↓ average
                </motion.div>

                {/* Result */}
                <motion.div
                    className="pool-row result"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: tokens.length * 0.06 + 0.5, type: "spring" }}
                >
                    <span className="pool-token-label" style={{ color: "#818cf8" }}>sentence</span>
                    <div className="pool-bar-track">
                        {pooled_embed?.slice(0, 64).map((v, j) => (
                            <div
                                key={j}
                                className="pool-bar-cell"
                                style={{
                                    backgroundColor: v >= 0 ? `rgba(45,212,191,${Math.min(Math.abs(v), 1)})` : `rgba(251,113,133,${Math.min(Math.abs(v), 1)})`,
                                    height: "16px",
                                }}
                            />
                        ))}
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
