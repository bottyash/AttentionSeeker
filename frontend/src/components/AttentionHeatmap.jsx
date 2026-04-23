import { useEffect, useRef } from "react";
import * as d3 from "d3";
import useVizStore from "../store/useVizStore";

const LAYERS = [0, 1, 2, 3, 4, 5];
const HEADS = Array.from({ length: 12 }, (_, i) => i);

export default function AttentionHeatmap() {
    const { data, selectedLayer, selectedHead, setSelectedLayer, setSelectedHead } = useVizStore();
    const svgRef = useRef(null);

    const attentions = data?.data?.attentions;
    const tokens = data?.data?.tokens;

    useEffect(() => {
        if (!attentions || !tokens || !svgRef.current) return;

        const grid = attentions[selectedLayer]?.[selectedHead];
        if (!grid) return;

        const el = svgRef.current;
        el.innerHTML = "";

        const margin = { top: 40, right: 20, bottom: 40, left: 60 };
        const n = tokens.length;
        const cellSize = Math.min(Math.floor((500 - margin.left - margin.right) / n), 44);
        const W = cellSize * n + margin.left + margin.right;
        const H = cellSize * n + margin.top + margin.bottom;

        const svg = d3
            .select(el)
            .append("svg")
            .attr("width", W)
            .attr("height", H);

        const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

        // Blue-purple sequential scale (color-blind safe)
        const colorScale = d3
            .scaleSequential(d3.interpolatePurples)
            .domain([0, 1]);

        // Tooltip
        const tooltip = d3.select("body").selectAll(".attn-tooltip").data([1])
            .join("div")
            .attr("class", "attn-tooltip")
            .style("position", "absolute")
            .style("padding", "6px 10px")
            .style("background", "#1e1b4b")
            .style("color", "#e0e7ff")
            .style("border-radius", "6px")
            .style("font-size", "12px")
            .style("pointer-events", "none")
            .style("opacity", 0);

        // Cells
        grid.forEach((row, r) => {
            row.forEach((val, c) => {
                g.append("rect")
                    .attr("x", c * cellSize)
                    .attr("y", r * cellSize)
                    .attr("width", cellSize - 1)
                    .attr("height", cellSize - 1)
                    .attr("fill", colorScale(val))
                    .attr("rx", 2)
                    .attr("aria-label", `${tokens[r]} to ${tokens[c]}: ${val.toFixed(3)}`)
                    .on("mouseover", function (event) {
                        tooltip
                            .style("opacity", 1)
                            .html(`<b>${tokens[r]}</b> → <b>${tokens[c]}</b>: ${val.toFixed(3)}`);
                    })
                    .on("mousemove", function (event) {
                        tooltip
                            .style("left", `${event.pageX + 12}px`)
                            .style("top", `${event.pageY - 28}px`);
                    })
                    .on("mouseout", function () {
                        tooltip.style("opacity", 0);
                    });
            });
        });

        // X axis token labels
        g.selectAll(".xlabel")
            .data(tokens)
            .join("text")
            .attr("class", "xlabel")
            .attr("x", (_, i) => i * cellSize + cellSize / 2)
            .attr("y", -8)
            .attr("text-anchor", "middle")
            .attr("font-size", Math.min(11, cellSize - 2))
            .attr("fill", "#94a3b8")
            .text(d => d.length > 5 ? d.slice(0, 4) + "…" : d);

        // Y axis token labels
        g.selectAll(".ylabel")
            .data(tokens)
            .join("text")
            .attr("class", "ylabel")
            .attr("x", -6)
            .attr("y", (_, i) => i * cellSize + cellSize / 2 + 4)
            .attr("text-anchor", "end")
            .attr("font-size", Math.min(11, cellSize - 2))
            .attr("fill", "#94a3b8")
            .text(d => d.length > 6 ? d.slice(0, 5) + "…" : d);
    }, [attentions, tokens, selectedLayer, selectedHead]);

    if (!data) return <div className="placeholder">Run encode first to see attention.</div>;

    return (
        <div className="attention-heatmap">
            <h2 className="section-title">Attention Weights — Layer {selectedLayer + 1} / Head {selectedHead + 1}</h2>
            <p className="section-desc">
                Each cell shows how much a query token (row) attends to a key token (column). Darker = stronger attention.
            </p>

            {/* Layer selector */}
            <div className="selector-row" role="tablist" aria-label="Layer selector">
                {LAYERS.map(l => (
                    <button
                        key={l}
                        role="tab"
                        aria-selected={selectedLayer === l}
                        className={`selector-tab ${selectedLayer === l ? "active" : ""}`}
                        onClick={() => setSelectedLayer(l)}
                    >
                        L{l + 1}
                    </button>
                ))}
            </div>

            {/* Head selector */}
            <div className="selector-row" role="tablist" aria-label="Head selector">
                {HEADS.map(h => (
                    <button
                        key={h}
                        role="tab"
                        aria-selected={selectedHead === h}
                        className={`selector-tab ${selectedHead === h ? "active" : ""}`}
                        onClick={() => setSelectedHead(h)}
                    >
                        H{h + 1}
                    </button>
                ))}
            </div>

            <div className="heatmap-container" ref={svgRef} />
        </div>
    );
}
