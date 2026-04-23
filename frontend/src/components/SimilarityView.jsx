import { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { motion } from "framer-motion";
import useVizStore from "../store/useVizStore";
import { getSimilarity } from "../api/client";

function CosineDial({ score }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!ref.current) return;
        const el = ref.current;
        el.innerHTML = "";

        const W = 220, H = 130;
        const cx = W / 2, cy = H - 20;
        const R = 90;

        const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);

        // Background arc (-1 to 1 mapped to -π/2 .. π/2 — bottom semicircle)
        const arcBg = d3.arc()
            .innerRadius(R - 18)
            .outerRadius(R)
            .startAngle(-Math.PI / 2)
            .endAngle(Math.PI / 2);

        svg.append("path")
            .attr("d", arcBg())
            .attr("transform", `translate(${cx},${cy})`)
            .attr("fill", "#1e1b4b");

        // Score arc
        const angle = ((score + 1) / 2) * Math.PI - Math.PI / 2;
        const arcFill = d3.arc()
            .innerRadius(R - 18)
            .outerRadius(R)
            .startAngle(-Math.PI / 2)
            .endAngle(angle);

        const color = score > 0.7 ? "#2dd4bf" : score > 0.3 ? "#818cf8" : "#fb7185";

        svg.append("path")
            .attr("d", arcFill())
            .attr("transform", `translate(${cx},${cy})`)
            .attr("fill", color)
            .attr("rx", 4);

        // Needle
        const needleX = cx + (R - 25) * Math.cos(angle - Math.PI / 2);
        const needleY = cy + (R - 25) * Math.sin(angle - Math.PI / 2);
        svg.append("line")
            .attr("x1", cx).attr("y1", cy)
            .attr("x2", needleX).attr("y2", needleY)
            .attr("stroke", "#e2e8f0").attr("stroke-width", 2).attr("stroke-linecap", "round");

        // Center label
        svg.append("text")
            .attr("x", cx).attr("y", cy - 12)
            .attr("text-anchor", "middle")
            .attr("font-size", 22)
            .attr("font-weight", "bold")
            .attr("fill", color)
            .text(score.toFixed(3));

        svg.append("text")
            .attr("x", cx).attr("y", cy + 4)
            .attr("text-anchor", "middle")
            .attr("font-size", 11)
            .attr("fill", "#94a3b8")
            .text("cosine similarity");
    }, [score]);

    return <div ref={ref} />;
}

function UMAPScatter({ coords }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!coords || !ref.current) return;
        const el = ref.current;
        el.innerHTML = "";

        const W = 420, H = 320;
        const margin = { top: 20, right: 20, bottom: 20, left: 20 };

        const xs = coords.map(d => d.x);
        const ys = coords.map(d => d.y);

        const xScale = d3.scaleLinear()
            .domain([d3.min(xs), d3.max(xs)])
            .range([margin.left, W - margin.right]);
        const yScale = d3.scaleLinear()
            .domain([d3.min(ys), d3.max(ys)])
            .range([H - margin.bottom, margin.top]);

        const svg = d3.select(el).append("svg").attr("width", W).attr("height", H)
            .style("background", "#0f0e1a")
            .style("border-radius", "12px");

        // Zoom
        const zoom = d3.zoom().scaleExtent([0.5, 8]).on("zoom", (event) => {
            g.attr("transform", event.transform);
        });
        svg.call(zoom);

        const g = svg.append("g");

        const tooltip = d3.select("body").selectAll(".umap-tooltip").data([1])
            .join("div")
            .attr("class", "umap-tooltip")
            .style("position", "absolute")
            .style("padding", "5px 9px")
            .style("background", "#1e1b4b")
            .style("color", "#e0e7ff")
            .style("border-radius", "6px")
            .style("font-size", "11px")
            .style("pointer-events", "none")
            .style("opacity", 0);

        coords.forEach(d => {
            const isRef = d.is_reference;
            const r = isRef ? 5 : 10;
            const fill = isRef ? "#4f46e5" : (coords.indexOf(d) === 0 ? "#2dd4bf" : "#f472b6");

            g.append("circle")
                .attr("cx", xScale(d.x))
                .attr("cy", yScale(d.y))
                .attr("r", r)
                .attr("fill", fill)
                .attr("opacity", isRef ? 0.6 : 1)
                .on("mouseover", (event) => tooltip.style("opacity", 1).text(d.label))
                .on("mousemove", (event) => tooltip.style("left", `${event.pageX + 10}px`).style("top", `${event.pageY - 20}px`))
                .on("mouseout", () => tooltip.style("opacity", 0));

            if (!isRef) {
                g.append("text")
                    .attr("x", xScale(d.x) + 12)
                    .attr("y", yScale(d.y) + 4)
                    .attr("font-size", 11)
                    .attr("fill", fill)
                    .text(d.label.length > 24 ? d.label.slice(0, 22) + "…" : d.label);
            }
        });
    }, [coords]);

    return <div ref={ref} />;
}

export default function SimilarityView() {
    const { sentence, sentenceB, setSentenceB, setSimilarityData, similarityData, setError, setLoading, loading } = useVizStore();
    const [localB, setLocalB] = useState("");

    async function handleCompare() {
        if (!sentence.trim() || !localB.trim()) return;
        setSentenceB(localB);
        setLoading(true);
        const { data, error } = await getSimilarity(sentence, localB);
        setLoading(false);
        if (error) setError(error);
        else setSimilarityData(data);
    }

    return (
        <div className="similarity-view">
            <h2 className="section-title">Similarity Comparison</h2>
            <p className="section-desc">
                Compare two sentences in embedding space. The dial shows cosine similarity (–1 to 1).
                The scatter plot shows where both sit relative to 8 reference sentences.
            </p>

            <div className="sim-inputs">
                <div className="sim-input-row">
                    <label>Sentence A (already encoded)</label>
                    <div className="sim-static">{sentence || "—"}</div>
                </div>
                <div className="sim-input-row">
                    <label htmlFor="sentence-b">Sentence B</label>
                    <input
                        id="sentence-b"
                        className="sim-input"
                        value={localB}
                        onChange={e => setLocalB(e.target.value)}
                        placeholder="Type a sentence to compare…"
                        onKeyDown={e => e.key === "Enter" && handleCompare()}
                    />
                </div>
                <button className="encode-btn" onClick={handleCompare} disabled={loading || !sentence || !localB}>
                    {loading ? "Comparing…" : "Compare →"}
                </button>
            </div>

            {similarityData && (
                <motion.div
                    className="sim-results"
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <CosineDial score={similarityData.cosine_similarity} />
                    <UMAPScatter coords={similarityData.umap_coords} />
                </motion.div>
            )}
        </div>
    );
}
