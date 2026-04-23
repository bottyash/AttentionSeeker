import { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import useVizStore from "../store/useVizStore";
import InfoCard from "./InfoCard";


function TokenPill({ token, embedding, tokenId }) {
    const [open, setOpen] = useState(false);
    const chartRef = useRef(null);

    const isSpecial = token === "[CLS]" || token === "[SEP]";
    const isSubword = token.startsWith("##");

    const pillClass = isSpecial ? "pill special" : isSubword ? "pill subword" : "pill regular";

    useEffect(() => {
        if (!open || !embedding || !chartRef.current) return;
        const el = chartRef.current;
        el.innerHTML = "";

        // Bin 384 dims into 32 buckets
        const bucketSize = Math.ceil(embedding.length / 32);
        const buckets = Array.from({ length: 32 }, (_, i) => {
            const slice = embedding.slice(i * bucketSize, (i + 1) * bucketSize);
            return slice.reduce((a, b) => a + b, 0) / slice.length;
        });

        const w = 260, h = 80;
        const svg = d3.select(el).append("svg").attr("width", w).attr("height", h);
        const x = d3.scaleBand().domain(d3.range(32)).range([0, w]).padding(0.1);
        const y = d3.scaleLinear().domain([d3.min(buckets), d3.max(buckets)]).range([h, 0]);

        svg.selectAll("rect")
            .data(buckets)
            .join("rect")
            .attr("x", (_, i) => x(i))
            .attr("y", d => d >= 0 ? y(d) : y(0))
            .attr("height", d => Math.abs(y(d) - y(0)))
            .attr("width", x.bandwidth())
            .attr("fill", d => d >= 0 ? "#2dd4bf" : "#fb7185");
    }, [open, embedding]);

    return (
        <div className="token-wrapper">
            <button className={pillClass} onClick={() => setOpen(!open)}>
                <span className="token-text">{token}</span>
                <span className="token-id">{tokenId}</span>
            </button>
            {open && (
                <div className="token-panel">
                    <p className="panel-title">384-dim embedding histogram</p>
                    <div ref={chartRef} />
                </div>
            )}
        </div>
    );
}

export default function TokenView() {
    const { data } = useVizStore();

    if (!data) return <div className="placeholder">Run encode first to see tokens.</div>;

    const { tokens, input_ids, token_embeds } = data.data;

    return (
        <div className="token-view">
            <div className="section-header">
                <h2 className="section-title">Tokenization</h2>
                <InfoCard title="What is tokenization?">
                    The model never sees raw text — it sees numbers. WordPiece first breaks your sentence
                    into <strong>sub-word pieces</strong> so that rare words (like "embeddings") can share
                    parts with common ones (like "embed"). Two special tokens are always added:{" "}
                    <span className="chip amber">[CLS]</span> (classification marker, at the start) and{" "}
                    <span className="chip amber">[SEP]</span> (separator, at the end). Click any token
                    pill to inspect its raw 384-dimensional numeric representation.
                </InfoCard>
            </div>
            <p className="section-desc">
                WordPiece splits your text into sub-word tokens.{" "}
                <span className="chip amber">[CLS]</span> and <span className="chip amber">[SEP]</span> are added.
                Click any token to inspect its 384-dim embedding.
            </p>
            <div className="pills-row" role="list" aria-label="Token list">
                {tokens.map((tok, i) => (
                    <TokenPill key={i} token={tok} tokenId={input_ids[i]} embedding={token_embeds[i]} />
                ))}
            </div>
            <p className="token-count-label">{tokens.length} tokens (including special)</p>
        </div>
    );
}

