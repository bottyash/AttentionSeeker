import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import useVizStore from "../store/useVizStore";

function EmbedChart({ title, embedding, color }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!embedding || !ref.current) return;
        const el = ref.current;
        el.innerHTML = "";

        const dims = embedding.length;
        const w = el.clientWidth || 560;
        const h = 60;
        const svg = d3.select(el).append("svg").attr("width", w).attr("height", h);
        const x = d3.scaleBand().domain(d3.range(dims)).range([0, w]).padding(0.05);
        const extent = d3.extent(embedding);
        const y = d3.scaleLinear().domain([Math.min(extent[0], 0), Math.max(extent[1], 0)]).range([h, 0]);
        const zero = y(0);

        svg.selectAll("rect")
            .data(embedding)
            .join("rect")
            .attr("x", (_, i) => x(i))
            .attr("y", d => d >= 0 ? y(d) : zero)
            .attr("height", d => Math.abs(y(d) - zero))
            .attr("width", x.bandwidth())
            .attr("fill", d => d >= 0 ? "#2dd4bf" : "#fb7185");
    }, [embedding]);

    return (
        <div className="embed-chart">
            <p className="chart-label">{title}</p>
            <div ref={ref} style={{ width: "100%", overflowX: "auto" }} />
        </div>
    );
}

function SumChart({ tokenEmbed, posEmbed }) {
    const summed = tokenEmbed?.map((v, i) => v + (posEmbed?.[i] ?? 0));
    return <EmbedChart title="Sum (input to layer 1)" embedding={summed} />;
}

export default function EmbeddingBar({ step }) {
    const { data, selectedLayer } = useVizStore();
    const [zoomIdx, setZoomIdx] = useState(0);
    const WINDOW = 48;

    if (!data) return <div className="placeholder">Run encode first to see embeddings.</div>;

    const { token_embeds, pos_embeds, tokens, layer_outputs } = data.data;

    // step 3 = token embed, step 4 = positional, step 7 = pooled (mean of last layer)
    const showToken = step === 3 || step === 4;
    const showLayer = step === 5;
    const showPool = step === 7;

    const pooled = showPool
        ? data.data.pooled_embed
        : null;

    const layerTensor = showLayer ? layer_outputs?.[`layer_${selectedLayer}`] : null;

    return (
        <div className="embedding-bar">
            <h2 className="section-title">
                {showPool ? "Mean Pooled Embedding" : showLayer ? `Layer ${selectedLayer + 1} Hidden States` : "Token & Positional Embeddings"}
            </h2>

            {showToken && (
                <>
                    <EmbedChart title="Token embedding (word)" embedding={token_embeds?.[0]} />
                    <EmbedChart title="Positional embedding" embedding={pos_embeds?.[0]} />
                    <SumChart tokenEmbed={token_embeds?.[0]} posEmbed={pos_embeds?.[0]} />
                </>
            )}

            {showLayer && layerTensor && (
                <>
                    <p className="section-desc">
                        Showing hidden states for all {tokens.length} tokens after layer {selectedLayer + 1}.
                    </p>
                    {layerTensor.map((vec, i) => (
                        <EmbedChart key={i} title={tokens[i]} embedding={vec} />
                    ))}
                </>
            )}

            {showPool && (
                <>
                    <p className="section-desc">
                        All {tokens.length} token vectors averaged dimension-by-dimension into one 384-dim sentence vector.
                    </p>
                    <EmbedChart title="Pooled sentence embedding (384 dims)" embedding={pooled} />
                </>
            )}
        </div>
    );
}
