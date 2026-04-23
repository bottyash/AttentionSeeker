import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import useVizStore from "./store/useVizStore";
import { encode, checkHealth } from "./api/client";

import StepStepper from "./components/StepStepper";
import TokenView from "./components/TokenView";
import AttentionHeatmap from "./components/AttentionHeatmap";
import EmbeddingBar from "./components/EmbeddingBar";
import PoolingAnim from "./components/PoolingAnim";
import SimilarityView from "./components/SimilarityView";
import { TokenSkeleton, HeatmapSkeleton, EmbedSkeleton, PoolSkeleton, GenericSkeleton } from "./components/Skeleton";

const STEP_INFO = [
  {
    title: "Raw Text Input",
    desc: "Type any sentence. The model accepts up to 128 tokens — about 90–100 English words.",
  },
  {
    title: "Tokenization",
    desc: "WordPiece splits your text into sub-word pieces. 'running' might become ['run', '##ning']. Special [CLS] and [SEP] tokens bookend the sequence.",
  },
  {
    title: "Token Embeddings",
    desc: "A lookup table maps each token ID to a 384-dimensional vector. These are learned weights — similar words start close together in this space.",
  },
  {
    title: "Positional Embeddings",
    desc: "Transformers see all tokens at once (no recurrence). Positional embeddings inject word-order information by adding a position-dependent vector to each token.",
  },
  {
    title: "Transformer Layers ×6",
    desc: "Each layer lets every word look at every other word and update its meaning based on context. After 6 rounds, 'bank' in 'river bank' looks very different from 'bank' in 'bank account'.",
  },
  {
    title: "Attention Weights",
    desc: "For each of the 12 heads in each of the 6 layers, a score says how much each word should attend to every other. The heatmap shows these scores.",
  },
  {
    title: "Mean Pooling",
    desc: "To get one vector for the whole sentence, all token vectors at the final layer are averaged dimension-by-dimension. Watch them collapse into one.",
  },
  {
    title: "L2 Normalization",
    desc: "The pooled vector is scaled to unit length. This means cosine similarity between two embeddings equals their dot product — fast and numerically stable.",
  },
  {
    title: "Similarity Comparison",
    desc: "Enter a second sentence to compare. Cosine similarity of 1 = identical meaning, 0 = unrelated, –1 = opposite.",
  },
];

function StepPanel({ step, loading }) {
  if (loading) {
    switch (step) {
      case 1: case 2: return <TokenSkeleton />;
      case 3: case 4: case 5: return <EmbedSkeleton />;
      case 6: return <HeatmapSkeleton />;
      case 7: return <PoolSkeleton />;
      default: return <GenericSkeleton />;
    }
  }

  switch (step) {
    case 0:
      return (
        <div className="step0-hint">
          <span className="hint-icon">💡</span>
          Type a sentence above and click <strong>Encode &rarr;</strong> to begin the walkthrough.
        </div>
      );
    case 1: case 2: return <TokenView />;
    case 3: case 4: return <EmbeddingBar step={step} />;
    case 5: return <EmbeddingBar step={step} />;
    case 6: return <AttentionHeatmap />;
    case 7: return <PoolingAnim />;
    case 8: return <SimilarityView />;
    default: return null;
  }
}

export default function App() {
  const {
    sentence, setSentence, loading, setLoading,
    currentStep, setCurrentStep, setData, setError, error, data,
  } = useVizStore();

  const [healthOk, setHealthOk] = useState(null);
  const [inputVal, setInputVal] = useState("");
  const [tokenCount, setTokenCount] = useState(0);

  // Live approximate token count (~1.3 tokens/word + 2 special tokens)
  const approxTokens = Math.round(inputVal.trim().split(/\s+/).filter(Boolean).length * 1.3) + 2;
  const isLong = inputVal.trim().length > 0 && approxTokens > 100;
  const isEmpty = inputVal.trim().length === 0;

  useEffect(() => {
    checkHealth().then(h => setHealthOk(!!h));
  }, []);

  async function handleEncode() {
    if (isEmpty) return;
    setSentence(inputVal.trim());
    setLoading(true);
    setError(null);
    setCurrentStep(1);
    const { data: result, error: err } = await encode(inputVal.trim());
    setLoading(false);
    if (err) { setError(err); setCurrentStep(0); }
    else {
      setData(result);
      setTokenCount(result.token_count);
    }
  }

  const info = STEP_INFO[currentStep];

  return (
    <div className="app">
      {/* Offline banner */}
      {healthOk === false && (
        <div className="offline-banner">
          ⚠ Cannot reach backend — check that the server is running on port 8000
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <span className="logo-main">Attention</span>
          <span className="logo-accent">Seeker</span>
        </div>
        <p className="tagline">It's not you, it's your embeddings.</p>
      </header>

      {/* Encode input */}
      <div className="encode-bar">
        <div className="encode-input-wrap">
          <input
            id="sentence-input"
            className={`encode-input${isLong ? " warn-border" : ""}`}
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            placeholder="Type a sentence to begin…"
            maxLength={600}
            onKeyDown={e => e.key === "Enter" && handleEncode()}
          />
          {inputVal.length > 0 && (
            <span className={`token-count-badge${isLong ? " warn" : ""}`}>
              ~{approxTokens} tokens{isLong ? " ⚠" : ""}
            </span>
          )}
        </div>
        <button
          id="encode-btn"
          className="encode-btn"
          onClick={handleEncode}
          disabled={loading || isEmpty}
          title={isEmpty ? "Type a sentence to begin" : ""}
        >
          {loading ? <span className="spinner" /> : "Encode →"}
        </button>
      </div>

      {isLong && (
        <div className="warn-banner">
          ⚠ Long input (~{approxTokens} tokens). The model truncates at 128 — consider shortening.
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {/* Stepper */}
      {data && <StepStepper />}

      {/* Step info card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          className="step-info-card"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.25 }}
        >
          <h3 className="step-info-title">
            <span className="step-badge">{currentStep + 1}</span>
            {info.title}
          </h3>
          <p className="step-info-desc">{info.desc}</p>
        </motion.div>
      </AnimatePresence>

      {/* Main visualization panel */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`panel-${currentStep}`}
          className="viz-panel"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          <StepPanel step={currentStep} loading={loading} />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
