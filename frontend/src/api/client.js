import axios from "axios";

// When VITE_API_BASE_URL is empty (HF Spaces), calls go to same origin.
// In local dev, set it to http://localhost:8000 in .env.local.
const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

const api = axios.create({
    baseURL: BASE,
    timeout: 60000,
    headers: { "Content-Type": "application/json" },
});

function fmtError(err) {
    if (!err.response) return "Cannot reach backend — check that the server is running on port 8000.";
    const d = err.response.data;
    if (err.response.status === 422) return "Invalid input — please enter a non-empty sentence.";
    return d?.detail ?? `Server error ${err.response.status}`;
}

export async function checkHealth() {
    try {
        const r = await api.get("/health");
        return r.data;
    } catch {
        return null;
    }
}

export async function encode(sentence, layer = null) {
    try {
        const payload = { sentence };
        if (layer !== null) payload.layer = layer;
        const r = await api.post("/encode", payload);
        return { data: r.data, error: null };
    } catch (err) {
        return { data: null, error: fmtError(err) };
    }
}

export async function similarity(sentenceA, sentenceB) {
    try {
        const r = await api.post("/similarity", { sentence_a: sentenceA, sentence_b: sentenceB });
        return { data: r.data, error: null };
    } catch (err) {
        return { data: null, error: fmtError(err) };
    }
}

export const getSimilarity = async (sentenceA, sentenceB) => {
    const response = await api.post("/similarity", {
        sentence_a: sentenceA,
        sentence_b: sentenceB,
    });
    return response.data;
};