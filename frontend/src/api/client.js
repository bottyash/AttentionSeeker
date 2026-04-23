import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({ baseURL: BASE_URL });

export async function encode(sentence, layer = null) {
    try {
        const res = await client.post("/encode", { sentence, layer });
        return { data: res.data, error: null };
    } catch (err) {
        const detail = err.response?.data?.detail;
        if (err.response?.status === 422) {
            return { data: null, error: detail || "Sentence exceeds 128 tokens — please shorten it." };
        }
        if (!err.response) {
            return { data: null, error: "Cannot reach backend — check that the server is running on port 8000." };
        }
        return { data: null, error: detail || "Unexpected server error." };
    }
}

export async function getSimilarity(sentence_a, sentence_b) {
    try {
        const res = await client.post("/similarity", { sentence_a, sentence_b });
        return { data: res.data, error: null };
    } catch (err) {
        const detail = err.response?.data?.detail;
        if (!err.response) {
            return { data: null, error: "Cannot reach backend — check that the server is running on port 8000." };
        }
        return { data: null, error: detail || "Unexpected server error." };
    }
}

export async function checkHealth() {
    try {
        const res = await client.get("/health");
        return res.data;
    } catch {
        return null;
    }
}
