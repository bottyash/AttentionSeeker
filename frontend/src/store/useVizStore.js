import { create } from "zustand";

const useVizStore = create((set) => ({
    sentence: "",
    sentenceB: "",       // for similarity view (step 9)
    loading: false,
    currentStep: 0,      // 0–8
    data: null,          // full /encode response payload
    similarityData: null,// /similarity response
    selectedLayer: 0,    // 0–5 (for attention heatmap)
    selectedHead: 0,     // 0–11
    error: null,

    setSentence: (s) => set({ sentence: s }),
    setSentenceB: (s) => set({ sentenceB: s }),
    setLoading: (v) => set({ loading: v }),
    setCurrentStep: (n) => set({ currentStep: Math.max(0, Math.min(8, n)) }),
    nextStep: () => set((state) => ({ currentStep: Math.min(8, state.currentStep + 1) })),
    prevStep: () => set((state) => ({ currentStep: Math.max(0, state.currentStep - 1) })),
    setData: (d) => set({ data: d }),
    setSimilarityData: (d) => set({ similarityData: d }),
    setSelectedLayer: (n) => set({ selectedLayer: n }),
    setSelectedHead: (n) => set({ selectedHead: n }),
    setError: (e) => set({ error: e }),
    reset: () => set({ data: null, similarityData: null, currentStep: 0, error: null }),
}));

export default useVizStore;
