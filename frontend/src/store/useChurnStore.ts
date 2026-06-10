import { create } from "zustand";

interface PredictionResult {
  prediction: number;
  churn_probability: number;
  retention_risk: string;
}

interface RetentionResult {
  script: string;
}

interface ChurnState {
  prediction: PredictionResult | null;
  retention: RetentionResult | null;
  setResults: (prediction: PredictionResult, retention: RetentionResult) => void;
  clearResults: () => void;
}

export const useChurnStore = create<ChurnState>((set) => ({
  prediction: null,
  retention: null,
  setResults: (prediction, retention) => set({ prediction, retention }),
  clearResults: () => set({ prediction: null, retention: null }),
}));
