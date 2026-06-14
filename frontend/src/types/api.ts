export interface PredictResponse {
  prediction: number;
  churn_probability: number;
  retention_risk: string;
}

export interface RetentionScriptResponse {
  script: string;
}
