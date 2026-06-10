import { z } from "zod";

export const ChurnInputSchema = z.object({
  Gender: z.string().nullable().optional(),
  SeniorCitizen: z.number().int().min(0).max(1).nullable().optional(),
  Partner: z.number().int().min(0).max(1).nullable().optional(),
  Dependents: z.number().int().min(0).max(1).nullable().optional(),
  tenure: z.number().int().nonnegative().nullable().optional(),
  PhoneService: z.number().int().min(0).max(1).nullable().optional(),
  MultipleLines: z.number().int().min(0).max(1).nullable().optional(),
  InternetService: z.number().int().min(0).max(1).nullable().optional(),
  OnlineSecurity: z.number().int().min(0).max(1).nullable().optional(),
  OnlineBackup: z.number().int().min(0).max(1).nullable().optional(),
  DeviceProtection: z.number().int().min(0).max(1).nullable().optional(),
  TechSupport: z.number().int().min(0).max(1).nullable().optional(),
  StreamingTV: z.number().int().min(0).max(1).nullable().optional(),
  StreamingMovies: z.number().int().min(0).max(1).nullable().optional(),
  Contract: z.string().nullable().optional(),
  PaperlessBilling: z.number().int().min(0).max(1).nullable().optional(),
  PaymentMethod: z.string().nullable().optional(),
  MonthlyCharges: z.number().nonnegative().nullable().optional(),
  TotalCharges: z.number().nonnegative().nullable().optional(),
  Married: z.number().int().min(0).max(1).nullable().optional(),
  NumberOfDependents: z.number().int().nonnegative().nullable().optional(),
  NumberOfReferrals: z.number().int().nonnegative().nullable().optional(),
  SatisfactionScore: z.number().int().min(1).max(5).nullable().optional(),
  InternetType: z.string().nullable().optional(),
  Offer: z.string().nullable().optional(),
  Age: z.number().int().nonnegative().nullable().optional(),
  AvgMonthlyGBDownload: z.number().int().nonnegative().nullable().optional(),
  AvgMonthlyLongDistanceCharges: z.number().nonnegative().nullable().optional(),
  CLTV: z.number().int().nonnegative().nullable().optional(),
  Under30: z.number().int().min(0).max(1).nullable().optional(),
  UnlimitedData: z.number().int().min(0).max(1).nullable().optional(),
  StreamingMusic: z.number().int().min(0).max(1).nullable().optional(),
  ReferredAFriend: z.number().int().min(0).max(1).nullable().optional(),
  TotalRefunds: z.number().nonnegative().nullable().optional(),
  TotalExtraDataCharges: z.number().int().nonnegative().nullable().optional(),
  TotalLongDistanceCharges: z.number().nonnegative().nullable().optional(),
  TotalRevenue: z.number().nonnegative().nullable().optional(),
});

export type ChurnInput = z.infer<typeof ChurnInputSchema>;

export const ChurnResponseSchema = z.object({
  prediction: z.number().int(),
  churn_probability: z.number(),
  retention_risk: z.string(),
});

export type ChurnResponse = z.infer<typeof ChurnResponseSchema>;

export const RetentionScriptResponseSchema = z.object({
  script: z.string(),
});

export type RetentionScriptResponse = z.infer<typeof RetentionScriptResponseSchema>;
