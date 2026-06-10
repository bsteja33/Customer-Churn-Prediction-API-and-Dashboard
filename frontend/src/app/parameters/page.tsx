"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Send, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useChurnStore } from "../../store/useChurnStore";
import { FormField, FieldDef } from "../../components/ui/FormField";
import { ChurnInputSchema } from "../../lib/schema";
import { z } from "zod";

interface FieldGroup {
  title: string;
  fields: FieldDef[];
}

const YES_NO_FIELDS = new Set([
  "Married", "PaperlessBilling", "PhoneService", "MultipleLines",
  "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
  "TechSupport", "StreamingTV", "StreamingMovies", "StreamingMusic",
  "UnlimitedData", "SeniorCitizen", "Partner", "Dependents",
  "Under30", "ReferredAFriend",
]);

const FIELDS: FieldDef[] = [
  { key: "Gender", label: "Gender", type: "select", options: ["Male", "Female"] },
  { key: "SeniorCitizen", label: "Senior Citizen", type: "select", options: ["Yes", "No"] },
  { key: "Partner", label: "Partner", type: "select", options: ["Yes", "No"] },
  { key: "Dependents", label: "Has Dependents", type: "select", options: ["Yes", "No"] },
  { key: "Married", label: "Married", type: "select", options: ["Yes", "No"] },
  { key: "Under30", label: "Under 30", type: "select", options: ["Yes", "No"] },
  { key: "ReferredAFriend", label: "Referred a Friend", type: "select", options: ["Yes", "No"] },
  { key: "Age", label: "Age", type: "number" },
  { key: "NumberOfDependents", label: "Dependents (Count)", type: "number" },
  { key: "NumberOfReferrals", label: "Referrals", type: "number" },
  { key: "SatisfactionScore", label: "Satisfaction Score (1-5)", type: "number" },
  { key: "CLTV", label: "CLTV", type: "number" },
  { key: "tenure", label: "Tenure (Months)", type: "number" },
  { key: "Contract", label: "Contract", type: "select", options: ["Month-to-Month", "One Year", "Two Year"] },
  { key: "Offer", label: "Offer", type: "select", options: ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"] },
  { key: "PaperlessBilling", label: "Paperless Billing", type: "select", options: ["Yes", "No"] },
  { key: "PaymentMethod", label: "Payment Method", type: "select", options: ["Bank Withdrawal", "Credit Card", "Mailed Check"] },
  { key: "PhoneService", label: "Phone Service", type: "select", options: ["Yes", "No"] },
  { key: "MultipleLines", label: "Multiple Lines", type: "select", options: ["Yes", "No"] },
  { key: "InternetService", label: "Internet Service", type: "select", options: ["Yes", "No"] },
  { key: "InternetType", label: "Internet Type", type: "select", options: ["DSL", "Fiber Optic", "Cable", "None"] },
  { key: "OnlineSecurity", label: "Online Security", type: "select", options: ["Yes", "No"] },
  { key: "OnlineBackup", label: "Online Backup", type: "select", options: ["Yes", "No"] },
  { key: "DeviceProtection", label: "Device Protection", type: "select", options: ["Yes", "No"] },
  { key: "TechSupport", label: "Premium Support", type: "select", options: ["Yes", "No"] },
  { key: "StreamingTV", label: "Streaming TV", type: "select", options: ["Yes", "No"] },
  { key: "StreamingMovies", label: "Streaming Movies", type: "select", options: ["Yes", "No"] },
  { key: "StreamingMusic", label: "Streaming Music", type: "select", options: ["Yes", "No"] },
  { key: "UnlimitedData", label: "Unlimited Data", type: "select", options: ["Yes", "No"] },
  { key: "AvgMonthlyLongDistanceCharges", label: "Avg LD Charges", type: "number" },
  { key: "AvgMonthlyGBDownload", label: "Avg GB Download", type: "number" },
  { key: "MonthlyCharges", label: "Monthly Charge", type: "number" },
  { key: "TotalCharges", label: "Total Charges", type: "number" },
  { key: "TotalRefunds", label: "Total Refunds", type: "number" },
  { key: "TotalExtraDataCharges", label: "Extra Data Charges", type: "number" },
  { key: "TotalLongDistanceCharges", label: "Total LD Charges", type: "number" },
  { key: "TotalRevenue", label: "Total Revenue", type: "number" },
];

const FIELD_MAP = new Map(FIELDS.map((f) => [f.key, f]));

const GROUPS: FieldGroup[] = [
  {
    title: "Personal & Account",
    fields: ["Gender", "SeniorCitizen", "Partner", "Dependents", "Married", "Under30", "ReferredAFriend", "Age", "NumberOfDependents", "NumberOfReferrals", "SatisfactionScore", "tenure", "Contract", "Offer"]
      .map((k) => FIELD_MAP.get(k)!),
  },
  {
    title: "Services",
    fields: ["PhoneService", "MultipleLines", "InternetService", "InternetType", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
      .map((k) => FIELD_MAP.get(k)!),
  },
  {
    title: "Streaming & Media",
    fields: ["StreamingTV", "StreamingMovies", "StreamingMusic", "UnlimitedData", "PaperlessBilling", "PaymentMethod"]
      .map((k) => FIELD_MAP.get(k)!),
  },
  {
    title: "Charges & Usage",
    fields: ["MonthlyCharges", "TotalCharges", "TotalRefunds", "TotalExtraDataCharges", "TotalLongDistanceCharges", "TotalRevenue", "AvgMonthlyLongDistanceCharges", "AvgMonthlyGBDownload", "CLTV"]
      .map((k) => FIELD_MAP.get(k)!),
  },
];

const INITIAL_FORM: Record<string, string | number | null> = Object.fromEntries(
  FIELDS.map((f) => [f.key, f.type === "number" ? null : ""])
);

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ParametersPage() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setResults = useChurnStore((state) => state.setResults);
  const router = useRouter();

  const updateField = (key: string, value: string | number) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    try {
      const payload: Record<string, string | number | null | undefined> = {};
      for (const [k, v] of Object.entries(form)) {
        if (v !== null && v !== "") {
          const field = FIELD_MAP.get(k);
          if (field?.type === "number") {
            payload[k] = Number(v);
          } else if (YES_NO_FIELDS.has(k)) {
            payload[k] = v === "Yes" ? 1 : v === "No" ? 0 : v;
          } else {
            payload[k] = v;
          }
        } else {
          payload[k] = null;
        }
      }

      const validatedPayload = ChurnInputSchema.parse(payload);

      const predictRes = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validatedPayload),
      });

      if (!predictRes.ok) {
        const errBody = await predictRes.json().catch(() => null);
        throw new Error(errBody?.detail || `Prediction failed with status ${predictRes.status}`);
      }

      const predData = await predictRes.json();

      const scriptRes = await fetch(`${API_BASE}/generate_retention_script`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          risk_level: predData.retention_risk,
          reasons: `Churn probability ${(predData.churn_probability * 100).toFixed(1)}%. Contract: ${form["Contract"] || "Unknown"}. Tenure: ${form["tenure"] || "Unknown"} months.`,
        }),
      });

      const scriptData = scriptRes.ok ? await scriptRes.json() : { script: "Failed to generate script." };

      setResults(predData, scriptData);
      router.push("/analysis");
    } catch (err) {
      if (err instanceof z.ZodError) {
        setError(
          err.issues
            .map((e) => `${e.path.join(".")}: ${e.message}`)
            .join(", ")
        );
      } else {
        setError(err instanceof Error ? err.message : "Analysis request failed");
      }
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans antialiased overflow-x-hidden">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 flex flex-col gap-10">
        <header className="flex items-center justify-between border-b border-white/10 pb-6">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-white/50 hover:text-white transition-colors">
              <ArrowLeft size={24} />
            </Link>
            <h1 className="text-2xl font-bold tracking-widest uppercase">Input Engine</h1>
          </div>
          <span className="text-xs uppercase tracking-widest text-white/50">Telco Parameters</span>
        </header>

        {error && (
          <div className="p-4 border border-red text-red text-sm font-mono tracking-wide">
            {error}
          </div>
        )}

        <div className="flex flex-col gap-y-12">
          {GROUPS.map((group) => (
            <section
              key={group.title}
              className="bg-zinc-950 border border-zinc-800 p-6 rounded-none"
            >
              <h2 className="text-sm uppercase tracking-widest text-white/70 mb-8 pb-4 border-b border-zinc-800 font-sans">
                {group.title}
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {group.fields.map((field) => (
                  <FormField
                    key={field.key}
                    field={field}
                    value={form[field.key]}
                    onChange={updateField}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>

        <footer className="pt-8 border-t border-white/10 flex justify-end">
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="flex items-center gap-3 px-8 py-4 bg-white text-black text-sm font-bold tracking-widest uppercase hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-sans"
          >
            {loading ? "Processing..." : "Analyze"}
            {!loading && <Send size={16} />}
          </button>
        </footer>
      </div>
    </div>
  );
}
