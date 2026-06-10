"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, Database, ChevronRight } from "lucide-react";

export default function CommandCenter() {
  const [apiConnected, setApiConnected] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          const data = await res.json();
          setApiConnected(data.status === "healthy");
          setModelLoaded(data.model_loaded);
        }
      } catch {
        setApiConnected(false);
        setModelLoaded(false);
      }
    }
    checkHealth();
  }, [API_BASE]);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-black text-white px-8">
      <div className="max-w-2xl w-full flex flex-col gap-12">
        <div className="flex flex-col gap-4">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tighter">
            Enterprise Churn Engine
          </h1>
          <p className="text-white/60 text-lg md:text-xl font-light tracking-wide max-w-lg">
            Predict customer churn risk and generate AI-driven retention strategies in real-time.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center gap-4 p-6 border border-white/10">
            <Activity className={apiConnected ? "text-white" : "text-white/30"} />
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-widest uppercase">API Engine</span>
              <span className="text-xs text-white/50 tracking-wider">
                {apiConnected ? "Connected" : "Disconnected"}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4 p-6 border border-white/10">
            <Database className={modelLoaded ? "text-white" : "text-white/30"} />
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-widest uppercase">Predictive Model</span>
              <span className="text-xs text-white/50 tracking-wider">
                {modelLoaded ? "Loaded" : "Offline"}
              </span>
            </div>
          </div>
        </div>

        <div className="pt-8">
          <Link
            href="/parameters"
            className="group flex items-center justify-between p-6 border border-white/10 hover:border-white/40 transition-colors duration-500 w-full md:w-2/3"
          >
            <span className="text-sm font-bold tracking-widest uppercase">Enter Engine</span>
            <ChevronRight className="text-white/50 group-hover:text-white group-hover:translate-x-2 transition-all duration-500" />
          </Link>
        </div>
      </div>
    </div>
  );
}
