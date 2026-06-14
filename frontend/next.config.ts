import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_BUILD_TIME: new Date().toISOString().replace("T", " ").split(".")[0] + " UTC",
  },
};

export default nextConfig;
