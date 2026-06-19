import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_BUILD_TIME: new Date().toISOString().replace("T", " ").split(".")[0] + " UTC",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination:
          process.env.BACKEND_INTERNAL_URL || "http://127.0.0.1:8000/:path*",
      },
    ];
  },
};

export default nextConfig;
