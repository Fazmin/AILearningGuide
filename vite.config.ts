import path from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import remarkFrontmatter from "remark-frontmatter";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [
    mdx({
      remarkPlugins: [remarkFrontmatter],
    }),
    react(),
  ],
  resolve: {
    alias: {
      "@app": path.resolve(__dirname, "./src"),
      "@app/module-sdk": path.resolve(__dirname, "./src/module-sdk/index.ts"),
    },
  },
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts"],
  },
});
