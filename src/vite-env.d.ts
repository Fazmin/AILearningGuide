/// <reference types="vite/client" />

declare module "*.mdx" {
  import type { ComponentType } from "react";

  const Component: ComponentType;
  export default Component;
}

interface Window {
  __TAURI_INTERNALS__?: unknown;
}
