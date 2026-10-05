import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "@fontsource-variable/newsreader";
import "@fontsource-variable/newsreader/wght-italic.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MDXProvider } from "@mdx-js/react";
import App from "./App";
import "./styles.css";

const mdxComponents = {
  code: (props: React.ComponentProps<"code">) => <code {...props} />,
};

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MDXProvider components={mdxComponents}>
      <App />
    </MDXProvider>
  </StrictMode>,
);
