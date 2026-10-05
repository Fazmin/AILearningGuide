import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "@fontsource-variable/newsreader";
import "@fontsource-variable/newsreader/wght-italic.css";
import "./styles.css";
import { VisualGallery } from "./module-sdk/VisualGallery";

document.documentElement.dataset.theme = "light";
document.documentElement.dataset.contrast = "normal";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <VisualGallery />
  </StrictMode>,
);
