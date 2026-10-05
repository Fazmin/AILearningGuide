import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

const SAFE_HREF = /^(https?:|mailto:)/i;

const components: Components = {
  a({ href, children }) {
    const safe = href && SAFE_HREF.test(href) ? href : undefined;
    if (!safe) return <span>{children}</span>;
    return (
      <a href={safe} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  },
  img({ alt }) {
    return alt ? <span className="guide-md__alt">{alt}</span> : null;
  },
  table({ children }) {
    return (
      <div className="guide-md__table">
        <table>{children}</table>
      </div>
    );
  },
};

export function GuideMarkdown({ text }: { text: string }) {
  return (
    <div className="guide-md">
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </Markdown>
    </div>
  );
}
