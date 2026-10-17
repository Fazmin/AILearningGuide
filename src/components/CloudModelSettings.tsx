import { Gauge } from "lucide-react";
import { useState } from "react";
import {
  DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  MAX_MAX_REPLY_TOKENS,
  MAX_SYSTEM_PROMPT_CHARS,
  MIN_MAX_REPLY_TOKENS,
} from "@app/lib/cloud-providers";
import { useAppStore } from "@app/store/app-store";

export function CloudModelSettings() {
  const store = useAppStore();
  const [maxTokens, setMaxTokens] = useState(String(store.maxReplyTokens));
  // An empty stored prompt means "use the built-in template", so the field shows that template.
  const [systemPrompt, setSystemPrompt] = useState(store.systemPrompt || DEFAULT_SYSTEM_PROMPT_TEMPLATE);

  const commitMaxTokens = () => {
    const parsed = Number.parseInt(maxTokens, 10);
    if (Number.isFinite(parsed)) store.setMaxReplyTokens(parsed);
    // Show the clamped value the store kept, so an out-of-range entry is visibly corrected.
    setMaxTokens(String(useAppStore.getState().maxReplyTokens));
  };

  // Storing the untouched template as "" keeps it following future built-in updates.
  const commitSystemPrompt = (text: string) => {
    store.setSystemPrompt(text === DEFAULT_SYSTEM_PROMPT_TEMPLATE ? "" : text);
  };

  const restoreDefaultPrompt = () => {
    setSystemPrompt(DEFAULT_SYSTEM_PROMPT_TEMPLATE);
    store.setSystemPrompt("");
  };

  return (
    <div className="provider-form">
      <label className="settings-field">
        <span>Max Reply Tokens</span>
        <div>
          <Gauge />
          <input
            type="number"
            inputMode="numeric"
            min={MIN_MAX_REPLY_TOKENS}
            max={MAX_MAX_REPLY_TOKENS}
            step={1}
            value={maxTokens}
            onChange={(event) => setMaxTokens(event.target.value)}
            onBlur={commitMaxTokens}
          />
        </div>
      </label>
      <p>
        Higher values can increase costs on cloud models, which are billed per generated token.
      </p>

      <label className="settings-field">
        <span>System Prompt</span>
        <textarea
          value={systemPrompt}
          maxLength={MAX_SYSTEM_PROMPT_CHARS}
          rows={10}
          spellCheck={false}
          onChange={(event) => setSystemPrompt(event.target.value)}
          onBlur={() => commitSystemPrompt(systemPrompt)}
        />
      </label>
      <button
        type="button"
        className="settings-action settings-action--secondary"
        disabled={systemPrompt === DEFAULT_SYSTEM_PROMPT_TEMPLATE}
        onClick={restoreDefaultPrompt}
      >
        Restore default prompt
      </button>
      <p>
        Both settings apply to the cloud topic guide. The placeholders in braces, such as {"{module_title}"}, are filled in for each question.
      </p>
    </div>
  );
}
