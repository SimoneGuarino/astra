import type { KeyboardEvent } from "react";

export type MemoryGraphSearchOverlayProps = {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  onClose: () => void;
};

export function MemoryGraphSearchOverlay({ value, onChange, onSearch, onClose }: MemoryGraphSearchOverlayProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") onSearch();
    if (event.key === "Escape") onClose();
  };

  return (
    <section className="memory-graph-search-overlay">
      <input
        autoFocus
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Cerca nel cervello RAG: identità, preferenze, recap, STT, routing, bug..."
      />
      <button type="button" className="memory-graph-button" onClick={onSearch}>
        Search
      </button>
    </section>
  );
}
