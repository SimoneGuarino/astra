export type MemoryGraphLabelMode = "hidden" | "selected" | "active" | "important" | "all";
export type MemoryGraphLayoutPreset = "vault" | "local_focus" | "identity" | "research" | "debug";

export type MemoryGraphLayoutSettings = {
  preset: MemoryGraphLayoutPreset;
  labelMode: MemoryGraphLabelMode;
  labelSize: number;
  repulsion: number;
  linkDistance: number;
  centerForce: number;
  clusterForce: number;
};

export type MemoryGraphViewMode = "global" | "local";

export type MemoryGraphViewSettings = {
  mode: MemoryGraphViewMode;
  localDepth: number;
  showIsolatedNodes: boolean;
  visibleKinds: Record<string, boolean>;
};

export const DEFAULT_MEMORY_GRAPH_VIEW_SETTINGS: MemoryGraphViewSettings = {
  mode: "global",
  localDepth: 2,
  showIsolatedNodes: true,
  visibleKinds: {},
};

export const MEMORY_GRAPH_LAYOUT_PRESETS: Record<MemoryGraphLayoutPreset, MemoryGraphLayoutSettings> = {
  vault: {
    preset: "vault",
    labelMode: "important",
    labelSize: 0.76,
    repulsion: 1.22,
    linkDistance: 1.12,
    centerForce: 0.72,
    clusterForce: 1.36,
  },
  local_focus: {
    preset: "local_focus",
    labelMode: "active",
    labelSize: 0.82,
    repulsion: 1.05,
    linkDistance: 0.96,
    centerForce: 0.88,
    clusterForce: 1.48,
  },
  identity: {
    preset: "identity",
    labelMode: "all",
    labelSize: 0.84,
    repulsion: 1.06,
    linkDistance: 0.96,
    centerForce: 0.86,
    clusterForce: 1.5,
  },
  research: {
    preset: "research",
    labelMode: "important",
    labelSize: 0.72,
    repulsion: 1.36,
    linkDistance: 1.26,
    centerForce: 0.68,
    clusterForce: 1.24,
  },
  debug: {
    preset: "debug",
    labelMode: "all",
    labelSize: 0.68,
    repulsion: 1.42,
    linkDistance: 1.32,
    centerForce: 0.58,
    clusterForce: 1.08,
  },
};

export const DEFAULT_MEMORY_GRAPH_LAYOUT_SETTINGS = MEMORY_GRAPH_LAYOUT_PRESETS.vault;

export type MemoryGraphControlsOverlayProps = {
  settings: MemoryGraphLayoutSettings;
  viewSettings: MemoryGraphViewSettings;
  availableKinds: string[];
  nodeKindLabels: Record<string, string>;
  onChange: (settings: MemoryGraphLayoutSettings) => void;
  onViewChange: (settings: MemoryGraphViewSettings) => void;
  onClose: () => void;
};

const PRESET_LABELS: Record<MemoryGraphLayoutPreset, string> = {
  vault: "Vault",
  local_focus: "Local Focus",
  identity: "Identity",
  research: "Research",
  debug: "Debug",
};

const LABEL_MODE_LABELS: Record<MemoryGraphLabelMode, string> = {
  hidden: "Hidden",
  selected: "Selected",
  active: "Active",
  important: "Important",
  all: "All",
};

export function MemoryGraphControlsOverlay({
  availableKinds,
  nodeKindLabels,
  onChange,
  onClose,
  onViewChange,
  settings,
  viewSettings,
}: MemoryGraphControlsOverlayProps) {
  const update = (partial: Partial<MemoryGraphLayoutSettings>) => {
    onChange({ ...settings, ...partial });
  };
  const updateView = (partial: Partial<MemoryGraphViewSettings>) => {
    onViewChange({ ...viewSettings, ...partial });
  };
  const toggleKind = (kind: string) => {
    const current = viewSettings.visibleKinds[kind] !== false;
    updateView({
      visibleKinds: {
        ...viewSettings.visibleKinds,
        [kind]: !current,
      },
    });
  };

  return (
    <section className="memory-graph-controls-overlay" aria-label="Memory graph layout controls">
      <div className="memory-graph-controls-header">
        <div>
          <span>GRAPH CONTROLS</span>
          <strong>Layout & labels</strong>
        </div>
        <button type="button" onClick={onClose} aria-label="Close graph controls">×</button>
      </div>

      <div className="memory-graph-control-group">
        <span>Graph mode</span>
        <div className="memory-graph-preset-grid">
          <button
            type="button"
            className={viewSettings.mode === "global" ? "memory-graph-control-chip memory-graph-control-chip--active" : "memory-graph-control-chip"}
            onClick={() => updateView({ mode: "global" })}
          >
            Global vault
          </button>
          <button
            type="button"
            className={viewSettings.mode === "local" ? "memory-graph-control-chip memory-graph-control-chip--active" : "memory-graph-control-chip"}
            onClick={() => updateView({ mode: "local" })}
          >
            Local focus
          </button>
          <button
            type="button"
            className={viewSettings.showIsolatedNodes ? "memory-graph-control-chip memory-graph-control-chip--active" : "memory-graph-control-chip"}
            onClick={() => updateView({ showIsolatedNodes: !viewSettings.showIsolatedNodes })}
          >
            {viewSettings.showIsolatedNodes ? "Orphans on" : "Orphans off"}
          </button>
        </div>
      </div>

      <MemoryGraphSlider label="Local depth" value={viewSettings.localDepth} min={1} max={4} step={1} onChange={(value) => updateView({ localDepth: Math.round(value) })} />

      {availableKinds.length ? (
        <div className="memory-graph-control-group">
          <span>Node types</span>
          <div className="memory-graph-kind-filter-grid">
            {availableKinds.map((kind) => {
              const visible = viewSettings.visibleKinds[kind] !== false;
              return (
                <button
                  key={kind}
                  type="button"
                  className={visible ? "memory-graph-control-chip memory-graph-control-chip--active" : "memory-graph-control-chip"}
                  onClick={() => toggleKind(kind)}
                  title={kind}
                >
                  {nodeKindLabels[kind] ?? kind}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className="memory-graph-control-group">
        <span>Preset</span>
        <div className="memory-graph-preset-grid">
          {(Object.keys(MEMORY_GRAPH_LAYOUT_PRESETS) as MemoryGraphLayoutPreset[]).map((preset) => (
            <button
              key={preset}
              type="button"
              className={settings.preset === preset ? "memory-graph-control-chip memory-graph-control-chip--active" : "memory-graph-control-chip"}
              onClick={() => onChange(MEMORY_GRAPH_LAYOUT_PRESETS[preset])}
            >
              {PRESET_LABELS[preset]}
            </button>
          ))}
        </div>
      </div>

      <div className="memory-graph-control-group">
        <span>Labels</span>
        <div className="memory-graph-preset-grid memory-graph-preset-grid--labels">
          {(Object.keys(LABEL_MODE_LABELS) as MemoryGraphLabelMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              className={settings.labelMode === mode ? "memory-graph-control-chip memory-graph-control-chip--active" : "memory-graph-control-chip"}
              onClick={() => update({ labelMode: mode })}
            >
              {LABEL_MODE_LABELS[mode]}
            </button>
          ))}
        </div>
      </div>

      <MemoryGraphSlider label="Label size" value={settings.labelSize} min={0.42} max={1.2} step={0.02} onChange={(value) => update({ labelSize: value })} />
      <MemoryGraphSlider label="Node spacing" value={settings.repulsion} min={0.45} max={1.75} step={0.05} onChange={(value) => update({ repulsion: value })} />
      <MemoryGraphSlider label="Link distance" value={settings.linkDistance} min={0.55} max={1.65} step={0.05} onChange={(value) => update({ linkDistance: value })} />
      <MemoryGraphSlider label="Center force" value={settings.centerForce} min={0.35} max={1.9} step={0.05} onChange={(value) => update({ centerForce: value })} />
      <MemoryGraphSlider label="Cluster force" value={settings.clusterForce} min={0.35} max={1.9} step={0.05} onChange={(value) => update({ clusterForce: value })} />
    </section>
  );
}

function MemoryGraphSlider({
  label,
  max,
  min,
  onChange,
  step,
  value,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="memory-graph-slider-row">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
      <strong>{value.toFixed(2)}</strong>
    </label>
  );
}
