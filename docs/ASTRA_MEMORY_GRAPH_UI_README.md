# AstraOS Memory Graph UI v0.5.3

Questo pacchetto aggiunge il primo pannello visuale del Cognitive Memory Graph.

## Obiettivo

Rendere visibile la memoria relazionale di AstraOS come grafo:

- nodi memoria generati da Work Session / recap / tool outcome
- relazioni tra nodi
- timeline delle attivazioni
- ricerca nel Memory Graph tramite i comandi Tauri gia' disponibili
- highlight dei nodi attivati dall'evento `memory-activation`
- effetto visivo tipo “elettricita'” sugli archi attivi

## File inclusi

- `src/components/DesktopAgentPanel.tsx`
- `src/hooks/useMemoryGraph.ts`
- `src/types/memory.ts`
- `src/features/memory/index.ts`
- `src/features/memory/components/MemoryGraphPanel.tsx`
- `src/App.css`

## Note architetturali

La UI non decide nulla e non bypassa la governance Rust. Usa solo comandi read/query gia' esposti dal runtime:

- `get_memory_graph_status`
- `export_memory_graph_snapshot`
- `query_memory_graph`
- `get_recent_memory_activations`

La memoria resta governata lato Rust. Il frontend visualizza lo stato, le relazioni e le attivazioni.

## Validazione

Dopo l'import:

```powershell
npm run build
cd src-tauri
cargo check
```

Poi apri il Desktop Agent Panel e usa il tab `memory`.
