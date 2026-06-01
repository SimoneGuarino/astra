export type GraphPoint = {
  id: string;
  x: number;
  y: number;
  radius: number;
  activated: boolean;
  intensity: number;
};

export type GraphPosition = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  cluster: number;
};

export type GraphViewport = {
  scale: number;
  tx: number;
  ty: number;
};

export type GraphSurfaceSize = {
  width: number;
  height: number;
};

export type GraphCluster = {
  id: number;
  nodeIds: string[];
  x: number;
  y: number;
};

export type GraphPerformanceStats = {
  visibleNodes: number;
  visibleEdges: number;
  renderedLabels: number;
  culledLabels: number;
  simulationState: "running" | "settled" | "paused";
  lastTickMs: number;
};

export type GraphSurfaceProjection = {
  scale: number;
  offsetX: number;
  offsetY: number;
  viewport: GraphViewport;
};

export const GRAPH_WIDTH = 2600;
export const GRAPH_HEIGHT = 1600;
export const GRAPH_CENTER_X = GRAPH_WIDTH / 2;
export const GRAPH_CENTER_Y = GRAPH_HEIGHT / 2;
export const MAX_VISIBLE_GRAPH_NODES = 420;
export const MAX_VISIBLE_GRAPH_EDGES = 900;
export const MAX_RENDERED_LABELS = 140;
export const LARGE_GRAPH_NODE_THRESHOLD = 180;
export const FORCE_SPATIAL_CELL_SIZE = 280;
export const FORCE_STABLE_VELOCITY = 0.018;
export const FORCE_STABLE_TICKS = 90;

export const NODE_KIND_LABELS: Record<string, string> = {
  work_session: "Work Session",
  summary: "Summary",
  transcript_segment: "Transcript",
  tool_use: "Tool",
  concept: "Concept",
  entity: "Entity",
  user_preference: "Preference",
  procedure: "Procedure",
  research_topic: "Research Topic",
  research_finding: "Finding",
  source_document: "Source",
  claim: "Claim",
  decision: "Decision",
  error: "Error",
  fix: "Fix",
  workflow: "Workflow",
  code_pattern: "Code Pattern",
  conversation_turn: "Conversation",
};

export const NODE_KIND_PRIORITY: Record<string, number> = {
  user_preference: 18,
  procedure: 16,
  workflow: 15,
  research_topic: 14,
  concept: 13,
  entity: 12,
  decision: 12,
  summary: 11,
  fix: 11,
  error: 10,
  work_session: 9,
  tool_use: 8,
  transcript_segment: 5,
  conversation_turn: 4,
};
