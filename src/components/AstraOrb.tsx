import "./AstraOrb.css";
import type { AssistantStatus } from "../types/assistant";

type AstraOrbProps = {
  status: AssistantStatus;
};

const ORB_STATE_LABEL: Record<AssistantStatus, string> = {
  idle: "",
  passive: "LIVE",
  armed: "ATTIVA",
  thinking: "ELABORO",
  listening: "ASCOLTO",
  speaking: "VOCE",
};

export default function AstraOrb({ status }: AstraOrbProps) {
  const stateLabel = ORB_STATE_LABEL[status];

  return (
    <div className={`astra-orb-shell ${status}`} aria-label={`Astra ${status}`}>
      <div className="astra-orb-core" />
      <div className="astra-orb-ring ring-1" />
      <div className="astra-orb-ring ring-2" />
      <div className="astra-orb-ring ring-3" />
      <div className="astra-orb-noise" />
      <div className="astra-orb-glow" />
      {stateLabel ? <span className="astra-orb-state-label">{stateLabel}</span> : null}
    </div>
  );
}
