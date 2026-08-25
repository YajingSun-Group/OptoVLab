import { Atom } from "lucide-react";
import { navigate } from "../App";

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <button className={`brand ${compact ? "brand-compact" : ""}`} onClick={() => navigate("/")}>
      <span className="brand-mark" aria-hidden="true">
        <Atom size={compact ? 26 : 34} strokeWidth={1.7} />
        <i />
      </span>
      {!compact && <span>OptoVLab</span>}
    </button>
  );
}
