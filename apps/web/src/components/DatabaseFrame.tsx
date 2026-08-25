import { ArrowLeft, ExternalLink } from "lucide-react";
import { navigate } from "../App";
import { Brand } from "./Brand";

export function DatabaseFrame() {
  const configured = import.meta.env.VITE_DATABASE_URL as string | undefined;
  const databaseUrl = configured || `${window.location.protocol}//${window.location.hostname}:3000`;
  return (
    <div className="database-frame-page">
      <header>
        <Brand />
        <div className="database-frame-title">
          <DatabaseLabel />
          <span>Organic Optoelectronics Database</span>
        </div>
        <div className="database-frame-actions">
          <button onClick={() => navigate("/")}><ArrowLeft size={17} /> Apps</button>
          <a href={databaseUrl} target="_blank" rel="noreferrer"><ExternalLink size={17} /> Open separately</a>
        </div>
      </header>
      <iframe src={databaseUrl} title="Organic Optoelectronics Database" />
    </div>
  );
}

function DatabaseLabel() {
  return <span className="database-label">OLED / OFET / OPV</span>;
}
