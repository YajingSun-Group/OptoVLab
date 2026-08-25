import {
  AppWindow,
  Atom,
  Database,
  FileSearch,
  FlaskConical,
  LayoutGrid,
  Menu,
  Network,
  X
} from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { navigate } from "../App";
import type { AppSummary } from "../types";
import { Brand } from "./Brand";

const appIcons: Record<string, typeof FileSearch> = {
  "data-mining": FileSearch,
  "device-modeling": Network,
  "experimental-design": FlaskConical,
  "optoelectronics-database": Database
};

const railItems = [
  { label: "Apps overview", icon: LayoutGrid, route: "/" },
  { label: "Data Mining", icon: FileSearch, route: "/agents/data-mining" },
  { label: "Device Modeling", icon: Network, route: "/agents/device-modeling" },
  { label: "Experimental Design", icon: FlaskConical, route: "/agents/experimental-design" },
  { label: "Database", icon: Database, route: "/database" }
];

export function Portal() {
  const [apps, setApps] = useState<AppSummary[]>([]);
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    void api
      .apps()
      .then(setApps)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  const agentApps = apps.filter((app) => app.category === "Research agents");
  const exploreApps = apps.filter((app) => app.category !== "Research agents");

  return (
    <div className="portal-shell">
      <header className="portal-header">
        <Brand />
        <nav aria-label="Primary navigation">
          <button className="nav-link nav-active" onClick={() => navigate("/")}>Apps</button>
          <button className="nav-link" onClick={() => navigate("/agents/data-mining")}>Mining</button>
          <button className="nav-link" onClick={() => navigate("/agents/device-modeling")}>Modeling</button>
          <button className="nav-link" onClick={() => navigate("/agents/experimental-design")}>Experiments</button>
          <button className="nav-link" onClick={() => navigate("/database")}>Database</button>
        </nav>
      </header>

      <aside className={`portal-rail ${menuOpen ? "rail-expanded" : ""}`}>
        <button
          className="rail-menu-button"
          onClick={() => setMenuOpen((current) => !current)}
          title={menuOpen ? "Close app menu" : "Open app menu"}
        >
          {menuOpen ? <X size={22} /> : <Menu size={22} />}
          {menuOpen && <span>Apps Overview</span>}
        </button>
        <div className="rail-section-label">{menuOpen && "RESEARCH AGENTS"}</div>
        {railItems.map(({ label, icon: Icon, route }, index) => (
          <button
            key={label}
            className={`rail-item ${index === 0 ? "rail-item-active" : ""}`}
            title={label}
            onClick={() => {
              setMenuOpen(false);
              navigate(route);
            }}
          >
            <Icon size={22} />
            {menuOpen && <span>{label}</span>}
          </button>
        ))}
      </aside>

      {menuOpen && <button className="rail-scrim" aria-label="Close app menu" onClick={() => setMenuOpen(false)} />}

      <main className="portal-main">
        <section className="overview-header">
          <div>
            <div className="breadcrumbs"><span>Home</span><b>/</b> Apps</div>
            <div className="overview-title">
              <span className="overview-title-icon"><LayoutGrid size={27} /></span>
              <h1>Apps Overview</h1>
            </div>
          </div>
          <div className="overview-actions">
            <button onClick={() => navigate("/agents/data-mining")}><FileSearch size={17} /> Data Mining</button>
            <button onClick={() => navigate("/database")}><Database size={17} /> Database</button>
          </div>
        </section>

        <section className="apps-band">
          {error && <div className="portal-error">Unable to load applications: {error}</div>}
          <AppSection title="Research Agents" apps={agentApps} loading={!apps.length && !error} />
          <AppSection title="Explore and Search" apps={exploreApps} loading={!apps.length && !error} />
          <footer className="portal-footer">
            <span><Atom size={17} /> OptoVLab research infrastructure</span>
            <span>Evidence-backed organic optoelectronics</span>
          </footer>
        </section>
      </main>
    </div>
  );
}

function AppSection({ title, apps, loading }: { title: string; apps: AppSummary[]; loading: boolean }) {
  if (!loading && apps.length === 0) {
    return null;
  }
  return (
    <div className="app-section">
      <h2>{title}</h2>
      <div className="app-grid">
        {loading
          ? Array.from({ length: title === "Research Agents" ? 3 : 1 }, (_, index) => (
              <div className="app-card app-card-loading" key={index} />
            ))
          : apps.map((app) => <AppCard key={app.app_id} app={app} />)}
      </div>
    </div>
  );
}

function AppCard({ app }: { app: AppSummary }) {
  const Icon = appIcons[app.app_id] || AppWindow;
  const metrics = Object.entries(app.metrics).slice(0, 2);
  return (
    <button className="app-card" onClick={() => navigate(app.route)}>
      <Icon className="app-card-icon" size={54} strokeWidth={1.7} />
      <h3>{app.name}</h3>
      <p>{app.description}</p>
      <div className="app-card-metrics">
        {metrics.map(([key, value]) => (
          <span key={key}>{formatMetric(value)} <small>{key.replaceAll("_", " ")}</small></span>
        ))}
      </div>
    </button>
  );
}

function formatMetric(value: unknown) {
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}
