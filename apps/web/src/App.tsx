import { useEffect, useState } from "react";
import { AgentWorkspace } from "./components/AgentWorkspace";
import { DatabaseFrame } from "./components/DatabaseFrame";
import { Portal } from "./components/Portal";
import type { AgentType } from "./types";

function usePathname() {
  const [pathname, setPathname] = useState(window.location.pathname);
  useEffect(() => {
    const update = () => setPathname(window.location.pathname);
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);
  return pathname;
}

export function navigate(path: string) {
  if (path === window.location.pathname) {
    return;
  }
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

const routes: Record<string, AgentType> = {
  "/agents/data-mining": "data_mining",
  "/agents/device-modeling": "device_modeling",
  "/agents/experimental-design": "experimental_design"
};

export function App() {
  const pathname = usePathname();
  if (pathname === "/database") {
    return <DatabaseFrame />;
  }
  const agentType = routes[pathname];
  if (agentType) {
    return <AgentWorkspace key={agentType} agentType={agentType} />;
  }
  return <Portal />;
}
