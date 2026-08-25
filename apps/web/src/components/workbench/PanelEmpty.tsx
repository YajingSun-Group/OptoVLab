import type { LucideIcon } from "lucide-react";

export function PanelEmpty({
  icon: Icon,
  title,
  detail
}: {
  icon: LucideIcon;
  title: string;
  detail: string;
}) {
  return (
    <div className="workbench-empty">
      <Icon size={26} />
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}
