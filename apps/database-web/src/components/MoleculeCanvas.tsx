"use client";

import { useEffect, useRef, useState } from "react";
import { Atom } from "lucide-react";
import SmilesDrawer from "smiles-drawer";

interface MoleculeCanvasProps {
  label: string;
  smiles: string | null | undefined;
}

export function MoleculeCanvas({ label, smiles }: MoleculeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    if (!smiles || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const width = Math.max(300, Math.floor(canvas.getBoundingClientRect().width));
    const height = 210;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(width * scale);
    canvas.height = Math.floor(height * scale);
    canvas.style.height = `${height}px`;
    const drawer = new SmilesDrawer.Drawer({
      width: canvas.width,
      height: canvas.height,
      bondThickness: Math.max(1.2, scale),
      padding: 18 * scale,
      compactDrawing: true,
    });
    SmilesDrawer.parse(
      smiles,
      (tree) => drawer.draw(tree, canvas, "light", false),
      () => setError("Structure preview unavailable for this representation."),
    );
  }, [smiles]);

  return (
    <figure className="molecule-figure">
      <figcaption>
        <Atom size={15} aria-hidden="true" />
        <span>{label}</span>
      </figcaption>
      {smiles ? (
        <>
          <canvas ref={canvasRef} aria-label={`${label} molecular structure`} />
          {error ? <p className="molecule-error">{error}</p> : null}
          <code title={smiles}>{smiles}</code>
        </>
      ) : (
        <div className="molecule-empty">No confirmed SMILES</div>
      )}
    </figure>
  );
}
