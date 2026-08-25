import { ExternalLink, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";
import type { PDFDocumentProxy } from "pdfjs-dist/legacy/build/pdf.mjs";

interface PdfRenderTask {
  cancel: () => void;
  promise: Promise<unknown>;
}

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/legacy/build/pdf.worker.mjs",
  import.meta.url
).toString();

export function PdfPreview({ url, title }: { url: string; title: string }) {
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [status, setStatus] = useState("Loading PDF");
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    let cancelled = false;
    const task = pdfjs.getDocument({ url, rangeChunkSize: 1024 * 1024 });
    setDocument(null);
    setStatus("Loading PDF");
    void task.promise
      .then((loaded) => {
        if (cancelled) return;
        setDocument(loaded);
        setStatus(`${loaded.numPages} pages`);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setStatus(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
      void task.destroy();
    };
  }, [url]);

  return (
    <section className="compact-pdf-viewer">
      <header>
        <div><strong>{title}</strong><span>{status}</span></div>
        <div>
          <button onClick={() => setZoom((value) => Math.max(.7, value - .1))} title="Zoom out"><ZoomOut size={15} /></button>
          <span>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom((value) => Math.min(1.8, value + .1))} title="Zoom in"><ZoomIn size={15} /></button>
          <a href={url} target="_blank" rel="noreferrer" title="Open PDF"><ExternalLink size={15} /></a>
        </div>
      </header>
      <div className="compact-pdf-pages">
        {document ? Array.from({ length: document.numPages }, (_, index) => (
          <PdfPage key={`${index + 1}-${zoom}`} document={document} pageNumber={index + 1} zoom={zoom} />
        )) : <div className="pdf-loading-state">{status}</div>}
      </div>
    </section>
  );
}

function PdfPage({ document, pageNumber, zoom }: { document: PDFDocumentProxy; pageNumber: number; zoom: number }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [width, setWidth] = useState(0);
  const [visible, setVisible] = useState(pageNumber <= 2);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const resize = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    resize.observe(host);
    const intersection = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisible(true);
          intersection.disconnect();
        }
      },
      { rootMargin: "800px 0px" }
    );
    intersection.observe(host);
    return () => {
      resize.disconnect();
      intersection.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!visible || !width) return;
    let cancelled = false;
    let renderTask: PdfRenderTask | null = null;
    void document.getPage(pageNumber).then((page) => {
      if (cancelled) return;
      const base = page.getViewport({ scale: 1 });
      const scale = Math.max(.25, (width / base.width) * zoom);
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (!canvas || !context) return;
      const outputScale = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      context.setTransform(outputScale, 0, 0, outputScale, 0, 0);
      renderTask = page.render({ canvas, canvasContext: context, viewport }) as PdfRenderTask;
      renderTask?.promise.catch(() => undefined);
    });
    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
  }, [document, pageNumber, visible, width, zoom]);

  return <div className="compact-pdf-page" ref={hostRef}><span>Page {pageNumber}</span><canvas ref={canvasRef} /></div>;
}
