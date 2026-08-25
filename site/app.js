const agents = {
  mining: {
    eyebrow: "DATA MINING AGENT",
    title: "Evidence-backed OLED extraction",
    status: "Completed",
    summary:
      "The agent converted a literature PDF into device-grouped records, preserved verbatim evidence anchors, and routed uncertain molecular structures to review.",
    metrics: [
      ["Devices", "4"],
      ["Materials", "14"],
      ["Evidence", "18"]
    ],
    critic:
      "Accepted the extraction plan. Two material structures remain explicitly unresolved rather than inferred.",
    stages: [
      ["PDF intake", "Validated input", "done"],
      ["Document parse", "MinerU blocks", "done"],
      ["Device mining", "Schema constrained", "done"],
      ["Material identity", "Cross-validated", "done"],
      ["Human review", "Evidence linked", "active"]
    ],
    tools: [
      ["PDF validation", "10 pages · checksum recorded"],
      ["MinerU", "Text, tables, captions, and figures parsed"],
      ["DeepSeek", "OLED device schema extracted"],
      ["PubChem + web search", "Known materials resolved"],
      ["DECIMER", "Eligible 2D structures converted to SMILES"]
    ],
    visual: "mining"
  },
  modeling: {
    eyebrow: "DEVICE MODELING AGENT",
    title: "Directed OLED graph and quantile EQE",
    status: "Evaluated",
    summary:
      "OLED-GAT represents ordered functional layers as nodes and adjacent interfaces as directed edges, then predicts the 10th, 50th, and 90th EQE quantiles.",
    metrics: [
      ["Test R²", "0.81"],
      ["MAE", "2.56"],
      ["Interval", "Q10–Q90"]
    ],
    critic:
      "Grouped evaluation and leakage controls passed. Attention is treated as a hypothesis-generating importance signal, not a causal proof.",
    stages: [
      ["Data audit", "Grouped split", "done"],
      ["Graph build", "Ordered layers", "done"],
      ["Message passing", "Interface aware", "done"],
      ["Quantile head", "Q10 / Q50 / Q90", "done"],
      ["Interpretation", "Attention audit", "active"]
    ],
    tools: [
      ["Dataset inspector", "Schema and leakage checks completed"],
      ["Graph builder", "Layer nodes and directed interfaces created"],
      ["OLED-GAT", "Multi-head message passing evaluated"],
      ["Quantile regression", "Prediction interval calibrated"],
      ["Attention audit", "Layer and interface attributions exported"]
    ],
    visual: "modeling"
  },
  experiment: {
    eyebrow: "EXPERIMENTAL DESIGN AGENT",
    title: "Critic-reviewed closed-loop optimization",
    status: "Validated",
    summary:
      "The agent diagnosed hole leakage, proposed a deeper-HOMO blocking layer, and converged to a PPF-containing 4CzIPN device within four experimental rounds.",
    metrics: [
      ["Baseline", "18.6%"],
      ["Optimized", "26.5%"],
      ["Gain", "+42%"]
    ],
    critic:
      "Approved the PPF experiment after checking physical rationale, literature precedent, and an explicit control against the baseline topology.",
    stages: [
      ["Retrieve", "Device precedents", "done"],
      ["Diagnose", "Hole leakage", "done"],
      ["Propose", "PPF blocking layer", "done"],
      ["Critique", "Controls checked", "done"],
      ["Experiment", "Round 4 validated", "active"]
    ],
    tools: [
      ["OLED device RAG", "Relevant 4CzIPN precedents retrieved"],
      ["Physics diagnosis", "mCP / transport interface flagged"],
      ["Candidate ranking", "Deep-HOMO HBL options compared"],
      ["Scientific Critic", "Hypothesis and controls approved"],
      ["Human experiment", "Measured EQE returned to the loop"]
    ],
    visual: "experiment"
  }
};

const tabs = document.querySelectorAll(".agent-tab");
const stageList = document.querySelector("#stage-list");
const toolEvents = document.querySelector("#tool-events");
const agentVisual = document.querySelector("#agent-visual");

function renderAgent(key) {
  const agent = agents[key];
  document.querySelector("#agent-eyebrow").textContent = agent.eyebrow;
  document.querySelector("#agent-title").textContent = agent.title;
  document.querySelector("#agent-status").textContent = agent.status;
  document.querySelector("#agent-summary").textContent = agent.summary;
  document.querySelector("#critic-text").textContent = agent.critic;
  document.querySelector("#tool-count").textContent = `${agent.tools.length} tools`;

  stageList.replaceChildren(
    ...agent.stages.map(([title, detail, state], index) => {
      const button = document.createElement("div");
      button.className = `stage-button ${state}`;
      button.innerHTML = `<span class="stage-marker">${state === "done" ? "✓" : index + 1}</span><span><strong>${title}</strong><span>${detail}</span></span>`;
      return button;
    })
  );

  toolEvents.replaceChildren(
    ...agent.tools.map(([title, detail]) => {
      const item = document.createElement("li");
      item.className = "tool-event";
      item.innerHTML = `<span class="event-check">✓</span><strong>${title}</strong><span>${detail}</span>`;
      return item;
    })
  );

  const metrics = document.querySelector("#agent-metrics");
  metrics.replaceChildren(
    ...agent.metrics.map(([label, value]) => {
      const group = document.createElement("div");
      group.innerHTML = `<dt>${label}</dt><dd>${value}</dd>`;
      return group;
    })
  );

  if (agent.visual === "mining") renderMining();
  if (agent.visual === "modeling") renderModeling();
  if (agent.visual === "experiment") renderExperiment();
}

function renderMining() {
  agentVisual.innerHTML = `
    <div class="document-flow" aria-label="PDF mining workflow">
      <div class="flow-node"><span>PDF</span><strong>Literature input</strong><p>Text, tables, figures, captions, and page coordinates.</p><code>angew-test.pdf · 10 pages</code></div>
      <div class="flow-arrow" aria-hidden="true">→</div>
      <div class="flow-node"><span>AI</span><strong>Multimodal mining</strong><p>Schema-constrained device extraction plus material identity and OCSR tools.</p><code>MinerU → DeepSeek → search → DECIMER</code></div>
      <div class="flow-arrow" aria-hidden="true">→</div>
      <div class="flow-node"><span>{ }</span><strong>Reviewed records</strong><p>Devices, ordered layers, performance, materials, and evidence anchors.</p><code>devices[4] · materials[14]</code></div>
    </div>`;
}

function renderModeling() {
  const nodes = [
    ["Anode", "ITO", 10, "#dce6f4"],
    ["HTL", "NPB", 27, "#d6ede7"],
    ["EML", "host:emitter", 45, "#fff0bd"],
    ["ETL", "BPhen", 63, "#f8dcd4"],
    ["Cathode", "LiF/Al", 80, "#dbe2ed"]
  ];
  const edges = [
    [36, 18, 2, "#7f98b9", "0.18"],
    [54, 18, 6, "#d76752", "0.42"],
    [71.5, 17, 3, "#178f86", "0.24"]
  ];
  agentVisual.innerHTML = `
    <div class="device-graph" aria-label="Directed OLED device graph">
      <div class="graph-line"></div>
      ${edges.map(([left, width, weight, color, label]) => `<div class="attention-edge" style="left:${left}%;width:${width}%;--edge-width:${weight}px;--edge-color:${color}"><span>attention ${label}</span></div>`).join("")}
      ${nodes.map(([role, material, left, color]) => `<div class="layer-node" style="left:${left}%;--node-color:${color}">${role}<small>${material}</small></div>`).join("")}
      <div class="quantiles"><span><strong>21.2%</strong>Q10</span><span><strong>23.4%</strong>Q50</span><span><strong>25.6%</strong>Q90</span></div>
    </div>`;
}

function renderExperiment() {
  const rounds = [
    ["Round 1", "18.6%", "Baseline topology", "#7f98b9", "42%"],
    ["Round 2", "21.1%", "Charge balance adjustment", "#386cb0", "57%"],
    ["Round 3", "23.8%", "HBL candidate screening", "#178f86", "72%"],
    ["Round 4", "26.5%", "PPF blocking layer", "#d76752", "92%"]
  ];
  agentVisual.innerHTML = `
    <div class="experiment-flow" aria-label="Four-round OLED optimization">
      <div class="round-track">
        ${rounds.map(([round, eqe, note, color, height]) => `<div class="round" style="--round-color:${color};--height:${height}"><small>${round}</small><strong>${eqe}</strong><span>${note}</span></div>`).join("")}
      </div>
      <div class="decision-line"><span>Agent proposal → Scientific Critic → Human approval → Experiment</span><b>Measured gain: +42%</b></div>
    </div>`;
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    renderAgent(tab.dataset.agent);
  });
});

const dialog = document.querySelector("#figure-dialog");
const dialogImage = document.querySelector("#dialog-image");
const dialogCaption = document.querySelector("#dialog-caption");

document.querySelectorAll("[data-image]").forEach((button) => {
  button.addEventListener("click", () => {
    dialogImage.src = button.dataset.image;
    dialogCaption.textContent = button.dataset.caption || "Scientific figure";
    dialog.showModal();
  });
});
document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

renderAgent("mining");
