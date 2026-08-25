from __future__ import annotations

from evolab_local.optovlab.repository import OptoVLabRepository


DEVICE_MODELING_DEMO_SEED = "device_modeling_v1"
DATA_MINING_ANALYSIS_DEMO_SEED = "data_mining_database_analysis_v1"
DATA_MINING_WORKFLOW_DEMO_SEED = "data_mining_workflow_v1"
DATA_MINING_WORKFLOW_DEMO_PROMPT = (
    "Mine OLED device and material data from the uploaded paper."
)
DATA_MINING_ANALYSIS_DEMO_PROMPT = (
    "Analyze the full OLED database and summarize its overall device composition."
)

DEVICE_MODELING_DEMO_MESSAGES = (
    (
        "user",
        "Use the current OLED device corpus to design a graph model for maximum EQE. "
        "Show the device representation, validation strategy, and a safe training plan.",
    ),
    (
        "assistant",
        """## OLED-GAT device modeling demo

I mapped each OLED stack to an **ordered directed graph** instead of flattening it into one feature vector.

- **Nodes:** substrate, electrodes, transport layers, blocking layers, and EML components.
- **Directed edges:** adjacent interfaces follow the physical layer order from anode to cathode; reverse message flow can be enabled as a separate edge type.
- **EML representation:** molecular features from available SMILES are fused with host/dopant role and composition.
- **Other layers:** controlled layer roles, material identity, thickness, and available process descriptors.
- **Prediction:** quantile heads estimate lower, median, and upper `EQE_max`, so experimental uncertainty is visible.

For evaluation, I would split by **paper or device family**, fit normalization on the training partition only, and report median-prediction R2/MAE together with interval coverage. Attention weights can identify interfaces associated with predictions, but they should be treated as statistical importance rather than proof of a physical mechanism.

The **Models**, **HPC**, and **Training** tabs in Workbench expose the current registry, live GPU/Slurm state, and a guarded training form.""",
    ),
    (
        "user",
        "Prepare a one-GPU OLED-GAT run, but do not submit a Slurm job without explicit approval.",
    ),
    (
        "assistant",
        """The demonstration run is configured with one GPU and the current campaign YAML. The generated Slurm script remains in **prepared** state.

No scheduler job is submitted until **Submit to Slurm** is explicitly enabled and confirmed in Workbench. This keeps model inspection and script generation safe for demonstrations.""",
    ),
)


def seed_demo_content(repository: OptoVLabRepository) -> int:
    """Add versioned demo messages to an existing Device Modeling demo session."""
    session = next(
        (
            item
            for item in repository.list_sessions("device_modeling", limit=500)
            if item.title == "Device Modeling demo"
        ),
        None,
    )
    if session is None:
        return 0
    if any(
        message.metadata.get("demo_seed") == DEVICE_MODELING_DEMO_SEED
        for message in repository.list_messages(session.session_id)
    ):
        return 0

    for sequence, (role, content) in enumerate(DEVICE_MODELING_DEMO_MESSAGES, start=1):
        repository.add_message(
            session.session_id,
            role,
            content,
            metadata={
                "agent_type": "device_modeling",
                "demo_seed": DEVICE_MODELING_DEMO_SEED,
                "sequence": sequence,
            },
            touch_session=False,
        )

    existing_event_markers = {
        event.payload.get("demo_seed") for event in repository.list_tool_events(session.session_id)
    }
    if DEVICE_MODELING_DEMO_SEED not in existing_event_markers:
        for tool_name, title, detail in (
            (
                "oled_dataset_summary",
                "Inspected the model-ready OLED corpus",
                "Prepared paper-grouped validation and leakage-safe normalization guidance.",
            ),
            (
                "oled_gat_model_registry",
                "Compared registered OLED-GAT configurations",
                "Loaded graph architecture, split policy, quantile heads, and saved metrics.",
            ),
            (
                "hpc_status",
                "Checked Slurm and local GPU availability",
                "Read-only infrastructure inspection; no scheduler job was submitted.",
            ),
        ):
            repository.add_tool_event(
                session.session_id,
                tool_name,
                "completed",
                title,
                detail=detail,
                payload={"demo_seed": DEVICE_MODELING_DEMO_SEED},
            )
    return len(DEVICE_MODELING_DEMO_MESSAGES)
