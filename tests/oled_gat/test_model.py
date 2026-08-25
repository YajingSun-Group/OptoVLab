from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from oled_gat.explainability import forward_with_attention
from oled_gat.model import OLEDGATQuantile, build_oled_gat, quantile_hybrid_loss


def _vocabulary() -> dict[str, object]:
    return {
        "layer_roles": ["<pad>", "<oov>", "eml"],
        "component_roles": ["<pad>", "<oov>", "emitter"],
        "materials": ["<pad>", "<oov>", "material"],
        "mechanisms": ["<pad>", "<oov>", "tadf"],
        "colors": ["<pad>", "<oov>", "green"],
        "processes": ["<pad>", "<oov>", "vacuum"],
        "device_types": ["<pad>", "<oov>", "bottom"],
        "edge_types": ["root_to_layer"],
        "molecular_fingerprint_size": 8,
        "numeric_stats": {
            "molecular_descriptor_mean": [0.0] * 12,
        },
    }


def _graph() -> Data:
    return Data(
        node_type=torch.tensor([0, 1, 2, 3]),
        layer_role=torch.tensor([0, 2, 0, 0]),
        component_role=torch.tensor([0, 0, 2, 0]),
        material_id=torch.tensor([0, 0, 2, 0]),
        atomic_number=torch.tensor([0, 0, 0, 6]),
        degree=torch.tensor([0, 0, 0, 1]),
        formal_charge=torch.tensor([4, 4, 4, 4]),
        hybridization=torch.tensor([0, 0, 0, 4]),
        chirality=torch.tensor([0, 0, 0, 1]),
        node_numeric=torch.tensor(
            [
                [0.0] * 10,
                [0.0] * 9 + [1.0],
                [0.0] * 9 + [1.0],
                [0.0] * 9 + [1.0],
            ]
        ),
        molecular_features=torch.zeros(4, 20),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
        ),
        edge_type=torch.zeros(6, dtype=torch.long),
        mechanism_id=torch.tensor([2]),
        color_id=torch.tensor([2]),
        process_id=torch.tensor([2]),
        device_type_id=torch.tensor([2]),
        context_numeric=torch.zeros(1, 2),
        y=torch.tensor([0.5]),
        y_raw=torch.tensor([15.0]),
        sample_weight=torch.tensor([1.0]),
        sample_row=torch.tensor([0]),
    )


def test_quantiles_are_non_crossing() -> None:
    model = OLEDGATQuantile(
        _vocabulary(),
        hidden_dim=32,
        attention_heads=4,
        message_passing_layers=2,
        edge_dim=8,
        dropout=0.1,
    )
    batch = Batch.from_data_list([_graph(), _graph()])
    output = model(batch)

    assert model.readout[0].in_features == 32 * 5 + 76
    assert output.shape == (2, 4)
    assert torch.all(output[:, 1] <= output[:, 2])
    assert torch.all(output[:, 2] <= output[:, 3])
    loss = quantile_hybrid_loss(
        output,
        batch.y,
        batch.sample_weight,
        quantiles=(0.1, 0.5, 0.9),
        mean_mse_weight=1.0,
        quantile_loss_weight=0.35,
    )
    assert torch.isfinite(loss)


def test_attention_forward_matches_standard_forward_and_keeps_gradients() -> None:
    model = OLEDGATQuantile(
        _vocabulary(),
        hidden_dim=32,
        attention_heads=4,
        message_passing_layers=2,
        edge_dim=8,
        dropout=0.1,
    )
    model.eval()
    batch = Batch.from_data_list([_graph()])
    batch.node_numeric = batch.node_numeric.detach().clone().requires_grad_(True)

    expected = model(batch)
    explained = forward_with_attention(
        model,
        batch,
        retain_attention_gradients=True,
    )
    explained.output[:, 0].sum().backward()

    assert torch.allclose(explained.output, expected)
    assert len(explained.attention) == 2
    assert all(
        block.coefficients.grad is not None
        for block in explained.attention
    )
    assert batch.node_numeric.grad is not None


def test_hierarchical_model_preserves_non_crossing_output_contract() -> None:
    model = build_oled_gat(
        _vocabulary(),
        {
            "architecture": "hierarchical_device_gat_v2",
            "hidden_dim": 32,
            "attention_heads": 4,
            "atom_message_passing_layers": 2,
            "device_message_passing_layers": 2,
            "edge_dim": 8,
            "dropout": 0.1,
        },
    )
    output = model(Batch.from_data_list([_graph(), _graph()]))

    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
    assert torch.all(output[:, 1] <= output[:, 2])
    assert torch.all(output[:, 2] <= output[:, 3])
