"""Adapters around the frozen upstream models, without future-field inputs."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .runtime import measured


def tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def make_model(config: dict, stage: str, device: torch.device):
    method = config["method"]
    if method == "mgn":
        from modulus.models.meshgraphnet import MeshGraphNet

        model = MeshGraphNet(6, 3, 3)
    elif method == "eagle":
        from Models.graphViT import GraphViT

        model = GraphViT(state_size=4, w_size=512)
    elif method == "aroma" and stage == "ae":
        from aroma.encoder_decoder import AROMAEncoderDecoderKL

        model = AROMAEncoderDecoderKL(
            input_dim=2,
            num_channels=3,
            hidden_dim=128,
            dim=64,
            num_self_attentions=3,
            num_latents=32,
            latent_dim=8,
            latent_heads=4,
            latent_dim_head=32,
            cross_heads=4,
            cross_dim_head=32,
            scales=[3, 4, 5],
            num_freq=12,
            depth_inr=3,
            bottleneck_index=0,
            encode_geo=True,
            mlp_feature_dim=16,
            max_pos_encoding_freq=4,
        )
    elif method == "aroma" and stage == "dynamics":
        from aroma.DIT_conditionned import DiT

        model = DiT(
            input_size=8,
            num_tokens=32,
            in_channels=4,
            hidden_size=128,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            learn_sigma=False,
        )
    else:
        raise ValueError(f"unsupported method/stage: {method}/{stage}")
    return model.to(device)


def aroma_scheduler():
    from diffusers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=4,
        trained_betas=[1e-3 ** (k / 3) for k in (3, 2, 1, 0)],
        prediction_type="v_prediction",
        clip_sample=False,
    )


def aroma_inputs(field, points, stats, device):
    values = (tensor(field, device) - tensor(stats["mean"][:3], device)) / tensor(
        stats["std"][:3], device
    )
    # This is the upstream CylinderFlow coordinate transform, not a machine path.
    coords = (tensor(points, device) / 1.6 - 0.5) * 2
    return values.unsqueeze(0), coords.unsqueeze(0)


def prepare_clusters(dataset, directory: Path, seed: int) -> None:
    """Reuse the author's capacity-constrained geometric clustering."""
    from clusterize_eagle import constrained_clustering_numpy
    from .runtime import write_json

    directory.mkdir(exist_ok=False)
    for index in dataset.splits["train"] + dataset.splits["validation"]:
        points = dataset.static(index)["points"]
        np.random.seed(seed + index)
        capacity = 10
        cluster_count = int(np.ceil(len(points) / capacity)) + 1
        # One extra ghost cluster is native; it must not request more centers
        # than physical nodes in the tiny CPU fixture.
        cluster_count = min(cluster_count, len(points))
        labels, _ = constrained_clustering_numpy(points, None, cluster_count, capacity)
        if len(labels) != len(points) or not np.isfinite(labels).all():
            raise ValueError(f"invalid EAGLE clustering for trajectory {index}")
        clusters = np.full((cluster_count, capacity), -1, dtype=np.int64)
        for number in range(cluster_count):
            members = np.flatnonzero(labels == number)
            if len(members) > capacity:
                raise ValueError("upstream clustering exceeded cluster capacity")
            clusters[number, : len(members)] = members
        covered = clusters[clusters >= 0]
        if not np.array_equal(np.sort(covered), np.arange(len(points))):
            raise ValueError("clusters do not cover every node exactly once")
        np.save(directory / f"trajectory_{index:04d}.npy", clusters)
    write_json(
        directory / "identity.json",
        {
            "algorithm": "upstream_constrained_clustering_numpy",
            "capacity": 10,
            "seed": seed,
            "data_identity": dataset.identity(),
        },
    )


def eagle_inputs(sample, stats, prepared: Path, device, initial_only: bool = False):
    points = sample["points"]
    n = len(points)
    uvp = (
        np.repeat(sample["initial"][None], 65, axis=0)
        if initial_only
        else sample["field"]
    )
    q = 0.5 * np.sum(uvp[..., :2] ** 2, axis=-1, keepdims=True)
    physical = np.concatenate([uvp, q], axis=-1)
    state = (tensor(physical, device) - tensor(stats["mean"], device)) / tensor(
        stats["std"], device
    )
    state = F.pad(state, (0, 0, 0, 1)).unsqueeze(0)
    length = state.shape[1]
    mesh = F.pad(tensor(points, device), (0, 0, 0, 1))[None, None].expand(
        1, length, -1, -1
    )
    edges = np.concatenate([sample["edges"], np.array([[n, n]])])
    edges = torch.as_tensor(edges, dtype=torch.long, device=device)[None, None].expand(
        1, length, -1, -1
    )
    labels = torch.as_tensor(
        np.append(sample["node_type"], 2), dtype=torch.long, device=device
    )
    types = F.one_hot(labels, 9).float()[None, None].expand(1, length, -1, -1)
    clusters = np.load(prepared / "clusters" / f"trajectory_{sample['index']:04d}.npy")
    mask = clusters >= 0
    clusters = np.where(mask, clusters, n)
    clusters = torch.as_tensor(clusters, dtype=torch.long, device=device)[
        None, None
    ].expand(1, length, -1, -1)
    mask = torch.as_tensor(mask, device=device)[None, None].expand(1, length, -1, -1)
    return mesh, edges, state, types, clusters, mask


def mgn_inputs(sample, velocity, stats, device):
    import dgl

    edges = sample["edges"]
    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(sample["points"])).to(
        device
    )
    classes = np.asarray(
        [{0: 0, 4: 1, 5: 2, 6: 3}[int(label)] for label in sample["node_type"]]
    )
    onehot = F.one_hot(torch.as_tensor(classes, device=device), 4).float()
    normalized = (velocity - tensor(stats["velocity_mean"], device)) / tensor(
        stats["velocity_std"], device
    )
    features = (
        tensor(sample["edge_features"], device) - tensor(stats["edge_mean"], device)
    ) / tensor(stats["edge_std"], device)
    return graph, torch.cat([normalized, onehot], dim=-1), features


def training_loss(
    model,
    config,
    stage,
    units,
    dataset,
    stats,
    prepared,
    device,
    group_nodes: int,
    group_samples: int,
    latent_cache=None,
):
    """Return losses already weighted to the complete effective batch."""
    method = config["method"]
    total = torch.zeros((), device=device)
    components = {}
    if method == "aroma" and stage == "dynamics":
        scheduler = aroma_scheduler()
        current = torch.stack(
            [
                tensor(latent_cache[f"trajectory_{index:04d}"][step], device)
                for index, step in units
            ]
        )
        target_next = torch.stack(
            [
                tensor(latent_cache[f"trajectory_{index:04d}"][step + 1], device)
                for index, step in units
            ]
        )
        k = torch.randint(0, 4, (len(units),), device=device)
        noise = torch.randn_like(target_next)
        noised = scheduler.add_noise(target_next, noise, k)
        prediction = model(torch.cat([current, noised], dim=1), k * (1000 / 3))
        alpha = scheduler.alphas_cumprod.to(device)[k][:, None, None]
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * target_next
        total = F.mse_loss(prediction, target) * (len(units) / group_samples)
        return total, {"dynamics_v_mse": float(total.detach())}
    for index, step in units:
        horizon = 6 if method == "eagle" else 2 if method == "mgn" else 1
        sample = dataset.read(index, step, step + horizon)
        n = len(sample["points"])
        if method == "mgn":
            field = tensor(sample["field"], device)
            noise = torch.randn_like(field[0, :, :2]) * config["noise_std"]
            mask = torch.as_tensor(sample["node_type"] == 0, device=device)
            noise = torch.where(mask[:, None], noise, torch.zeros_like(noise))
            graph, nodes, edges = mgn_inputs(
                sample, field[0, :, :2] + noise, stats, device
            )
            expected = torch.cat(
                [
                    (
                        field[1, :, :2]
                        - field[0, :, :2]
                        - noise
                        - tensor(stats["velocity_diff_mean"], device)
                    )
                    / tensor(stats["velocity_diff_std"], device),
                    (field[1, :, 2:3] - tensor(stats["pressure_mean"], device))
                    / tensor(stats["pressure_std"], device),
                ],
                dim=-1,
            )
            loss = F.mse_loss(model(graph, nodes, edges), expected) / group_samples
            terms = {"normalized_transition_mse": loss}
        elif method == "eagle":
            args = eagle_inputs(sample, stats, prepared, device)
            _, output, target = model(
                *args,
                apply_noise=True,
                boundary_values=args[2][:, 0],
                boundary_channels=(0, 1),
            )
            errors = (output[:, :, :n] - target[:, :, :n]) ** 2
            uv = errors[..., :2].sum() / (group_nodes * 5 * 2)
            pressure = errors[..., 2:].sum() / (group_nodes * 5 * 2)
            terms = {
                "velocity_delta_mse": uv,
                "pressure_delta_mse_weighted": 0.1 * pressure,
            }
        else:
            from aroma.encoder_decoder import dropout_seq

            values, coords = aroma_inputs(
                sample["field"][0], sample["points"], stats, device
            )
            mask = torch.ones(values.shape[:2], dtype=torch.bool, device=device)
            images, subset_coords, subset_mask = dropout_seq(
                values, coords, mask, dropout=random.choice([0, 0.25, 0.5, 0.75])
            )
            output, kl = model(images, subset_coords, subset_mask, target_coords=coords)
            terms = {
                "reconstruction_mse": ((output - values) ** 2).sum()
                / (group_nodes * 3),
                "kl_weighted": 1e-5 * kl / group_samples,
            }
        for key, value in terms.items():
            total = total + value
            components[key] = components.get(key, 0.0) + float(value.detach())
    return total, components


@torch.no_grad()
def rollout(
    model, config, initial, stats, prepared, device, autoencoder=None, diagnostics=None
):
    """Predict 64 future states; this function has no future-reference argument."""
    method = config["method"]
    model.eval()
    timings = {
        "encode_seconds": 0.0,
        "model_seconds": 0.0,
        "decode_seconds": 0.0,
        "preprocess_seconds": 0.0,
    }
    raw = []
    if method == "eagle":
        with measured(timings, "preprocess_seconds", device):
            args = eagle_inputs(initial, stats, prepared, device, initial_only=True)
        with measured(timings, "model_seconds", device):
            states, _, _, raw_states = model(
                *args,
                boundary_values=args[2][:, 0],
                boundary_channels=(0, 1),
                return_boundary_raw=True,
            )
        with measured(timings, "decode_seconds", device):
            n = len(initial["points"])
            states = states[0, :, :n] * tensor(stats["std"], device) + tensor(
                stats["mean"], device
            )
            raw_states = raw_states[0, :, :n] * tensor(stats["std"], device) + tensor(
                stats["mean"], device
            )
            prediction, raw = (
                states[..., :3].cpu().numpy(),
                raw_states[..., :3].cpu().numpy(),
            )
            if diagnostics is not None:
                diagnostics["eagle_state_uvpq"] = states.cpu().numpy()
                diagnostics["eagle_pre_boundary_uvpq"] = raw_states.cpu().numpy()
    elif method == "mgn":
        current = tensor(initial["initial"][:, :2], device)
        boundary = torch.as_tensor(np.isin(initial["node_type"], [4, 6]), device=device)
        initial_uv = current.clone()
        with measured(timings, "preprocess_seconds", device):
            graph, nodes, edges = mgn_inputs(initial, current, stats, device)
            onehot = nodes[:, 2:]
        states = [tensor(initial["initial"], device)]
        raw = [states[0]]
        with measured(timings, "model_seconds", device):
            for _ in range(64):
                nodes = torch.cat(
                    [
                        (current - tensor(stats["velocity_mean"], device))
                        / tensor(stats["velocity_std"], device),
                        onehot,
                    ],
                    dim=-1,
                )
                output = model(graph, nodes, edges)
                uv = (
                    current
                    + output[:, :2] * tensor(stats["velocity_diff_std"], device)
                    + tensor(stats["velocity_diff_mean"], device)
                )
                pressure = output[:, 2:3] * tensor(
                    stats["pressure_std"], device
                ) + tensor(stats["pressure_mean"], device)
                raw.append(torch.cat([uv, pressure], dim=-1))
                current = torch.where(boundary[:, None], initial_uv, uv)
                states.append(torch.cat([current, pressure], dim=-1))
        with measured(timings, "decode_seconds", device):
            prediction = torch.stack(states).cpu().numpy()
            raw = torch.stack(raw).cpu().numpy()
    else:
        if autoencoder is None:
            raise ValueError("AROMA rollout requires the locked AE")
        autoencoder.eval()
        with measured(timings, "encode_seconds", device):
            values, coords = aroma_inputs(
                initial["initial"], initial["points"], stats, device
            )
            current, _ = autoencoder.encode(values, coords)
        scheduler = aroma_scheduler()
        latents = []
        with measured(timings, "model_seconds", device):
            for _ in range(64):
                candidate = torch.randn_like(current)
                for timestep in scheduler.timesteps:
                    k = torch.full((1,), float(timestep) * (1000 / 3), device=device)
                    output = model(torch.cat([current, candidate], dim=1), k)
                    candidate = scheduler.step(output, timestep, candidate).prev_sample
                current = candidate
                latents.append(current)
        with measured(timings, "decode_seconds", device):
            states = [tensor(initial["initial"], device)]
            for latent in latents:
                state = autoencoder.decode(latent, coords)[0]
                states.append(
                    state * tensor(stats["std"][:3], device)
                    + tensor(stats["mean"][:3], device)
                )
            raw = torch.stack(states).cpu().numpy()
            prediction = raw.copy()
            mask = np.isin(initial["node_type"], [4, 6])
            prediction[1:, mask, :2] = initial["initial"][None, mask, :2]
    prediction[0] = initial["initial"]
    raw[0] = initial["initial"]
    if diagnostics is not None and "eagle_state_uvpq" in diagnostics:
        for name in ("eagle_state_uvpq", "eagle_pre_boundary_uvpq"):
            diagnostics[name][0, :, :3] = initial["initial"]
            diagnostics[name][0, :, 3] = 0.5 * np.sum(
                initial["initial"][:, :2] ** 2, axis=-1
            )
    return prediction.astype(np.float32), np.asarray(raw, dtype=np.float32), timings
