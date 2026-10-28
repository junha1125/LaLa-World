"""Fit per-env linear proprio propagation proprio_{t+5} ~= [proprio_t, action_chunk, 1] @ W.

Same (t, t+5) windows as the JSONL builder; raw units. Writes proprio_dyn.json
next to this file with W, per-dim R^2, residual std.
"""

import json
import os
import pickle

import numpy as np

RAW = os.path.join(os.environ.get("LALA_DATA_ROOT", "data"), "sim_raw")
WINDOW = 5
MAX_ROWS = 200_000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proprio_dyn.json")


def windows(actions, proprio, n_frames):
    for t0 in range(0, n_frames - WINDOW, WINDOW):
        if t0 + WINDOW >= len(proprio):  # wall: 50 states vs 51 frames
            break
        yield proprio[t0], actions[t0 : t0 + WINDOW].sum(axis=0), proprio[t0 + WINDOW]


def iter_wall():
    import torch

    root = f"{RAW}/wall_single"
    st = torch.load(f"{root}/states.pth", map_location="cpu").double().numpy()
    ac = torch.load(f"{root}/actions.pth", map_location="cpu").double().numpy()
    for i in range(len(st)):
        yield ac[i], st[i], ac.shape[1] + 1


def iter_point_maze():
    import torch

    root = f"{RAW}/point_maze"
    st = torch.load(f"{root}/states.pth", map_location="cpu").double().numpy()
    ac = torch.load(f"{root}/actions.pth", map_location="cpu").double().numpy()
    sl = torch.load(f"{root}/seq_lengths.pth", map_location="cpu").numpy()
    for i in range(len(sl)):
        yield ac[i, : sl[i]], st[i, : sl[i]], int(sl[i])


def iter_pusht():
    import torch

    root = f"{RAW}/pusht_noise/train"
    st = torch.load(f"{root}/states.pth", map_location="cpu").double().numpy()
    ac = torch.load(f"{root}/rel_actions.pth", map_location="cpu").double().numpy()
    vel = torch.load(f"{root}/velocities.pth", map_location="cpu").double().numpy()
    with open(f"{root}/seq_lengths.pkl", "rb") as f:
        sl = pickle.load(f)
    for i in range(len(sl)):
        T = int(sl[i])
        yield ac[i, :T], np.concatenate([st[i, :T, :2], vel[i, :T]], axis=1), T


def iter_metaworld():
    import pyarrow.parquet as pq

    mw = f"{RAW}/metaworld/data"
    for fname in sorted(f for f in os.listdir(mw) if f.endswith(".parquet")):
        pf = pq.ParquetFile(os.path.join(mw, fname))
        for rg in range(pf.num_row_groups):
            tb = pf.read_row_group(rg, columns=["states", "actions"])
            for i in range(tb.num_rows):
                states = np.asarray(tb["states"][i].as_py(), dtype=np.float64)[:-1]
                actions = np.asarray(tb["actions"][i].as_py(), dtype=np.float64)
                yield actions, states[:, :4], len(actions)


ENVS = {"wall": iter_wall, "point_maze": iter_point_maze, "pusht": iter_pusht, "metaworld": iter_metaworld}


def fit(env_name):
    P0, A, P1 = [], [], []
    for actions, proprio, n_frames in ENVS[env_name]():
        for p0, a, p1 in windows(actions, proprio, n_frames):
            P0.append(p0)
            A.append(a)
            P1.append(p1)
        if len(P0) >= MAX_ROWS:
            break
    P0, A, P1 = np.stack(P0), np.stack(A), np.stack(P1)
    X = np.concatenate([P0, A, np.ones((len(P0), 1))], axis=1)
    W, *_ = np.linalg.lstsq(X, P1, rcond=None)
    pred = X @ W
    ss_res = ((P1 - pred) ** 2).sum(axis=0)
    ss_tot = ((P1 - P1.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    resid_std = (P1 - pred).std(axis=0)
    print(f"{env_name}: n={len(P0)}  R2={np.round(r2, 4).tolist()}  resid_std={np.round(resid_std, 4).tolist()}")
    return {"W": W.tolist(), "r2": r2.tolist(), "resid_std": resid_std.tolist(), "n": int(len(P0))}


def main():
    out = {name: fit(name) for name in ENVS}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
