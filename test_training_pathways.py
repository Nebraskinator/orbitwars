#!/usr/bin/env python3
"""
Smoke test for the two Orbit Wars training pathways.

Runs a SHORT, REAL Ray cluster (CPU-only, tiny model, tiny batches) and asserts
that each learner pathway actually performs at least one optimizer update:

  * imitation : an OfflineRolloutWorker streams the sample JSON replays in
                ``./data`` into a LearnerActor running in ``imitation`` mode.
  * ppo       : a RolloutWorker self-plays a couple of tiny matches against the
                live InferenceActor and feeds episodes into a LearnerActor
                running in ``ppo`` mode.

A pathway passes when ``LearnerActor.get_stats()["update"]`` reaches >= 1 before
the timeout. This exercises the genuine distributed wiring (Ray actors, weight
store, GAE, ppo_update) end-to-end -- it is a functional check, not a training
run, so the model and hyperparameters are deliberately minimal for speed.

Usage:
    python test_training_pathways.py             # both pathways
    python test_training_pathways.py imitation   # one pathway
    python test_training_pathways.py ppo
    python test_training_pathways.py --timeout 240

Requires: torch, ray, numpy (see requirements.txt).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATA_DIR = os.path.join(HERE, "data")


def _tiny_cfg(mode: str, ckpt_dir: str):
    """A minimal CPU RunConfig that still builds the real OrbitTransformer.

    The architecture is shrunk (d_model=32, 1 layer) purely so a single update
    runs in seconds on CPU; the code paths are identical to a full run.
    """
    from config import RunConfig

    base = RunConfig.default()

    model = dataclasses.replace(
        base.model,
        d_model=32,
        pair_dim=32,
        pair_hidden_dim=32,
        n_layers=1,
        n_heads=2,
        grad_checkpoint=False,
        use_ar_head=False,
    )
    env = dataclasses.replace(base.env, max_steps=40, n_agents=(2,))
    rollout = dataclasses.replace(
        base.rollout,
        target_concurrent_battles=2,
        rooms_per_pair=2,
        learn_min_episodes=1,
        learn_max_episodes=2,
        learn_max_pending_batches=2,
    )
    infer = dataclasses.replace(base.infer, device="cpu")
    learner = dataclasses.replace(
        base.learner,
        device="cpu",
        mode=mode,
        steps_per_update=64,     # tiny: one sample replay / a couple matches suffices
        minibatch_size=16,
        grad_accum_steps=1,
        update_epochs=1,
        save_every_updates=10_000,   # don't checkpoint during the smoke test
        resume=False,
        ckpt_dir=ckpt_dir,
    )
    return dataclasses.replace(
        base, model=model, env=env, rollout=rollout, infer=infer, learner=learner
    )


def _wait_for_update(learner, timeout_s: float, label: str) -> bool:
    """Poll the learner until it reports update >= 1, or time out."""
    import ray

    deadline = time.time() + timeout_s
    last_steps = -1
    while time.time() < deadline:
        try:
            stats = ray.get(learner.get_stats.remote(), timeout=10.0)
        except Exception as e:
            print(f"  [{label}] get_stats failed (retrying): {e}")
            time.sleep(2.0)
            continue
        upd = int(stats.get("update", 0))
        steps = int(stats.get("total_steps", 0))
        if steps != last_steps:
            print(f"  [{label}] buffered_steps={steps} updates={upd}")
            last_steps = steps
        if upd >= 1:
            print(f"  [{label}] PASS -- learner completed {upd} update(s).")
            return True
        time.sleep(2.0)
    print(f"  [{label}] FAIL -- no learner update within {timeout_s:.0f}s.")
    return False


def run_imitation(timeout_s: float) -> bool:
    import ray
    from inference import WeightStore
    from learner import LearnerActor
    from offline_worker import OfflineRolloutWorker

    print("[imitation] launching learner + offline replay worker ...")
    ckpt_dir = tempfile.mkdtemp(prefix="ow_il_ckpt_")
    cfg = _tiny_cfg("imitation", ckpt_dir)
    cfg_ref = ray.put(cfg)

    ws = WeightStore.remote()
    learner = ray.remote(num_gpus=0)(LearnerActor).remote(cfg_ref, ws)
    worker = (
        ray.remote(num_cpus=1, max_concurrency=2)(OfflineRolloutWorker)
        .remote(
            cfg=cfg_ref,
            inference_actor=None,
            learner_actor=learner,
            data_dir=DATA_DIR,
            worker_id=0,
            num_workers=1,
            refresh_sec=2.0,
            # Empty allowlist -> train on every player in the replays, so the
            # smoke test does not depend on which curated names are present.
            curated_players_path="",
        )
    )

    worker.run.remote()  # fire-and-forget; the worker loops over replay epochs
    try:
        return _wait_for_update(learner, timeout_s, "imitation")
    finally:
        for h in (worker, learner, ws):
            try:
                ray.kill(h)
            except Exception:
                pass


def run_ppo(timeout_s: float) -> bool:
    import ray
    from inference import WeightStore, InferenceActor
    from learner import LearnerActor
    from worker import RolloutWorker

    print("[ppo] launching learner + inference + self-play rollout worker ...")
    ckpt_dir = tempfile.mkdtemp(prefix="ow_ppo_ckpt_")
    cfg = _tiny_cfg("ppo", ckpt_dir)
    cfg_ref = ray.put(cfg)

    ws = WeightStore.remote()
    learner = ray.remote(num_gpus=0)(LearnerActor).remote(cfg_ref, ws)
    infer = ray.remote(num_gpus=0)(InferenceActor).remote(cfg_ref, ws)
    worker = (
        ray.remote(num_cpus=1, max_concurrency=2)(RolloutWorker)
        .remote(
            cfg=cfg_ref,
            inference_actor=infer,
            learner_actor=learner,
            pairs=2,
            worker_id=0,
            seed_base=12345,
            search_actor=None,
        )
    )

    worker.run.remote()  # fire-and-forget; the worker self-plays forever
    try:
        return _wait_for_update(learner, timeout_s, "ppo")
    finally:
        for h in (worker, infer, learner, ws):
            try:
                ray.kill(h)
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "pathway",
        nargs="?",
        default="both",
        choices=["both", "imitation", "ppo"],
        help="Which training pathway to exercise (default: both).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-pathway timeout in seconds (default: 300).",
    )
    args = ap.parse_args()

    import torch
    import ray

    torch.set_num_threads(2)
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_level="warning",
    )

    results = {}
    try:
        if args.pathway in ("both", "imitation"):
            results["imitation"] = run_imitation(args.timeout)
        if args.pathway in ("both", "ppo"):
            results["ppo"] = run_ppo(args.timeout)
    finally:
        ray.shutdown()

    print("\n=== RESULTS ===")
    for name, ok in results.items():
        print(f"  {name:10s}: {'PASS' if ok else 'FAIL'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
