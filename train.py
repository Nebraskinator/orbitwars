"""
Cluster Driver for Distributed Orbit Wars Reinforcement Learning.

Initializes the Ray cluster, spawns fractional-GPU actors for inference and
training, and distributes self-play matches across N rollout-worker
actors (each owning M Kaggle ``orbit_wars`` envs).

When the configuration mode is set to imitation learning, it automatically
bypasses live self-play allocation and spawns a single offline dataloader
worker to stream JSON replays directly into the learner.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Final

import ray
from config import RunConfig
from inference import InferenceActor, WeightStore
from learner import LearnerActor

# Import both worker types
from worker import RolloutWorker
from offline_worker import OfflineRolloutWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Number of live rollout-worker actors to spawn during standard RL.
N_ROLLOUT_ACTORS: Final[int] = 12
N_OFFLINE_WORKERS: Final[int] = 10
OFFLINE_REFRESH_SEC: Final[float] = 60.0

# Base seed for live PPO env maps. Individual worker/match/episode seeds are
# derived from this. Change this to intentionally generate a different run.
LIVE_ENV_SEED_BASE: Final[int] = 12345

async def main():
    """
    Entry point for the training cluster.
    """
    try:
        ray.init(ignore_reinit_error=True)
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return

    # 1. Global Configuration (Single Source of Truth).
    cfg = RunConfig.default()

    # Determine execution mode
    run_mode = str(getattr(cfg.learner, "mode", "ppo"))
    is_offline = run_mode.startswith("imitation")

    matches_per_actor = int(cfg.rollout.rooms_per_pair)
    target_concurrent = int(cfg.rollout.target_concurrent_battles)

    if is_offline:
        logger.info(f"Initializing run in OFFLINE IMITATION mode. Routing data from JSON replays.")
    else:
        total_matches = target_concurrent
        base = total_matches // N_ROLLOUT_ACTORS
        remainder = total_matches % N_ROLLOUT_ACTORS
        matches_per_worker = [
            base + (1 if i < remainder else 0) for i in range(N_ROLLOUT_ACTORS)
        ]
        logger.info(
            f"Initializing run in LIVE RL mode: target_matches={target_concurrent}, "
            f"total_matches={total_matches}, rollout_actors={N_ROLLOUT_ACTORS}"
        )

    # 2. Shared Infrastructure.
    cfg_ref = ray.put(cfg)
    weight_store = WeightStore.remote()

    InferRemote = ray.remote(num_gpus=0.35)(InferenceActor)
    LearnerRemote = ray.remote(num_gpus=0.65)(LearnerActor)

    infer = InferRemote.remote(cfg_ref, weight_store)
    learner = LearnerRemote.remote(cfg_ref, weight_store)

    # Dedicated GPU SearchActor for SBR search (only when enabled). Owns its own
    # synced model replica and the whole decode -> sim -> assemble -> value
    # forward model, replacing the workers' CPU search path.
    search = None
    if bool(getattr(cfg.search, "enabled", False)):
        from search_actor import SearchActor
        search_gpus = float(getattr(cfg.search, "actor_num_gpus", 0.0))
        SearchRemote = ray.remote(num_gpus=search_gpus)(SearchActor)
        search = SearchRemote.remote(cfg_ref, weight_store)

    # 3. Spawn Workers based on execution mode.
    workers = []

    if is_offline:
        offline_workers = int(getattr(cfg.rollout, "offline_workers", N_OFFLINE_WORKERS))
        offline_refresh_sec = float(getattr(cfg.rollout, "offline_refresh_sec", OFFLINE_REFRESH_SEC))

        logger.info(
            f"Spawning {offline_workers} offline dataloader worker(s). "
            f"Replay directory will be rescanned every {offline_refresh_sec:.1f}s."
        )

        WorkerRemote = ray.remote(num_cpus=1)(OfflineRolloutWorker).options(
            max_restarts=-1,
            max_task_retries=0,
            max_concurrency=2,  # needed so heartbeat can run while run() is looping
        )

        for worker_id in range(offline_workers):
            worker_actor = WorkerRemote.remote(
                cfg=cfg_ref,
                inference_actor=infer,
                learner_actor=learner,
                data_dir="./data",
                worker_id=worker_id,
                num_workers=offline_workers,
                refresh_sec=offline_refresh_sec,
            )
            workers.append(worker_actor)
    else:
        # Spawn N live Rollout actors
        WorkerRemote = ray.remote(num_cpus=1)(RolloutWorker).options(
            max_restarts=-1, max_task_retries=0, max_concurrency=2
        )
        for i in range(N_ROLLOUT_ACTORS):
            count = matches_per_worker[i]
            if count <= 0:
                continue
            worker_actor = WorkerRemote.remote(
                cfg=cfg_ref,
                inference_actor=infer,
                learner_actor=learner,
                pairs=count,
                worker_id=i,
                seed_base=LIVE_ENV_SEED_BASE,
                search_actor=search,
            )
            workers.append(worker_actor)

    logger.info(f"Deployment complete. Spawned {len(workers)} worker(s).")

    # 4. Staggered startup.
    logger.info("Staggering actor initialization...")
    await asyncio.sleep(2)

    run_refs = []
    for w in workers:
        run_refs.append(w.run.remote())
        time.sleep(0.5)

    # 5. Monitoring loop.
    logger.info("Training started. Entering telemetry loop.")
    while True:
        try:
            finished, _ = ray.wait(run_refs, num_returns=len(run_refs), timeout=0)
            if finished:
                for ref in finished:
                    try:
                        ray.get(ref)
                    except Exception as e:
                        logger.critical(f"FATAL: A Worker crashed in run(): {e}")
                        import traceback
                        traceback.print_exception(type(e), e, e.__traceback__)
                        time.sleep(5)
                logger.critical("Shutting down cluster due to worker crash.")
                ray.shutdown()
                sys.exit(1)

            istats_ref = infer.get_stats.remote()
            lstats_ref = learner.get_stats.remote()
            wstats_refs = [w.heartbeat.remote() for w in workers]

            results = await asyncio.to_thread(
                ray.get, [istats_ref, lstats_ref] + wstats_refs
            )

            istats, lstats = results[0], results[1]
            wstats_list = results[2:]

            total_matches_active = sum(w.get("active_matches", 0) for w in wstats_list)
            mem_matches_active = sum(w.get("total_steps_in_memory", 0) for w in wstats_list)

            stats_msg = (
                f"[Telemetry] MATCHES: {total_matches_active}, MEM: {mem_matches_active} | "
                f"INFER: {istats} | TRAIN: {lstats}"
            )
            logger.info(stats_msg)

        except Exception as e:
            logger.warning(f"Telemetry loop error: {e}")

        await asyncio.sleep(5.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Terminating cluster...")
        ray.shutdown()
