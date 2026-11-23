"""
Train an RL policy with GreenLight-Gym2's own ExperimentManager, in its own process.

The Simulation Lab starts this script as a separate local process so training
never blocks web requests. Everything that decides how training works comes
from the Gym2 repository: the environment config (gl_gym/configs/envs), the
agent hyperparameters (configs/agents/<algorithm>.yml), the vectorised,
normalised environments and the evaluation callbacks (RL/utils.py). The only
thing this wrapper changes is how long it trains (`total_timesteps`) and how
many environments run side by side (`n_envs`), so a short trial run is possible
on a laptop. W&B runs offline (WANDB_MODE=offline is set by the caller).

Usage (cwd must be the Gym2 repository root, as Gym2's scripts expect):
    python train_runner.py --algorithm ppo --total_timesteps 50000 --n_envs 2 \
        --seed 666 --project SplattoLab --group lab --status_file /path/status.json
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback

# Gym2's SubprocVecEnv workers unpickle `gl_gym.components.weather` first, which
# imports `gl_gym.environments`, which imports `weather` again: a circular import
# that fails under macOS's forkserver/spawn start methods (Linux forks, and
# never hits it). Loading `gl_gym.environments` in the forkserver before any
# worker starts resolves the cycle without changing Gym2's code.
multiprocessing.set_forkserver_preload(["gl_gym.environments"])


def write_status(path, **fields):
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(fields, updated_at=time.time())
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "sac", "recurrentppo"])
    parser.add_argument("--total_timesteps", type=int, required=True)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--n_evals", type=int, default=5)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--project", default="SplattoLab")
    parser.add_argument("--group", default="lab")
    parser.add_argument("--status_file", required=True)
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    write_status(args.status_file, state="starting", pid=os.getpid())
    try:
        from RL.experiment_manager import ExperimentManager
        from RL.utils import load_env_params, load_model_hyperparams

        env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
        hyperparameters = load_model_hyperparams(args.algorithm, "GreenLightEnv")
        configured = {"total_timesteps": hyperparameters["total_timesteps"], "n_envs": hyperparameters["n_envs"]}
        hyperparameters["total_timesteps"] = int(args.total_timesteps)
        if args.n_envs:
            hyperparameters["n_envs"] = int(args.n_envs)

        manager = ExperimentManager(
            env_id="GreenLightEnv",
            project=args.project,
            env_kwargs=env_kwargs,
            hyperparameters=hyperparameters,
            group=args.group,
            n_eval_episodes=1,
            n_evals=args.n_evals,
            algorithm=args.algorithm,
            env_seed=args.seed,
            model_seed=args.seed,
            stochastic=False,
            save_model=True,
            save_env=True,
            hp_tuning=False,
            device="cpu",
        )
        run_name = manager.run.name
        write_status(
            args.status_file,
            state="training",
            run_name=run_name,
            configured_defaults=configured,
            model_dir=f"train_data/{args.project}/{args.algorithm}/deterministic/models/{run_name}",
            env_dir=f"train_data/{args.project}/{args.algorithm}/deterministic/envs/{run_name}",
        )
        manager.run_experiment()
        write_status(args.status_file, state="finished")
    except BaseException as exc:  # noqa: BLE001 - report every failure, including KeyboardInterrupt
        write_status(args.status_file, state="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
        raise


if __name__ == "__main__":
    main()
