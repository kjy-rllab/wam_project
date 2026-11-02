"""RoboCasa success rates of DreamZero servers, one server per context noise level (scripts/inference/robocasa_serve.sbatch
with SERVE_LEVELS).

Every level plays the same episodes: each client run (server_inference.run_robocasa) is one episode with the episode
index as its seed, so episode e has the same scene at every level. The levels run in parallel (a server takes one
client session at a time). Finished episodes are skipped, so running the command again resumes.

Run in the robocasa environment, with the servers tunneled to local ports (the serve job log prints the command):
    python eval_utils/robocasa_sim_eval.py --servers 0=tcp://127.0.0.1:5555 0.125=tcp://127.0.0.1:5556 \
        --output-dir ~/robocasa_workspace/server_inference/outputs/sim_eval/checkpoint-25000
"""
import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

# Tasks for the context-noise run (runs/dreamzero_robocasa_context_noise): its 8 most-sampled atomic tasks, and 8 atomic
# tasks it never sampled whose reverse it did (CloseMicrowave: OpenMicrowave, CoffeeSetupMug: CoffeeServeMug,
# PickPlaceCounterToStove: PickPlaceStoveToCounter, TurnOnStove: TurnOffStove, ...)
SEEN = ["CheesyBread", "NavigateKitchen", "CoffeeServeMug", "TurnOnToaster", "CloseStandMixerHead",
        "PickPlaceCounterToSink", "CloseCabinet", "PickPlaceMicrowaveToCounter"]
UNSEEN = ["CloseMicrowave", "CloseToasterOvenDoor", "CoffeeSetupMug", "PickPlaceCounterToStove", "CloseDrawer",
          "CloseOven", "TurnOnStove", "CloseFridgeDrawer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--servers", nargs="+", required=True, help="LEVEL=ADDRESS, e.g. 0=tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seen", nargs="*", default=SEEN, help="tasks the run trained on")
    parser.add_argument("--unseen", nargs="*", default=UNSEEN, help="tasks the run never sampled")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--max-steps", type=int, help="default: the task horizon of the RoboCasa dataset registry")
    parser.add_argument("--video", action="store_true", help="save each episode's three-camera MP4")
    parser.add_argument("--workspace", type=Path, default=Path.home() / "robocasa_workspace")
    args = parser.parse_args()

    sys.path[:0] = [str(args.workspace / "robosuite"), str(args.workspace / "robocasa")]  # as run_robocasa.make_env
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    servers = dict(server.split("=", 1) for server in args.servers)
    tasks = args.seen + args.unseen
    episode_dir = lambda level, task, e: args.output_dir / f"tcont_{level}" / task / f"seed_{e:03d}"

    def result(level, task, e):
        """The episode's summary, or None if it has not finished without error."""
        path = episode_dir(level, task, e) / "summary.json"
        episodes = json.loads(path.read_text())["episodes"] if path.exists() else []
        return episodes[0] if episodes and episodes[0]["error"] is None else None

    def run(level, address):
        for e in range(args.episodes):  # episode by episode, so partial results cover every task
            for task in tasks:
                if result(level, task, e) is not None:
                    continue
                out = episode_dir(level, task, e)
                out.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-m", "server_inference.run_robocasa", "--server", address, "--task", task,
                           "--split", args.split, "--seed", str(e), "--episodes", "1", "--replan-steps", "24",
                           "--max-steps", str(args.max_steps or get_task_horizon(task)), "--clip-actions",
                           "--output-dir", str(out)] + ["--video"] * args.video
                with open(out / "client.log", "w") as log:
                    subprocess.run(command, cwd=args.workspace, stdout=log, stderr=subprocess.STDOUT)
                episode = result(level, task, e)
                status = "error, see client.log" if episode is None else "success" if episode["success"] else "fail"
                print(f"tcont {level} {task} seed {e}: {status}", flush=True)

    threads = [threading.Thread(target=run, args=server) for server in servers.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Successes / finished episodes per task and level
    table = {level: {task: [r["success"] for e in range(args.episodes) if (r := result(level, task, e))]
                     for task in tasks} for level in servers}
    print("\n| task | " + " | ".join(f"tcont {level}" for level in servers) + " |")
    print("|---" * (len(servers) + 1) + "|")
    for task in tasks:
        name = task if task in args.seen else f"{task} (unseen)"
        print(f"| {name} | " + " | ".join(f"{sum(table[l][task])}/{len(table[l][task])}" for l in servers) + " |")
    for group, names in (("seen", args.seen), ("unseen", args.unseen)):
        rates = [sum(sum(table[l][t]) for t in names) / max(1, sum(len(table[l][t]) for t in names)) for l in servers]
        solved = [sum(any(table[l][t]) for t in names) for l in servers]
        print(f"| {group} success rate | " + " | ".join(f"{r:.0%}" for r in rates) + " |")
        print(f"| {group} tasks with a success | " + " | ".join(f"{s}/{len(names)}" for s in solved) + " |")
    (args.output_dir / "results.json").write_text(json.dumps(
        {"seen": args.seen, "unseen": args.unseen, "split": args.split, "successes": table}, indent=2))


if __name__ == "__main__":
    main()
