import argparse
import os
import time

from dataflow.status import collect, render, trackio_metrics


def status(args: argparse.Namespace) -> None:
    tracker = None
    if args.trackio:
        import trackio

        name = args.run_name or args.logging_dir.rstrip("/").split("/")[-2]
        trackio.init(project=args.project, name=name, space_id=args.trackio, resume="allow")
        tracker = trackio
    alerted: set[tuple[int, str]] = set()
    while True:
        current = collect(args.logging_dir, stale_after=args.stale_after)
        if args.watch:
            print("\033[2J\033[H", end="")
        print(render(current), flush=True)
        if tracker:
            tracker.log(trackio_metrics(current))
            for task in current.tasks:
                if task.status in ("failed", "stale") and (task.rank, task.status) not in alerted:
                    alerted.add((task.rank, task.status))
                    message = current.errors.get(task.rank, "no progress update for a while")
                    tracker.alert(f"task {task.rank} {task.status}", text=message, level=tracker.AlertLevel.ERROR)
        finished = all(task.status in ("done", "failed") for task in current.tasks)
        if not args.watch or finished:
            break
        time.sleep(args.watch)
    if tracker:
        tracker.finish()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    parser = argparse.ArgumentParser(prog="dataflow")
    commands = parser.add_subparsers(required=True)
    status_parser = commands.add_parser("status", help="progress, funnel and cost of a run")
    status_parser.add_argument("logging_dir", help="the run's logging folder, local or hf://buckets/...")
    status_parser.add_argument("--watch", type=float, default=0, help="refresh every N seconds until the run ends")
    status_parser.add_argument("--stale-after", type=float, default=120, help="seconds without progress = stale")
    status_parser.add_argument("--trackio", metavar="SPACE_ID", help="also log to a Trackio dashboard on this Space")
    status_parser.add_argument("--project", default="dataflow", help="Trackio project")
    status_parser.add_argument("--run-name", help="Trackio run name (default: the run folder name)")
    status_parser.set_defaults(handler=status)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
