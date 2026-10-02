import argparse
import os
import time
from itertools import islice

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


def inspect(args: argparse.Namespace) -> None:
    from dataflow.browse import describe, read_documents, sample, select

    documents = select(read_documents(args.path), reason=args.reason, domain=args.domain, grep=args.grep)
    if args.sample:
        chosen = sample(documents, args.sample, seed=args.seed)
    else:
        chosen = list(islice(documents, args.limit))
    for document in chosen:
        print(describe(document, width=args.width), end="\n\n")
    print(f"{len(chosen)} documents shown")


def report(args: argparse.Namespace) -> None:
    from dataflow.report import build_report, render_report, save_report

    rows = build_report(args.logging_dir, kept=args.kept, top=args.top)
    print(render_report(rows))
    if args.save:
        print(f"\nsaved {save_report(rows, args.logging_dir)} in {args.logging_dir}")


def publish(args: argparse.Namespace) -> None:
    import tempfile

    from dataflow.publish import build_preview, upload

    out_dir = args.out or tempfile.mkdtemp(prefix="dataflow-publish-")
    title = args.title or (args.repo or "dataflow run").split("/")[-1]
    counts = build_preview(args.kept, args.removed, out_dir, title=title)
    print("configs:", ", ".join(f"{name} {count:,}" for name, count in counts.items()))
    if args.repo:
        print(upload(out_dir, args.repo, private=not args.public))


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
    inspect_parser = commands.add_parser("inspect", help="browse documents in a kept or removed folder")
    inspect_parser.add_argument("path", help="folder of .jsonl(.gz) or .parquet files, local or hf://")
    inspect_parser.add_argument("--reason", help="only documents removed for this filter_reason")
    inspect_parser.add_argument("--domain", help="only documents whose URL host ends with this")
    inspect_parser.add_argument("--grep", help="only documents whose text matches this regex")
    inspect_parser.add_argument("--sample", type=int, help="uniform random sample of N matching documents")
    inspect_parser.add_argument("--seed", type=int, default=0)
    inspect_parser.add_argument("--limit", type=int, default=10, help="without --sample, the first N matches")
    inspect_parser.add_argument("--width", type=int, default=300, help="characters of text to show")
    inspect_parser.set_defaults(handler=inspect)
    report_parser = commands.add_parser("report", help="funnel, languages and domains of a run")
    report_parser.add_argument("logging_dir")
    report_parser.add_argument("--kept", help="output folder of kept documents, for language and domain counts")
    report_parser.add_argument("--top", type=int, default=20)
    report_parser.add_argument("--save", action="store_true", help="write report.parquet into the logging folder")
    report_parser.set_defaults(handler=report)
    publish_parser = commands.add_parser("publish", help="kept and removed documents as a browsable dataset repo")
    publish_parser.add_argument("--kept", required=True)
    publish_parser.add_argument("--removed", help="folder with one subfolder of removed documents per step")
    publish_parser.add_argument("--repo", help="dataset repo id to upload to (omit to only build locally)")
    publish_parser.add_argument("--public", action="store_true")
    publish_parser.add_argument("--title")
    publish_parser.add_argument("--out", help="local folder to build into (default: a temporary folder)")
    publish_parser.set_defaults(handler=publish)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
