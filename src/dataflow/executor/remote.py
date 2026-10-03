import shlex

from huggingface_hub import HfApi, get_token

from dataflow.executor.jobs import UV_IMAGE

SOURCE = "https://github.com/kuluruvineeth/dataflow/archive/{commit}.tar.gz"


def source_command(module: str, arguments: list[str], commit: str) -> list[str]:
    """Download this repository at `commit` (a full sha), unpack it and run `python -m module` from the source tree,
    so that pipelines it starts on Jobs can build their wheel from the same code."""
    unpack = (
        "import io, tarfile, urllib.request; "
        f"tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('{SOURCE.format(commit=commit)}').read()))"
        ".extractall('/src')"
    )
    run = f"uv run python -m {module} {' '.join(shlex.quote(argument) for argument in arguments)}"
    return ["sh", "-c", f'python -c "{unpack}" && cd /src/dataflow-{commit} && {run}']


def launch_driver(
    module: str, arguments: list[str], commit: str, name: str, timeout: str = "7d", flavor: str = "cpu-basic"
):
    """Run a long pipeline's launcher as a Job itself, so it keeps going when the laptop that started it doesn't."""
    return HfApi().run_job(
        image=UV_IMAGE, command=source_command(module, arguments, commit), env={"PYTHONUNBUFFERED": "1"},
        secrets={"HF_TOKEN": get_token()}, flavor=flavor, timeout=timeout, name=name,
    )  # fmt: skip


def schedule_daily(module: str, arguments: list[str], commit: str, name: str, flavor: str = "cpu-basic"):
    """A Job every day at 03:00 UTC for up to 23 hours, never two at once: for work that resumes where it stopped and
    takes longer than one Job may run."""
    return HfApi().create_scheduled_job(
        image=UV_IMAGE, command=source_command(module, arguments, commit), schedule="0 3 * * *", concurrency=False,
        timeout="23h", flavor=flavor, env={"PYTHONUNBUFFERED": "1"}, secrets={"HF_TOKEN": get_token()}, name=name,
    )  # fmt: skip
