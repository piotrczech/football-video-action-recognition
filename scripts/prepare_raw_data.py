#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import shutil
import zipfile
import logging
import argparse
import importlib
from pathlib import Path
from dataclasses import dataclass
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
BALL_EXTRA_ROOT = RAW_ROOT / "ball-extra"
SOCCERNET_ROOT = RAW_ROOT / "soccernet"
ENV_PATH = ROOT / ".env"

ROBOFLOW_URL = "https://universe.roboflow.com/footballvisionai/playersdetection-2-vfmys/dataset/2"
SOCCERNET_TASK = "tracking-2023"
ALL_SOCCERNET_SPLITS = ("train", "test", "challenge")
COCO_KEYS = {"images", "annotations", "categories"}

logger = logging.getLogger("prepare_raw_data")


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]
    fixes: list[str]


class PrepareError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate raw datasets.")
    parser.add_argument(
        "--download-soccernet",
        action="store_true",
        help="Download SoccerNet before validation. Defaults to train only.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --download-soccernet, download and validate all SoccerNet splits: train,test,challenge.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def load_env_values() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    return {k: v for k, v in dotenv_values(ENV_PATH).items() if isinstance(v, str) and v.strip()}


def get_soccernet_password(env_values: dict[str, str]) -> str | None:
    return os.getenv("SOCCERNET_PASSWORD") or env_values.get("SOCCERNET_PASSWORD")


def is_coco(payload: object) -> bool:
    return isinstance(payload, dict) and COCO_KEYS.issubset(payload)


def has_coco_annotation(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False

    for candidate in sorted(split_dir.rglob("*.json"), key=lambda p: p.as_posix()):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if is_coco(payload):
            return True
    return False


def has_soccernet_tracking_split(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False

    sequences = [p for p in sorted(split_dir.iterdir(), key=lambda p: p.name) if p.is_dir() and p.name.startswith("SNMOT-")]
    if not sequences:
        return False

    return all((seq / "img1").is_dir() and (seq / "gt").is_dir() for seq in sequences)


def safe_extract_archive(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive, "r") as handle:
            for member in handle.infolist():
                target = (destination / member.filename).resolve()
                if target != destination and destination not in target.parents:
                    raise PrepareError(f"Unsafe zip entry in {rel(archive)}: '{member.filename}'")
            handle.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise PrepareError(f"Invalid zip archive: {rel(archive)}") from exc
    except OSError as exc:
        raise PrepareError(f"Failed to extract archive: {rel(archive)} ({exc})") from exc


def archive_marker_signature(archive: Path) -> str:
    stat = archive.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def extract_archives_once(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for archive in sorted(root.glob("*.zip"), key=lambda p: p.name):
        marker = root / f".{archive.name}.extracted"
        signature = archive_marker_signature(archive)
        if marker.exists():
            try:
                if marker.read_text(encoding="utf-8").strip() == signature:
                    continue
            except OSError:
                pass
        logger.info("Extracting archive: %s", rel(archive))
        safe_extract_archive(archive, root)
        marker.write_text(f"{signature}\n", encoding="utf-8")


def move_path(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return

    if dst.exists():
        if src.is_dir() and dst.is_dir():
            for child in sorted(src.iterdir(), key=lambda p: p.name):
                move_path(child, dst / child.name)
            src.rmdir()
            return
        raise PrepareError(f"Cannot normalize layout, target already exists: {rel(dst)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def normalize_dataset_layout(
    root: Path,
    split_aliases: dict[str, tuple[str, ...]],
    allowed_containers: tuple[str, ...] = (),
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    for container_name in allowed_containers:
        container = root / container_name
        if not container.is_dir():
            continue

        for canonical, aliases in split_aliases.items():
            for alias in aliases:
                nested = container / alias
                if nested.is_dir():
                    move_path(nested, root / canonical)
                    break

        for archive in sorted(container.glob("*.zip"), key=lambda p: p.name):
            move_path(archive, root / archive.name)

        if not any(container.iterdir()):
            container.rmdir()

    for canonical, aliases in split_aliases.items():
        direct = root / canonical
        if direct.is_dir():
            continue

        for alias in aliases:
            candidate = next(
                (
                    p
                    for p in sorted(root.rglob(alias), key=lambda x: (len(x.parts), x.as_posix()))
                    if p.is_dir() and root in p.resolve().parents and p.parent != root
                ),
                None,
            )
            if candidate is None:
                continue
            move_path(candidate, direct)
            break

    if "valid" in split_aliases:
        val_dir = root / "val"
        valid_dir = root / "valid"
        if val_dir.is_dir() and not valid_dir.exists():
            move_path(val_dir, valid_dir)


def missing_ball_extra_splits(root: Path, required_splits: tuple[str, ...]) -> list[str]:
    return [split for split in required_splits if not has_coco_annotation(root / split)]


def missing_soccernet_splits(root: Path, required_splits: tuple[str, ...]) -> list[str]:
    return [split for split in required_splits if not has_soccernet_tracking_split(root / split)]


def format_split_list(splits: tuple[str, ...] | list[str]) -> str:
    return ",".join(splits)


def soccernet_download_command(required_splits: tuple[str, ...]) -> str:
    return f"python scripts/prepare_raw_data.py --download-soccernet{' --all' if required_splits == ALL_SOCCERNET_SPLITS else ''}"


def run_soccernet_download(password: str, splits: tuple[str, ...]) -> None:
    try:
        module = importlib.import_module("SoccerNet.Downloader")
    except ModuleNotFoundError as exc:
        raise PrepareError("Missing dependency 'SoccerNet'. Install with: pip install -r requirements.txt") from exc

    SOCCERNET_ROOT.mkdir(parents=True, exist_ok=True)
    downloader = module.SoccerNetDownloader(LocalDirectory=str(SOCCERNET_ROOT))

    logger.info("Downloading SoccerNet task=%s splits=%s", SOCCERNET_TASK, list(splits))

    try:
        downloader.downloadDataTask(
            task=SOCCERNET_TASK,
            split=list(splits),
            password=password,
            source="HuggingFace",
            version=None,
        )
    except Exception as exc:
        raise PrepareError(f"SoccerNet downloader failed: {exc}") from exc


def check_ball_extra() -> CheckResult:
    split_aliases = {"train": ("train",), "valid": ("valid", "val"), "test": ("test",)}

    try:
        extract_archives_once(BALL_EXTRA_ROOT)
        normalize_dataset_layout(BALL_EXTRA_ROOT, split_aliases)
    except PrepareError as exc:
        return CheckResult(
            name="ball-extra",
            ok=False,
            details=[f"dataset layout normalization failed: {exc}"],
            fixes=[
                f"download ball-extra dataset from {ROBOFLOW_URL}",
                f"unzip data into `{rel(BALL_EXTRA_ROOT)}`",
                f"ensure structure `{rel(BALL_EXTRA_ROOT)}/{{train,valid,test}}` and COCO json per split",
                "rerun `python scripts/prepare_raw_data.py`",
            ],
        )

    missing = missing_ball_extra_splits(BALL_EXTRA_ROOT, ("train", "valid", "test"))
    if not missing:
        return CheckResult(name="ball-extra", ok=True, details=[], fixes=[])

    return CheckResult(
        name="ball-extra",
        ok=False,
        details=[
            f"missing splits: {format_split_list(missing)}",
        ],
        fixes=[
            f"download ball-extra dataset from {ROBOFLOW_URL}",
            f"unzip data into `{rel(BALL_EXTRA_ROOT)}`",
            "rerun `python scripts/prepare_raw_data.py`",
        ],
    )


def check_soccernet(download_requested: bool, required_splits: tuple[str, ...], env_values: dict[str, str]) -> CheckResult:
    split_aliases = {split: (split,) for split in ALL_SOCCERNET_SPLITS}

    if download_requested:
        password = get_soccernet_password(env_values)
        if not password:
            return CheckResult(
                name="soccernet",
                ok=False,
                details=["missing `SOCCERNET_PASSWORD` in `.env` or environment"],
                fixes=[
                    "set `SOCCERNET_PASSWORD` in `.env` (see `.env.example`)",
                    f"run `{soccernet_download_command(required_splits)}`",
                ],
            )

        try:
            run_soccernet_download(password, required_splits)
        except PrepareError as exc:
            return CheckResult(
                name="soccernet",
                ok=False,
                details=[f"soccernet download failed: {exc}"],
                fixes=[f"run `{soccernet_download_command(required_splits)}`"],
            )

    extract_error: PrepareError | None = None
    try:
        normalize_dataset_layout(SOCCERNET_ROOT, split_aliases, allowed_containers=(SOCCERNET_TASK,))
        extract_archives_once(SOCCERNET_ROOT)
        normalize_dataset_layout(SOCCERNET_ROOT, split_aliases, allowed_containers=(SOCCERNET_TASK,))
    except PrepareError as exc:
        extract_error = exc

    missing = missing_soccernet_splits(SOCCERNET_ROOT, required_splits)
    if not missing:
        return CheckResult(name="soccernet", ok=True, details=[], fixes=[])

    details = [f"missing splits: {format_split_list(missing)}"]
    if extract_error:
        details.append(f"dataset layout normalization failed: {extract_error}")

    return CheckResult(
        name="soccernet",
        ok=False,
        details=details,
        fixes=[
            f"run `{soccernet_download_command(required_splits)}`",
            "rerun `python scripts/prepare_raw_data.py`",
        ],
    )


def print_summary(results: list[CheckResult]) -> None:
    print("Verification files")
    for result in results:
        if result.ok:
            print(f"- Dataset {result.name}: OK")
            continue
        status = f"MISSING ({result.details[0]})" if result.details else "MISSING"
        print(f"- Dataset {result.name}: {status}")

    failed = [result for result in results if not result.ok]
    if not failed:
        return

    if len(failed) == 1:
        print("\nTo fix this issue:")
    else:
        print("\nTo fix these issues:")

    for result in failed:
        for detail in result.details[1:]:
            print(f"- [{result.name}] {detail}")
        for fix in result.fixes:
            print(f"- [{result.name}] {fix}")


def maybe_print_soccernet_scope_note(required_splits: tuple[str, ...]) -> None:
    if required_splits == ALL_SOCCERNET_SPLITS:
        return

    print(f"Note: validating SoccerNet split={format_split_list(required_splits)} only. Use `--all` for train,test,challenge.")


def main() -> int:
    args = parse_args()
    configure_logging()

    required_splits = ALL_SOCCERNET_SPLITS if args.all else ("train",)

    try:
        env_values = load_env_values()
    except PrepareError as exc:
        logger.error("%s", exc)
        return 2

    results = [
        check_ball_extra(),
        check_soccernet(
            download_requested=args.download_soccernet,
            required_splits=required_splits,
            env_values=env_values,
        ),
    ]

    maybe_print_soccernet_scope_note(required_splits)
    print_summary(results)
    return 0 if all(result.ok for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
