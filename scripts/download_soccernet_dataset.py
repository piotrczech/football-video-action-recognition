#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path
import random
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("data/raw/soccernet")
DEFAULT_EXTRACT_ROOT = Path("data/raw/soccernet/soccernet_tracking_2023_coco")
ENV_PATH = ROOT / ".env"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("download_soccernet_dataset")


def build_downloader(local_directory: str):
    try:
        module = importlib.import_module("SoccerNet.Downloader")
        utils_module = importlib.import_module("SoccerNet.utils")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'SoccerNet'. Install requirements first: pip install -r requirements.txt"
        ) from exc

    return module.SoccerNetDownloader(LocalDirectory=local_directory), utils_module.getListGames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SoccerNet data using SoccerNet library.")
    parser.add_argument("--mode", choices=["task", "games"], default="task", help="Download mode")
    parser.add_argument("--task", default="tracking-2023", help="SoccerNet task name, e.g. tracking-2023")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Task splits to download")
    parser.add_argument(
        "--task-version",
        default=None,
        help="Optional version passed to downloadDataTask, e.g. 224p",
    )
    parser.add_argument(
        "--games-task",
        default="spotting",
        help="Task used by getListGames in games mode, e.g. spotting",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["1_224p.mkv", "2_224p.mkv", "Labels-v2.json"],
        help="Files downloaded per game in games mode",
    )
    parser.add_argument("--max-games", type=int, default=0, help="Limit number of games per split in games mode")
    parser.add_argument("--randomized", action="store_true", help="Randomize game order in games mode")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Base local output directory")
    parser.add_argument(
        "--extract-root",
        default=str(DEFAULT_EXTRACT_ROOT),
        help="Where extracted zip contents should be placed",
    )
    parser.add_argument("--source", default="HuggingFace", help="Download source passed to SoccerNetDownloader")
    parser.add_argument("--password", default=None, help="SoccerNet task password if required")
    parser.add_argument("--no-extract", action="store_true", help="Skip zip extraction step")
    return parser.parse_args()


def resolve_path(path_arg: str) -> Path:
    candidate = Path(path_arg)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip().strip("\"'")
    return parsed


def resolve_password(cli_password: str | None, dotenv_values: dict[str, str]) -> tuple[str | None, str]:
    if cli_password:
        return cli_password, "cli"

    for key in ("SOCCERNET_PASSWORD", "SOCCERNET_DATA_PASSWORD"):
        value = dotenv_values.get(key)
        if value and "CHANGEME" not in value and "YOUR_" not in value:
            return value, f".env:{key}"

    for key in ("SOCCERNET_PASSWORD", "SOCCERNET_DATA_PASSWORD"):
        value = os.getenv(key)
        if value and "CHANGEME" not in value and "YOUR_" not in value:
            return value, f"env:{key}"
    return None, "default"


def find_zip_archives(task_root: Path) -> list[Path]:
    if not task_root.exists():
        return []
    return sorted([p for p in task_root.rglob("*.zip") if p.is_file()])


def split_from_name(archive_name: str) -> str:
    lowered = archive_name.lower()
    if "train" in lowered:
        return "train"
    if "valid" in lowered or "val" in lowered:
        return "valid"
    if "test" in lowered:
        return "test"
    if "challenge" in lowered:
        return "challenge"
    return "other"


def extract_archives(archives: list[Path], extract_root: Path) -> list[Path]:
    extracted_to: list[Path] = []
    extract_root.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        destination = extract_root / split_from_name(archive.stem)
        destination.mkdir(parents=True, exist_ok=True)
        logger.info("extracting %s -> %s", archive, destination)
        with zipfile.ZipFile(archive, "r") as zip_handle:
            zip_handle.extractall(destination)
        extracted_to.append(destination)

    return extracted_to


def select_games(get_list_games, split: str, games_task: str, max_games: int, randomized: bool) -> list[str]:
    game_list = list(get_list_games(split=split, task=games_task))
    if randomized:
        random.shuffle(game_list)
    if max_games > 0:
        return game_list[:max_games]
    return game_list


def warn_on_action_spotting_inputs(mode: str, games_task: str, files: list[str]) -> None:
    if mode != "games":
        return

    normalized = {f.lower() for f in files}
    has_action_labels = "labels-v2.json" in normalized
    if has_action_labels:
        logger.warning(
            "Labels-v2.json are action spotting events, not player/ball detection annotations."
        )
    if games_task == "spotting" and has_action_labels:
        logger.warning(
            "games_task=spotting targets action spotting data. For player/ball use mode=task with task=tracking-2023."
        )


def main() -> int:
    args = parse_args()

    output_root = resolve_path(args.output_root)
    extract_root = resolve_path(args.extract_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dotenv_values = load_dotenv(ENV_PATH)
    password, password_source = resolve_password(args.password, dotenv_values)
    effective_password = password or "SoccerNet"
    logger.info("password source=%s", password_source)
    warn_on_action_spotting_inputs(args.mode, args.games_task, args.files)

    downloader, get_list_games = build_downloader(str(output_root))
    downloader.password = effective_password

    if args.mode == "task":
        if args.max_games > 0:
            logger.info("--max-games is ignored in task mode")

        logger.info("downloading task=%s splits=%s to %s", args.task, args.splits, output_root)
        downloader.downloadDataTask(
            task=args.task,
            split=args.splits,
            password=effective_password,
            source=args.source,
            version=args.task_version,
        )

        task_root = output_root / args.task
        archives = find_zip_archives(task_root)
        logger.info("download completed, found %s zip archives in %s", len(archives), task_root)

        if args.no_extract:
            return 0

        if not archives:
            logger.warning("no zip archives found for extraction")
            return 0

        extracted_to = extract_archives(archives, extract_root)
        logger.info("extraction completed into %s", extract_root)
        logger.info("touched split folders: %s", ", ".join(sorted({str(p) for p in extracted_to})))
        return 0

    logger.info(
        "downloading selected games (games_task=%s, splits=%s, max_games=%s)",
        args.games_task,
        args.splits,
        args.max_games,
    )
    for split in args.splits:
        games = select_games(get_list_games, split, args.games_task, args.max_games, args.randomized)
        logger.info("split=%s selected_games=%s", split, len(games))
        for game in games:
            downloader.downloadGame(game=game, files=args.files, spl=split, verbose=True)

    logger.info("games mode download completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
