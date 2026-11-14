"""Platform paths and passive game launch, separate from route execution."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
MAC_BRIDGE_FILES = ("CelesteInstrumentation.dll", "0Harmony.dll",
                    "System.ValueTuple.dll", "Newtonsoft.Json.dll")


def default_game_dir():
    if sys.platform == "darwin":
        return WORKSPACE / "GAME/Celeste-Mac/Celeste.app/Contents/Resources"
    return WORKSPACE / "GAME/Celeste-1"


def normalize_game_dir(value):
    """Accept a managed resources directory or its enclosing macOS app."""
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() == ".app":
        path = path / "Contents/Resources"
    elif path.name == "Contents" and (path / "Resources").is_dir():
        path = path / "Resources"
    return path.resolve()


def require_isolated_mac_game(game_dir):
    path = normalize_game_dir(game_dir)
    if sys.platform == "darwin" and not path.is_relative_to((WORKSPACE / "GAME").resolve()):
        raise ValueError("Mac bridge installation and launch require a project-local GAME copy; the Steam original is never modified.")
    return path


def game_launch_command(game_dir):
    path = normalize_game_dir(game_dir)
    if sys.platform == "darwin":
        if path.name != "Resources" or path.parent.name != "Contents":
            raise ValueError("Select Celeste.app or its Contents/Resources directory on macOS")
        executable = path.parent / "MacOS/Celeste"
    else:
        executable = path / "Celeste.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return [str(executable)]


def game_is_running(game_dir):
    """Inspect the native executable, including games launched by another helper."""
    if sys.platform != "darwin":
        return False
    executable = game_launch_command(game_dir)[0]
    try:
        processes = subprocess.run(["ps", "-axww", "-o", "comm="], capture_output=True,
                                   text=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        raise OSError("Could not inspect running games before launch") from error
    return any(line.strip() == executable for line in processes.splitlines())


def require_game_stopped(game_dir):
    if game_is_running(game_dir):
        raise ValueError("The selected Celeste copy is already running; connect with --no-launch instead of starting another copy")


def backup_mac_saves(source=None, destination=None):
    """Snapshot existing saves once per content version; never alter originals."""
    source = Path(source) if source is not None else Path.home() / "Library/Application Support/Celeste"
    destination = Path(destination) if destination is not None else WORKSPACE / "artifacts/mac-save-backups"
    files = sorted(path for name in ("Saves", "Backups")
                   for path in (source / name).rglob("*") if path.is_file())
    if not files:
        return None
    entries = []
    for path in files:
        entries.append({"path": path.relative_to(source).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    snapshot = destination / digest[:20]
    manifest = snapshot / "snapshot.json"
    if manifest.is_file():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded.get("digest") == digest and all(
            (snapshot / entry["path"]).is_file() and
            hashlib.sha256((snapshot / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
            for entry in entries
        ):
            return snapshot
        raise ValueError(f"Existing save snapshot failed verification: {snapshot}")
    snapshot.mkdir(parents=True, exist_ok=True)
    for path, entry in zip(files, entries):
        target = snapshot / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise OSError("Save files changed during backup; close other Celeste instances and retry")
    manifest.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                   "source": str(source), "digest": digest, "files": entries},
                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


def launch_game_process(game_dir, *, fna3d_driver=None, sdl_videodriver=None):
    path = require_isolated_mac_game(game_dir)
    command = game_launch_command(path)
    require_game_stopped(path)
    environment = os.environ.copy()
    if fna3d_driver:
        environment["FNA3D_FORCE_DRIVER"] = str(fna3d_driver)
    if sdl_videodriver:
        environment["SDL_VIDEODRIVER"] = str(sdl_videodriver)
    if sys.platform == "darwin":
        backup_mac_saves()
        # Prevent Steam from restarting the unmodified store installation.
        (path / "steam_appid.txt").write_text("504230\n", encoding="ascii")
        environment["SteamAppId"] = "504230"
        environment["SteamGameId"] = "504230"
    startup = subprocess.STARTUPINFO() if hasattr(subprocess, "STARTUPINFO") else None
    if startup:
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 1
    return subprocess.Popen(command, cwd=path, env=environment, startupinfo=startup)
