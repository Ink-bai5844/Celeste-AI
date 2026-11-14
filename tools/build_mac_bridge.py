"""Build and inject the FNA bridge into a project-local copy of owned Celeste."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Celeste-Search"))
from celeste_search.game_platform import MAC_BRIDGE_FILES, normalize_game_dir, require_game_stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-app", type=Path, default=Path.home() / "Library/Application Support/Steam/steamapps/common/Celeste/Celeste.app")
    parser.add_argument("--game-app", type=Path, default=ROOT / "GAME/Celeste-Mac/Celeste.app")
    parser.add_argument("--dotnet", type=Path, default=ROOT / ".dev/dotnet/dotnet")
    args = parser.parse_args()
    dotnet = args.dotnet.expanduser().resolve()
    source_app = args.source_app.expanduser().resolve()
    target_app = args.game_app.expanduser().resolve()
    source = normalize_game_dir(source_app)
    target = normalize_game_dir(target_app)
    if not target.is_relative_to((ROOT / "GAME").resolve()) or target_app == source_app:
        parser.error("The destination must be a separate app inside this project's GAME directory")
    if target_app.suffix.lower() != ".app" or source_app.suffix.lower() != ".app":
        parser.error("--source-app and --game-app must name .app bundles")
    for path in (source / "Celeste.exe", source / "FNA.dll", dotnet):
        if not path.is_file():
            parser.error(f"Required local file is missing: {path}")
    # Loaded assemblies are not replaced; stop the selected copy normally first.
    if target_app.exists():
        try:
            require_game_stopped(target)
        except ValueError:
            parser.error("The selected game copy is running; close it normally before rebuilding its bridge")
    if not target_app.exists():
        target_app.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_app, target_app)
    environment = os.environ.copy()
    environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    environment["DOTNET_CLI_HOME"] = str(ROOT / ".dev/dotnet-home")
    environment["NUGET_PACKAGES"] = str(ROOT / ".dev/nuget")
    build = ROOT / "artifacts/bridge-mac"
    project = ROOT / "Celeste-Instrumentation/CelesteInstrumentation/CelesteInstrumentation/CelesteInstrumentation.csproj"
    installer = ROOT / "Celeste-Instrumentation/tools/BridgeInstaller/BridgeInstaller.csproj"
    subprocess.run([str(dotnet), "build", str(project), "-c", "Release", "-p:CelesteFramework=FNA",
                    f"-p:CelesteGameDir={source}", "-o", str(build)], cwd=ROOT, env=environment, check=True)
    subprocess.run([str(dotnet), "build", str(installer), "-c", "Release"], cwd=ROOT, env=environment, check=True)
    report = subprocess.run([str(dotnet), str(installer.parent / "bin/Release/net9.0/BridgeInstaller.dll"),
                             "patch", str(source / "Celeste.exe"), str(build / "CelesteInstrumentation.dll"),
                             str(target / "Celeste.exe")], cwd=ROOT, env=environment, check=True,
                            capture_output=True, text=True)
    parsed = json.loads(report.stdout)
    dependencies = []
    for name in MAC_BRIDGE_FILES:
        shutil.copy2(build / name, target / name)
        payload = (target / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != hashlib.sha256((build / name).read_bytes()).hexdigest():
            raise OSError(f"Installed bridge dependency failed verification: {name}")
        dependencies.append({"name": name, "sha256": digest, "size": len(payload)})
    (target / "steam_appid.txt").write_text("504230\n", encoding="ascii")
    output = ROOT / "artifacts/mac-port/bridge-installation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_name("bridge-dependencies.json").write_text(
        json.dumps(dependencies, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FNA bridge prepared in {target}; Steam originals unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
