#!/bin/zsh
set -euo pipefail
project_dir="${0:A:h}"
cd "$project_dir"
python_bin="${CELESTE_PYTHON:-python3}"
if [[ -x "$project_dir/.venv/bin/python" && -z "${CELESTE_PYTHON:-}" ]]; then
    python_bin="$project_dir/.venv/bin/python"
elif [[ -x /Library/Frameworks/Python.framework/Versions/Current/bin/python3 && -z "${CELESTE_PYTHON:-}" ]]; then
    python_bin=/Library/Frameworks/Python.framework/Versions/Current/bin/python3
fi
launch_args=(--launch)
build_bridge=false
assist_args=()
for argument in "$@"; do
    case "$argument" in
        --build) build_bridge=true ;;
        --no-launch) launch_args=() ;;
        *) assist_args+=("$argument") ;;
    esac
done
if "$build_bridge"; then
    "$python_bin" "$project_dir/tools/build_mac_bridge.py"
fi
exec "$python_bin" "$project_dir/Celeste-Search/search.py" assist "${launch_args[@]}" "${assist_args[@]}"
