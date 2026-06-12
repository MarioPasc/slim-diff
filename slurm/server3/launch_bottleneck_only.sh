#!/usr/bin/env bash
# =============================================================================
# launch_bottleneck_only.sh — local entry point for the PRL bottleneck-only run
# =============================================================================
# Reads .claude/server3.yaml for paths and SSH alias. Performs:
#   1. rsync local repo  → server3 remote_repo
#   2. rsync local data  → server3 remote_data  (skippable: --no-data-sync)
#   3. pip install -e .  on server3 so the new slimdiff-train entrypoint dispatch is live
#   4. ssh server3 tmux new-session -d "bash worker_bottleneck_only.sh ..."
# The tmux session persists after SSH closes — monitor via:
#       ssh icai-server tmux attach -t slimdiff-bottleneck-only-${MODE}
#
# Usage:
#   bash slurm/server3/launch_bottleneck_only.sh [--smoke|--prod] [--folds 0,1,2]
#                                                [--dry-run] [--no-data-sync]
#                                                [--no-code-sync]
# =============================================================================

set -euo pipefail

MODE="prod"
FOLDS="0,1,2"
DRY_RUN=0
SYNC_DATA=1
SYNC_CODE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)         MODE=smoke;       shift ;;
        --prod)          MODE=prod;        shift ;;
        --folds)         FOLDS="$2";       shift 2 ;;
        --dry-run)       DRY_RUN=1;        shift ;;
        --no-data-sync)  SYNC_DATA=0;      shift ;;
        --no-code-sync)  SYNC_CODE=0;      shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CFG="${REPO_ROOT}/.claude/server3.yaml"
if [[ ! -f "${CFG}" ]]; then
    echo "ERROR: ${CFG} missing" >&2
    exit 1
fi

# Load schema. Excludes are joined into a single space-separated rsync arg.
eval "$(python - <<PY "${CFG}"
import sys, shlex, yaml
c = yaml.safe_load(open(sys.argv[1]))["server3"]
for k in ("ssh_alias", "local_repo", "remote_repo", "local_data", "remote_data",
         "remote_results", "conda_env", "gpu_id", "tmux_session"):
    print(f"export {k.upper()}={shlex.quote(str(c[k]))}")
excl = " ".join(f"--exclude={shlex.quote(e)}" for e in c.get("rsync_excludes", []))
print(f"export RSYNC_EXCLUDES={shlex.quote(excl)}")
PY
)"

run() {
    echo "+ $*"
    if [[ ${DRY_RUN} -ne 1 ]]; then eval "$@"; fi
}

SESSION="${TMUX_SESSION}-bottleneck-only-${MODE}"

# server3 has GNU screen, not tmux. Use screen -dmS for detached sessions.
# Sessions survive SSH disconnect, are listable via `screen -ls`, attach via
# `screen -r ${SESSION}` (detach with ctrl-a d). The session name shows up
# in `screen -ls` as <PID>.${SESSION}.

echo "=========================================================================="
echo "[launch_bottleneck_only] $(date)"
echo "  mode          : ${MODE}"
echo "  folds         : ${FOLDS}"
echo "  ssh alias     : ${SSH_ALIAS}"
echo "  local_repo    : ${LOCAL_REPO}"
echo "  remote_repo   : ${REMOTE_REPO}"
echo "  local_data    : ${LOCAL_DATA}"
echo "  remote_data   : ${REMOTE_DATA}"
echo "  remote_results: ${REMOTE_RESULTS}"
echo "  conda_env     : ${CONDA_ENV}"
echo "  gpu_id        : ${GPU_ID}"
echo "  tmux session  : ${SESSION}"
echo "  dry-run       : ${DRY_RUN}"
echo "=========================================================================="

if [[ ${SYNC_CODE} -eq 1 ]]; then
    run "rsync -avz ${RSYNC_EXCLUDES} '${LOCAL_REPO}/' '${SSH_ALIAS}:${REMOTE_REPO}/'"
else
    echo "[skip] code sync (--no-code-sync)"
fi

if [[ ${SYNC_DATA} -eq 1 ]]; then
    run "rsync -avz '${LOCAL_DATA}/' '${SSH_ALIAS}:${REMOTE_DATA}/'"
else
    echo "[skip] data sync (--no-data-sync)"
fi

# Reinstall the package on remote so the slimdiff-train console_script picks up
# any new dispatch branches (e.g. BottleneckSharedTwinDDPM in train.py).
run "ssh '${SSH_ALIAS}' 'bash -lc \"cd ${REMOTE_REPO} && conda run -n ${CONDA_ENV} pip install -e . 2>&1 | tail -3\"'"

# Verify there's no live session of the same name. screen -ls exits non-zero
# when no matching session exists; we wrap with `|| true` and grep.
EXISTING=$(ssh "${SSH_ALIAS}" "screen -ls 2>/dev/null | grep -Eo '[0-9]+\\.${SESSION}' || true")
if [[ -n "${EXISTING}" ]]; then
    echo "ERROR: screen session '${SESSION}' already exists on ${SSH_ALIAS} (${EXISTING})." >&2
    echo "       Kill it first:  ssh ${SSH_ALIAS} screen -X -S ${SESSION} quit" >&2
    exit 1
fi

# Write the remote launcher script to a file on server3, then run it via
# `screen -dmS`. This avoids nested-quote escaping hell over SSH.
LAUNCH_TMP="/tmp/slimdiff_launch_${SESSION}.sh"
LOG_FILE="${REMOTE_RESULTS}/worker_${SESSION}.log"

read -r -d '' REMOTE_SCRIPT <<EOF || true
#!/usr/bin/env bash
export REMOTE_REPO=${REMOTE_REPO}
export REMOTE_DATA=${REMOTE_DATA}
export REMOTE_RESULTS=${REMOTE_RESULTS}
export CONDA_ENV=${CONDA_ENV}
export GPU_ID=${GPU_ID}
mkdir -p ${REMOTE_RESULTS}
cd ${REMOTE_REPO}
exec bash slurm/server3/worker_bottleneck_only.sh --mode ${MODE} --folds ${FOLDS} >> ${LOG_FILE} 2>&1
EOF

if [[ ${DRY_RUN} -ne 1 ]]; then
    printf '%s' "${REMOTE_SCRIPT}" | ssh "${SSH_ALIAS}" "cat > ${LAUNCH_TMP} && chmod +x ${LAUNCH_TMP}"
    ssh "${SSH_ALIAS}" "bash -lc 'mkdir -p ${REMOTE_RESULTS} && screen -dmS ${SESSION} bash ${LAUNCH_TMP}'"
    sleep 2
    ssh "${SSH_ALIAS}" "bash -lc 'screen -ls' || true"
else
    echo "+ [dry-run] would write ${LAUNCH_TMP} and screen -dmS ${SESSION} bash ${LAUNCH_TMP}"
fi

echo ""
echo "=========================================================================="
echo "Launched in detached GNU screen session."
echo "Monitor with:"
echo "  ssh ${SSH_ALIAS} bash -lc \"screen -ls\""
echo "  ssh ${SSH_ALIAS} bash -lc \"screen -r ${SESSION}\"    # ctrl-a d to detach"
echo "  ssh ${SSH_ALIAS} tail -f ${LOG_FILE}"
echo ""
echo "Kill (if needed):"
echo "  ssh ${SSH_ALIAS} bash -lc \"screen -X -S ${SESSION} quit\""
echo "=========================================================================="
