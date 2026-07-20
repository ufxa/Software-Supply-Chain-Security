#!/usr/bin/env bash
# deploy_llmsentry_a100.sh
# Copies a100_runner.py + dataset CSVs to the A100 and starts the
# experiment in a tmux session named "llmsentry".
#
# Author : Allan Douglas Costa (UFRA / LICA / SEC365)
# Project: LLM-Sentry — Software Supply Chain Security
set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
JUMP="ssh.recod.ic.unicamp.br"
USER="carlos.rocha"
TARGET="dl-28"
TMUX_SESSION="llmsentry"
PASS_FILE="/tmp/.llmsentry_pass2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/a100_runner.py"
DATA_DIR="$SCRIPT_DIR/../data"

REMOTE_WORKDIR="/tmp/llmsentry"

# Dataset CSVs to upload (upload any that exist locally)
DATA_FILES=(
    "mboss_labels.csv"
    "pmd_labels.csv"
    "benign_baseline.csv"
)

# ── Preflight checks ──────────────────────────────────────────────────────────
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: Runner script not found: $RUNNER"
    exit 1
fi

# Ensure sshpass is available
if ! command -v sshpass &>/dev/null; then
    echo "[setup] Installing sshpass…"
    brew install hudochenkov/sshpass/sshpass 2>/dev/null || \
    brew install sshpass 2>/dev/null || \
    { echo "ERROR: Could not install sshpass. Install manually and retry."; exit 1; }
fi

# Write password to temp file (avoids shell quoting issues in ProxyCommand)
printf '%s' "::&7}__'wE64g^MBc;" > "$PASS_FILE"
chmod 600 "$PASS_FILE"
trap 'rm -f "$PASS_FILE"' EXIT

SSH_OPTS="-o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null \
          -o LogLevel=ERROR \
          -o PubkeyAuthentication=no \
          -o PreferredAuthentications=password \
          -o ConnectTimeout=30"

PROXY_CMD="sshpass -f $PASS_FILE ssh $SSH_OPTS -W %h:%p $USER@$JUMP"

_ssh()  { sshpass -f "$PASS_FILE" ssh  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$USER@$TARGET" "$@"; }
_scp()  { sshpass -f "$PASS_FILE" scp  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$@"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LLM-Sentry A100 Deploy"
echo "  Target : $USER@$TARGET  (via $JUMP)"
echo "  Session: $TMUX_SESSION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Create remote working directory ───────────────────────────────────
echo "[1/4] Preparing remote working directory $REMOTE_WORKDIR…"
_ssh "bash -c 'mkdir -p $REMOTE_WORKDIR/results $REMOTE_WORKDIR/data'"
echo "      ✓ remote dirs ready"

# ── Step 2: Upload runner script ──────────────────────────────────────────────
echo "[2/4] Uploading a100_runner.py…"
_scp "$RUNNER" "$USER@$TARGET:$REMOTE_WORKDIR/a100_runner.py"
echo "      ✓ a100_runner.py uploaded"

# ── Step 3: Upload dataset CSVs (skip if not present locally) ─────────────────
echo "[3/4] Uploading dataset CSVs…"
uploaded=0
for csv in "${DATA_FILES[@]}"; do
    local_path="$DATA_DIR/$csv"
    if [[ -f "$local_path" ]]; then
        _scp "$local_path" "$USER@$TARGET:$REMOTE_WORKDIR/data/$csv" && \
        echo "      ✓ $csv" && ((uploaded++)) || \
        echo "      ! WARNING: failed to upload $csv"
    else
        echo "      - $csv not found locally — skipping (synthetic data will be used)"
    fi
done
if [[ $uploaded -eq 0 ]]; then
    echo "      (no CSVs uploaded — runner will use fully synthetic dataset)"
fi

# ── Step 4: Kill any previous session and start fresh ─────────────────────────
echo "[4/4] Starting tmux session '$TMUX_SESSION'…"

# Kill previous session + any stale runner processes
_ssh "bash -c '
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
    pkill -f a100_runner.py 2>/dev/null || true
    sleep 1
'"
echo "      ✓ previous session cleaned"

# Launch new tmux session running the experiment
_ssh "bash -c '
    cd $REMOTE_WORKDIR
    tmux new-session -d -s $TMUX_SESSION \
        \"python3 $REMOTE_WORKDIR/a100_runner.py 2>&1 | tee $REMOTE_WORKDIR/results/run.log; echo DONE; bash\"
'"
echo "      ✓ tmux session '$TMUX_SESSION' started"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Experiment is running on $TARGET!"
echo ""
echo "  Live monitoring:"
echo "    ssh -J $USER@$JUMP $USER@$TARGET"
echo "    tmux attach -t $TMUX_SESSION"
echo ""
echo "  Tail log (non-interactive):"
echo "    sshpass -f $PASS_FILE ssh $SSH_OPTS \\"
echo "      -o \"ProxyCommand=$PROXY_CMD\" \\"
echo "      $USER@$TARGET \\"
echo "      'tail -f $REMOTE_WORKDIR/results/run.log'"
echo ""
echo "  Expected runtime: 45–90 min on A100"
echo "  Results will be in: $REMOTE_WORKDIR/results/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
