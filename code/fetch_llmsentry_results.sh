#!/usr/bin/env bash
# fetch_llmsentry_results.sh
# Downloads all LLM-Sentry experiment results from the A100 back to
# the local results/ directory, then shows a brief summary.
#
# Author : Allan Douglas Costa (UFRA / LICA / SEC365)
# Project: LLM-Sentry — Software Supply Chain Security
set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
JUMP="ssh.recod.ic.unicamp.br"
USER="carlos.rocha"
TARGET="dl-28"
PASS_FILE="/tmp/.llmsentry_pass2"
TMUX_SESSION="llmsentry"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_RESULTS="$SCRIPT_DIR/../results"
REMOTE_WORKDIR="/tmp/llmsentry"

# ── Preflight ─────────────────────────────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    echo "[setup] Installing sshpass…"
    brew install hudochenkov/sshpass/sshpass 2>/dev/null || brew install sshpass 2>/dev/null
fi

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

_ssh() { sshpass -f "$PASS_FILE" ssh  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$USER@$TARGET" "$@"; }
_scp() { sshpass -f "$PASS_FILE" scp  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$@"; }
_rsync() {
    # rsync over SSH using sshpass via a wrapper script
    local wrap
    wrap="$(mktemp /tmp/sshwrap.XXXXXX.sh)"
    printf '#!/bin/sh\nsshpass -f "%s" ssh %s -o "ProxyCommand=%s" "$@"\n' \
        "$PASS_FILE" "$SSH_OPTS" "$PROXY_CMD" > "$wrap"
    chmod +x "$wrap"
    rsync -avz --progress -e "$wrap" "$@"
    rm -f "$wrap"
}

mkdir -p "$LOCAL_RESULTS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LLM-Sentry — Fetching Results from A100"
echo "  Source : $USER@$TARGET:$REMOTE_WORKDIR/results/"
echo "  Dest   : $LOCAL_RESULTS/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Check experiment status ───────────────────────────────────────────────────
echo "[1/4] Checking experiment status…"
STATUS=$(_ssh "bash -c '
    if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        echo RUNNING
    elif [[ -f $REMOTE_WORKDIR/results/run.log ]] && grep -q DONE $REMOTE_WORKDIR/results/run.log 2>/dev/null; then
        echo DONE
    elif [[ -f $REMOTE_WORKDIR/results/run.log ]]; then
        echo IN_PROGRESS
    else
        echo NOT_STARTED
    fi
'" 2>/dev/null || echo "UNKNOWN")

STATUS="${STATUS//$'\r'/}"   # strip CR
echo "      Status: $STATUS"

if [[ "$STATUS" == "NOT_STARTED" ]]; then
    echo ""
    echo "ERROR: Experiment has not been started yet."
    echo "       Run ./deploy_llmsentry_a100.sh first."
    exit 1
fi

if [[ "$STATUS" == "RUNNING" ]]; then
    echo ""
    echo "WARNING: Experiment is still running in tmux session '$TMUX_SESSION'."
    echo "         Fetching whatever results exist so far…"
    echo "         Re-run this script after the experiment completes for full results."
    echo ""
fi

# ── Download results ──────────────────────────────────────────────────────────
echo "[2/4] Downloading result files…"

# Prefer rsync for incremental downloads; fall back to scp
if command -v rsync &>/dev/null; then
    _rsync "$USER@$TARGET:$REMOTE_WORKDIR/results/" "$LOCAL_RESULTS/" 2>/dev/null || {
        echo "      rsync failed — falling back to scp"
        _scp -r "$USER@$TARGET:$REMOTE_WORKDIR/results/." "$LOCAL_RESULTS/" || {
            echo "ERROR: Could not download results."
            exit 1
        }
    }
else
    _scp -r "$USER@$TARGET:$REMOTE_WORKDIR/results/." "$LOCAL_RESULTS/" || {
        echo "ERROR: Could not download results (scp failed)."
        exit 1
    }
fi
echo "      ✓ download complete"

# ── Download split CSVs (for reproducibility) ────────────────────────────────
echo "[3/4] Downloading split CSVs…"
for f in split_train.csv split_val.csv split_test.csv; do
    _scp "$USER@$TARGET:$REMOTE_WORKDIR/results/$f" "$LOCAL_RESULTS/$f" 2>/dev/null && \
        echo "      ✓ $f" || echo "      - $f not yet available"
done

# ── Show summary ──────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Results summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# File inventory
echo ""
echo "Files in $LOCAL_RESULTS:"
if ls "$LOCAL_RESULTS" &>/dev/null; then
    ls -lh "$LOCAL_RESULTS" | awk 'NR>1 {printf "  %-40s %s\n", $NF, $5}'
fi

# Print test metrics if available
METRICS_FILE="$LOCAL_RESULTS/test_metrics.json"
if [[ -f "$METRICS_FILE" ]]; then
    echo ""
    echo "Test-set metrics (bootstrap 95% CI):"
    echo "  $(printf '%-12s %8s  %20s' 'Metric' 'Mean' '95% CI')"
    echo "  $(printf '%0.s-' {1..44})"
    python3 - "$METRICS_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
for k in ["f1","precision","recall","fpr","auc_roc","pr_auc"]:
    if k in data:
        v = data[k]
        print(f"  {k:<12} {v['mean']:>8.4f}  [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
PYEOF
fi

# Print ablation summary if available
ABLATION_FILE="$LOCAL_RESULTS/ablation_results.json"
if [[ -f "$ABLATION_FILE" ]]; then
    echo ""
    echo "Ablation study:"
    echo "  $(printf '%-22s %8s  %8s' 'Variant' 'F1' 'AUC-ROC')"
    echo "  $(printf '%0.s-' {1..42})"
    python3 - "$ABLATION_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
for vname, m in data.items():
    print(f"  {vname:<22} {m['f1']:>8.4f}  {m['auc_roc']:>8.4f}")
PYEOF
fi

# Print optimal weights if available
WEIGHTS_FILE="$LOCAL_RESULTS/prcs_optimal_weights.json"
if [[ -f "$WEIGHTS_FILE" ]]; then
    echo ""
    python3 - "$WEIGHTS_FILE" <<'PYEOF'
import json, sys
w = json.load(open(sys.argv[1]))
print(f"Optimal PRCS weights: w1(meta)={w.get('w1_meta','-'):.2f}  "
      f"w2(sem)={w.get('w2_sem','-'):.2f}  w3(beh)={w.get('w3_beh','-'):.2f}")
PYEOF
fi

# Tail the last 20 lines of the run log
LOG_FILE="$LOCAL_RESULTS/run.log"
if [[ -f "$LOG_FILE" ]]; then
    echo ""
    echo "Last 20 lines of run.log:"
    echo "  ---"
    tail -20 "$LOG_FILE" | sed 's/^/  /'
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  To commit results:"
echo "    cd \"$(dirname "$SCRIPT_DIR")\""
echo "    git add results/ && git commit -m 'results: LLM-Sentry A100 full evaluation'"
echo "    git push"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
