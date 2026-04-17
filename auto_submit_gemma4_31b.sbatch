#!/bin/bash
#SBATCH --job-name=auto_g4_31b
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/auto_g4_31b_%j.out
#SBATCH --error=logs/auto_g4_31b_%j.err

# ══════════════════════════════════════════════════════════════════════
# Gemma 4 31B-it BnB-4bit Robust Auto-Submitter v3
#
# Strategy:
#   - ALL jobs get 2h (shorter = easier card assignment)
#   - 3 partitions: gpu-short(4) + gpu(8) + multigpu(8) = 20 slots
#   - run_extract_per_user.py auto-skips completed iterations:
#       1st run: does V7 (saves npz) → starts V8 → times out
#       2nd run: skips V7 ("Already done") → does V8 → fits in 2h
#   - Auto-detects & deletes npz with OOM samples → resubmits clean
#   - Clear logging: every OOM/timeout/cleanup printed with ⚠/↻/✗
#   - Max 4 attempts per user
#   - Self-resubmits near 2-day SLURM limit
#
# Output: gemma-4-31B-it-unsloth-bnb-4bit/outputs_{v7,v8}/per_user/
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail

PROJECT=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
MODEL_DIR="${PROJECT}/gemma-4-31B-it-unsloth-bnb-4bit"
V7_DIR="${MODEL_DIR}/outputs_v7/per_user"
V8_DIR="${MODEL_DIR}/outputs_v8/per_user"
SB="${PROJECT}/scripts/gemma4_jobs"
ME=$(whoami)
INTERVAL=90
TIME_LIMIT="02:00:00"

MAX_SHORT=4
MAX_GPU=8
MAX_MULTI=8

cd "$PROJECT"
mkdir -p "$V7_DIR" "$V8_DIR" "$SB" logs
START_TIME=$(date +%s)

# ── Generate job scripts if missing ──
if [ ! "$(ls -A "$SB" 2>/dev/null)" ]; then
    echo "$(date) | Generating job scripts..."
    python generate_user_jobs_gemma4.py --align-v7
fi

# ── Load users ──
declare -A USER_EMP
declare -a ALL_UIDS=()
first=1
while IFS=, read -r uid emp rest; do
    if [ "$first" -eq 1 ]; then first=0; continue; fi
    USER_EMP["$uid"]="$emp"
    ALL_UIDS+=("$uid")
done < data/users.csv
TOTAL=${#ALL_UIDS[@]}

declare -A TRACKED

echo "$(date) | ════════════════════════════════════════════════════"
echo "$(date) | Gemma4-31B Robust Auto-Submitter v3"
echo "$(date) | Users: ${TOTAL} | Time: ${TIME_LIMIT}"
echo "$(date) | Slots: short(${MAX_SHORT}) gpu(${MAX_GPU}) multi(${MAX_MULTI}) = $((MAX_SHORT+MAX_GPU+MAX_MULTI)) total"
echo "$(date) | Strategy: 2h + auto-skip completed iters + OOM cleanup"
echo "$(date) | ════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════
# Validate npz — returns 0 if clean, 1 if bad (deletes bad files)
# ══════════════════════════════════════════════════════════════════════
validate_npz() {
    local path=$1
    local label=$2
    [ ! -f "$path" ] && return 1

    local result
    result=$(python -c "
import numpy as np, sys
try:
    d = np.load('$path', allow_pickle=True)
    a = d['answers']
    n_oom = sum(1 for x in a if x.get('status') == 'oom')
    n_ok  = sum(1 for x in a if x.get('status') == 'ok')
    n_total = len(a)
    if n_oom > 0:
        print(f'OOM:{n_oom}/{n_total}')
        sys.exit(1)
    if n_ok == 0:
        print(f'EMPTY:0/{n_total}')
        sys.exit(1)
    print(f'OK:{n_ok}/{n_total}')
    sys.exit(0)
except Exception as e:
    print(f'CORRUPT:{e}')
    sys.exit(1)
" 2>&1)

    if [ $? -ne 0 ]; then
        local uid_short
        uid_short=$(basename "$path" .npz | cut -c1-12)
        echo "$(date) | ⚠ INVALID ${label} ${uid_short}: ${result} → DELETING for re-run"
        rm -f "$path"
        return 1
    fi
    return 0
}

# ══════════════════════════════════════════════════════════════════════
# Check if user is fully done
# ══════════════════════════════════════════════════════════════════════
is_user_done() {
    local uid=$1
    local v7="${V7_DIR}/${uid}.npz"
    local v8="${V8_DIR}/${uid}.npz"
    [ ! -f "$v7" ] && return 1
    [ ! -f "$v8" ] && return 1
    validate_npz "$v7" "v7" || return 1
    validate_npz "$v8" "v8" || return 1
    return 0
}

# ══════════════════════════════════════════════════════════════════════
# Status string for logging
# ══════════════════════════════════════════════════════════════════════
user_status() {
    local uid=$1
    local s=""
    [ -f "${V7_DIR}/${uid}.npz" ] && s="v7✓" || s="v7✗"
    [ -f "${V8_DIR}/${uid}.npz" ] && s="${s}+v8✓" || s="${s}+v8✗"
    echo "$s"
}

# ══════════════════════════════════════════════════════════════════════
# Submit one user
# ══════════════════════════════════════════════════════════════════════
submit_user() {
    local uid=$1 part=$2 att=$3
    local uid_short="${uid:0:12}"
    local sbfile="${SB}/g4_${uid_short}.sbatch"
    if [ ! -f "$sbfile" ]; then
        echo "$(date) | MISSING $sbfile"
        return 1
    fi

    local out
    out=$(sbatch --partition="$part" --time="${TIME_LIMIT}" --gres="gpu:h200:1" "$sbfile" 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
        local jid
        jid=$(echo "$out" | grep -oP '\d+')
        TRACKED["$uid"]="${jid}:${att}"
        local tag="emp"; [[ "${USER_EMP[$uid]}" == "0" ]] && tag="unemp"
        local stat
        stat=$(user_status "$uid")
        echo "$(date) | → SUBMIT ${tag} ${uid_short} [${stat}] → ${part}(${TIME_LIMIT}) #${att} → job ${jid}"
        return 0
    else
        echo "$(date) | ✗ FAIL ${uid_short}: $out"
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════
CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    QCACHE=$(squeue -u "$ME" -h -o "%.20i %.200j %.10T %.20P" 2>/dev/null || echo "")

    n_short=$(echo "$QCACHE" | awk '{print $4}' | grep -cx "gpu-short")
    n_gpu=$(echo "$QCACHE" | awk '{print $4}' | grep -cx "gpu")
    n_multi=$(echo "$QCACHE" | awk '{print $4}' | grep -cx "multigpu")
    s_short=$((MAX_SHORT - n_short)); [ "$s_short" -lt 0 ] && s_short=0
    s_gpu=$((MAX_GPU - n_gpu));       [ "$s_gpu" -lt 0 ]   && s_gpu=0
    s_multi=$((MAX_MULTI - n_multi)); [ "$s_multi" -lt 0 ] && s_multi=0

    n_done=0; n_run=0; n_v7only=0; n_v8only=0
    RETRY=(); FRESH_U=(); FRESH_E=()

    for uid in "${ALL_UIDS[@]}"; do
        uid_short="${uid:0:12}"

        # ── Fully done? ──
        if is_user_done "$uid"; then
            n_done=$((n_done+1))
            continue
        fi

        # ── Partial status ──
        [ -f "${V7_DIR}/${uid}.npz" ] && [ ! -f "${V8_DIR}/${uid}.npz" ] && n_v7only=$((n_v7only+1))
        [ ! -f "${V7_DIR}/${uid}.npz" ] && [ -f "${V8_DIR}/${uid}.npz" ] && n_v8only=$((n_v8only+1))

        # ── In queue? ──
        if echo "$QCACHE" | grep -qF "g4_${uid_short}"; then
            n_run=$((n_run+1))
            continue
        fi

        # ── Was tracked, left queue → timed out or crashed ──
        if [ -n "${TRACKED[$uid]:-}" ]; then
            info="${TRACKED[$uid]}"
            old_att="${info##*:}"
            new_att=$((old_att + 1))
            unset TRACKED["$uid"]

            stat=$(user_status "$uid")

            if [ "$new_att" -le 4 ]; then
                RETRY+=("${uid}:${new_att}")
                echo "$(date) | ↻ REQUEUE ${uid_short} [${stat}] → attempt #${new_att} (prev timed out or finished partial)"
            else
                echo "$(date) | ✗ GIVEUP ${uid_short} [${stat}] after ${old_att} attempts"
            fi
            continue
        fi

        # ── Fresh ──
        if [ "${USER_EMP[$uid]}" == "0" ]; then
            FRESH_U+=("$uid")
        else
            FRESH_E+=("$uid")
        fi
    done

    # ── Status line ──
    echo "$(date) | [C${CYCLE}] Done:${n_done}/${TOTAL} Run:${n_run} V7only:${n_v7only} Retry:${#RETRY[@]} Fresh:$((${#FRESH_U[@]}+${#FRESH_E[@]})) | short=${s_short} gpu=${s_gpu} multi=${s_multi}"

    # ── All done? ──
    if [ "$n_done" -eq "$TOTAL" ]; then
        echo ""
        echo "$(date) | ════════════════════════════════════════════════════"
        echo "$(date) | ✓ ALL ${TOTAL} USERS COMPLETE (V7 + V8, 0 OOM)"
        echo "$(date) | ════════════════════════════════════════════════════"
        echo "$(date) | Output: ${MODEL_DIR}/outputs_{v7,v8}/per_user/"
        echo "$(date) |"
        echo "$(date) | Next steps:"
        echo "$(date) |   python merge_user_results.py --model gemma4_31b --iter v7 --exp exp1a exp2"
        echo "$(date) |   python merge_user_results.py --model gemma4_31b --iter v8 --exp exp1a exp2"
        echo "$(date) |   python prob_results.py --model gemma4_31b --iter v7 --exp exp2"
        echo "$(date) |   python prediction_results.py --model gemma4_31b --iter v7 --exp exp2"
        exit 0
    fi

    # ── Retries first (timed out users — likely just need V8) ──
    for entry in "${RETRY[@]}"; do
        r_uid="${entry%%:*}"; r_att="${entry##*:}"
        if [ "$s_gpu" -gt 0 ]; then
            submit_user "$r_uid" "gpu" "$r_att" && s_gpu=$((s_gpu-1))
        elif [ "$s_multi" -gt 0 ]; then
            submit_user "$r_uid" "multigpu" "$r_att" && s_multi=$((s_multi-1))
        elif [ "$s_short" -gt 0 ]; then
            submit_user "$r_uid" "gpu-short" "$r_att" && s_short=$((s_short-1))
        fi
        sleep 1
    done

    # ── Fresh unemployed (~60min V7+V8): short → gpu → multi ──
    for uid in "${FRESH_U[@]}"; do
        if [ "$s_short" -gt 0 ]; then
            submit_user "$uid" "gpu-short" "1" && s_short=$((s_short-1))
        elif [ "$s_gpu" -gt 0 ]; then
            submit_user "$uid" "gpu" "1" && s_gpu=$((s_gpu-1))
        elif [ "$s_multi" -gt 0 ]; then
            submit_user "$uid" "multigpu" "1" && s_multi=$((s_multi-1))
        else break; fi
        sleep 0.5
    done

    # ── Fresh employed (~90-120min): gpu → multi → short ──
    for uid in "${FRESH_E[@]}"; do
        if [ "$s_gpu" -gt 0 ]; then
            submit_user "$uid" "gpu" "1" && s_gpu=$((s_gpu-1))
        elif [ "$s_multi" -gt 0 ]; then
            submit_user "$uid" "multigpu" "1" && s_multi=$((s_multi-1))
        elif [ "$s_short" -gt 0 ]; then
            submit_user "$uid" "gpu-short" "1" && s_short=$((s_short-1))
        else break; fi
        sleep 0.5
    done

    unset RETRY FRESH_U FRESH_E

    # ── Self-resubmit near 2-day limit ──
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -gt 169200 ]; then
        echo "$(date) | Approaching 2-day limit. Self-resubmitting..."
        sbatch auto_submit_gemma4_31b.sbatch
        echo "$(date) | New auto-submitter queued. Exiting."
        exit 0
    fi

    sleep "$INTERVAL"
done
