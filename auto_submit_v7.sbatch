#!/bin/bash
#SBATCH --job-name=auto_v7
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/auto_v7_%j.out
#SBATCH --error=logs/auto_v7_%j.err

# ══════════════════════════════════════════════════════════════════════
# V7 Smart Auto-Submitter
# - Mixes H200 + A100@80g per partition (1 A100 + rest H200 each)
# - Unemployed → gpu-short first, Employed → gpu first
# - Failed jobs → retry on gpu/multigpu with 8h
# - Self-resubmits if not done within 2 days
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail

PROJECT=/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7
PER_USER="${PROJECT}/outputs_v8/per_user"
SB="${PROJECT}/scripts/user_jobs"
ME=$(whoami)
INTERVAL=90

MAX_SHORT=4
MAX_GPU=8
MAX_MULTI=8

cd "$PROJECT"
mkdir -p "$PER_USER" logs
START_TIME=$(date +%s)

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

# ── Track: uid → "jobid:attempt" ──
declare -A TRACKED

# ── Track A100 usage per partition: partition → count of a100 jobs in queue ──
# We want exactly 1 A100 per partition at any time
A100_GRES="gpu:a100:1 --constraint=a100@80g"
H200_GRES="gpu:h200:1"

echo "$(date) | V7 Auto-Submitter: ${TOTAL} users"
echo "$(date) | GPU mix: 1×A100@80g + rest H200 per partition"
echo "$(date) | Partitions: short(${MAX_SHORT}) gpu(${MAX_GPU}) multi(${MAX_MULTI})"
echo "==========================================================="

submit_user() {
    local uid=$1 part=$2 tl=$3 att=$4 gres_type=$5
    local uid_short="${uid:0:12}"
    local sbfile="${SB}/u_${uid_short}.sbatch"
    [ ! -f "$sbfile" ] && echo "$(date) | MISSING $sbfile" && return 1

    local gres_flag
    if [ "$gres_type" == "a100" ]; then
        gres_flag="--gres=gpu:a100:1 --constraint=a100@80g"
    else
        gres_flag="--gres=gpu:h200:1"
    fi

    local out
    out=$(sbatch --partition="$part" --time="$tl" $gres_flag "$sbfile" 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
        local jid
        jid=$(echo "$out" | grep -oP '\d+')
        TRACKED["$uid"]="${jid}:${att}"
        local tag="emp"; [[ "${USER_EMP[$uid]}" == "0" ]] && tag="unemp"
        echo "$(date) | SUBMIT ${tag} ${uid_short} → ${part}(${tl}) ${gres_type} #${att} → ${jid}"
        return 0
    else
        echo "$(date) | FAIL ${uid_short}: $out"
        return 1
    fi
}

pick_gpu() {
    # Returns "a100" if partition has no a100 job yet, else "h200"
    local part=$1
    local n_a100
    n_a100=$(echo "$QCACHE" | awk -v p="$part" '$4==p' | grep -c "a100" 2>/dev/null || echo 0)
    if [ "$n_a100" -lt 1 ]; then
        echo "a100"
    else
        echo "h200"
    fi
}

while true; do
    # ── Cache squeue (include gres info) ──
    QCACHE=$(squeue -u "$ME" -h -o "%.20i %.200j %.10T %.20P %.30b" 2>/dev/null || echo "")

    # ── Count per-partition ──
    n_short=$(echo "$QCACHE" | awk '{print $4}' | grep -cx "gpu-short")
    n_gpu=$(echo "$QCACHE" | awk '{print $4}' | grep -cx "gpu")
    n_multi=$(echo "$QCACHE" | awk '{print $4}' | grep -cx "multigpu")
    s_short=$((MAX_SHORT - n_short)); [ "$s_short" -lt 0 ] && s_short=0
    s_gpu=$((MAX_GPU - n_gpu));       [ "$s_gpu" -lt 0 ]   && s_gpu=0
    s_multi=$((MAX_MULTI - n_multi)); [ "$s_multi" -lt 0 ] && s_multi=0

    n_done=0; n_run=0
    RETRY=(); FRESH_U=(); FRESH_E=()

    # ── Classify each user ──
    for uid in "${ALL_UIDS[@]}"; do
        uid_short="${uid:0:12}"

        [ -f "${PER_USER}/${uid}.npz" ] && n_done=$((n_done+1)) && continue

        if echo "$QCACHE" | grep -qF "u_${uid_short}"; then
            n_run=$((n_run+1))
            continue
        fi

        if [ -n "${TRACKED[$uid]:-}" ]; then
            info="${TRACKED[$uid]}"
            old_att="${info##*:}"
            new_att=$((old_att + 1))
            unset TRACKED["$uid"]
            if [ "$new_att" -le 3 ]; then
                RETRY+=("${uid}:${new_att}")
            else
                echo "$(date) | GIVEUP ${uid_short} after ${old_att} attempts"
            fi
            continue
        fi

        if [ "${USER_EMP[$uid]}" == "0" ]; then
            FRESH_U+=("$uid")
        else
            FRESH_E+=("$uid")
        fi
    done

    echo "$(date) | Done:${n_done}/${TOTAL} Run:${n_run} Retry:${#RETRY[@]} Fresh:$((${#FRESH_U[@]}+${#FRESH_E[@]})) | short=${s_short} gpu=${s_gpu} multi=${s_multi}"

    # All done?
    if [ "$n_done" -eq "$TOTAL" ]; then
        echo "$(date) | ALL ${TOTAL} DONE!"
        echo "$(date) | Run: python merge_user_results.py"
        exit 0
    fi

    # ── Retries → gpu(8h) or multigpu(8h) ──
    for entry in "${RETRY[@]}"; do
        r_uid="${entry%%:*}"; r_att="${entry##*:}"
        if [ "$s_gpu" -gt 0 ]; then
            gpu=$(pick_gpu "gpu")
            submit_user "$r_uid" "gpu" "8:00:00" "$r_att" "$gpu" && s_gpu=$((s_gpu-1))
        elif [ "$s_multi" -gt 0 ]; then
            gpu=$(pick_gpu "multigpu")
            submit_user "$r_uid" "multigpu" "8:00:00" "$r_att" "$gpu" && s_multi=$((s_multi-1))
        fi
        sleep 1
    done

    # ── Fresh unemployed → gpu-short(2h) → gpu(2h) → multigpu(2h) ──
    for uid in "${FRESH_U[@]}"; do
        if [ "$s_short" -gt 0 ]; then
            gpu=$(pick_gpu "gpu-short")
            submit_user "$uid" "gpu-short" "2:00:00" "1" "$gpu" && s_short=$((s_short-1))
        elif [ "$s_gpu" -gt 0 ]; then
            gpu=$(pick_gpu "gpu")
            submit_user "$uid" "gpu" "2:00:00" "1" "$gpu" && s_gpu=$((s_gpu-1))
        elif [ "$s_multi" -gt 0 ]; then
            gpu=$(pick_gpu "multigpu")
            submit_user "$uid" "multigpu" "2:00:00" "1" "$gpu" && s_multi=$((s_multi-1))
        else
            break
        fi
        sleep 0.5
    done

    # ── Fresh employed → gpu(2h) → multigpu(2h) → gpu-short(2h) ──
    for uid in "${FRESH_E[@]}"; do
        if [ "$s_gpu" -gt 0 ]; then
            gpu=$(pick_gpu "gpu")
            submit_user "$uid" "gpu" "2:00:00" "1" "$gpu" && s_gpu=$((s_gpu-1))
        elif [ "$s_multi" -gt 0 ]; then
            gpu=$(pick_gpu "multigpu")
            submit_user "$uid" "multigpu" "2:00:00" "1" "$gpu" && s_multi=$((s_multi-1))
        elif [ "$s_short" -gt 0 ]; then
            gpu=$(pick_gpu "gpu-short")
            submit_user "$uid" "gpu-short" "2:00:00" "1" "$gpu" && s_short=$((s_short-1))
        else
            break
        fi
        sleep 0.5
    done

    unset RETRY FRESH_U FRESH_E

    # ── Self-resubmit 30 min before 2-day limit ──
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -gt 169200 ]; then  # 47h = 169200s (leave 1h buffer)
        echo "$(date) | Approaching 2-day limit. Self-resubmitting..."
        sbatch auto_submit_v7.sbatch
        echo "$(date) | New auto-submitter queued. Exiting this one."
        exit 0
    fi

    sleep "$INTERVAL"
done

