#!/bin/bash
# ================================================================
# EEG-FM 多数据集混合二分类
#
# 用法:
#   bash scripts/run_multi_dataset.sh                  # 两个都跑, 默认 linear 头
#   bash scripts/run_multi_dataset.sh --head mlp       # 两个都用 MLP 头
#   bash scripts/run_multi_dataset.sh --diagnosis depression           # 只跑抑郁症
#   bash scripts/run_multi_dataset.sh -g adhd --head cnn1d             # 只跑 ADHD + CNN1D 头
# ================================================================

set -e
cd "$(dirname "$0")/.."

# ================================================================
# 参数解析
# ================================================================
FILTER_DIAGNOSIS=""
HEAD_TYPE="linear"

while [[ $# -gt 0 ]]; do
    case $1 in
        --diagnosis|-g)
            FILTER_DIAGNOSIS="$2"
            shift 2
            ;;
        --head_type|--head)
            HEAD_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "多数据集混合二分类 — 每个病种只跑一次"
            echo ""
            echo "用法: bash scripts/run_multi_dataset.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --diagnosis, -g  诊断类型 (depression, adhd)，不传则两个都跑"
            echo "  --head, --head_type  分类头类型: linear (默认), mlp, cnn1d, attention"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FIXED_RESULT="outputs/results/multi_binary_summary.csv"

echo "=========================================="
echo " 多数据集混合二分类"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 结果: $FIXED_RESULT"
echo " 分类头: $HEAD_TYPE"
[[ -n "$FILTER_DIAGNOSIS" ]] && echo " 筛选诊断: $FILTER_DIAGNOSIS"
echo "=========================================="

# 如果文件不存在，写入表头
HEADER="dataset,diagnosis,head_type,balanced_acc,auroc,f1_weighted,subject_acc"
if [[ ! -f "$FIXED_RESULT" ]]; then
    echo "$HEADER" > "$FIXED_RESULT"
fi

should_run() {
    local diag=$1
    [[ -z "$FILTER_DIAGNOSIS" ]] && return 0
    local diag_lower="${diag,,}"
    local filter_lower="${FILTER_DIAGNOSIS,,}"
    [[ "$diag_lower" == *"$filter_lower"* ]]
}

run_one() {
    local DATASETS="$1"
    local DIAG="$2"
    local NAME="$3"
    local BS=${4:-256}

    # 名称包含 head_type 后缀（linear 不添加以保持兼容）
    local SUFFIX=""
    if [[ "$HEAD_TYPE" != "linear" ]]; then
        SUFFIX="_${HEAD_TYPE}"
        NAME="${NAME}${SUFFIX}"
    fi

    echo ""
    echo "===== [$NAME] (head=$HEAD_TYPE) ====="
    echo "  Datasets: $DATASETS"
    echo "  Diagnosis: $DIAG"

    rm -rf "outputs/checkpoints/$NAME"

    python models/train.py \
        --datasets "$DATASETS" \
        --diagnosis "$DIAG" \
        --head_type "$HEAD_TYPE" \
        --local_files_only \
        --batch_size "$BS" \
        --name "$NAME" \
        ${MAX_CH:+"--max_channels" "$MAX_CH"}

    # 评估
    CKPT="outputs/checkpoints/$NAME/best_model.pt"
    echo ""
    echo "--- 评估 $NAME ---"

    python scripts/evaluate_model.py \
        --checkpoint "$CKPT" \
        --datasets "$DATASETS" \
        --output_dir "outputs/results/$NAME" \
        ${MAX_CH:+"--max_channels" "$MAX_CH"}

    JSON="outputs/results/$NAME/evaluation.json"

    # 提取每个数据集的 test 指标，upsert 到固定 CSV
    python -c "
import json
data = json.load(open('$JSON'))
per_ds = data['test'].get('per_dataset', {})
for ds_name, m in per_ds.items():
    ds_short = ds_name.replace('IEEE_ADHD', 'IEEE').replace('OpenNeuro_ds003478', 'OpenNeuro')
    auroc = m.get('auroc', None)
    auroc_str = f'{auroc:.4f}' if auroc is not None else '?'
    f1 = m.get('f1_weighted', '?')
    f1_str = f'{f1:.4f}' if isinstance(f1, float) else '?'
    subj = m.get('subject_acc', '?')
    subj_str = f'{subj:.4f}' if isinstance(subj, float) else '?'
    print(f'{ds_short},$DIAG,$HEAD_TYPE,{m[\"balanced_acc\"]:.4f},{auroc_str},{f1_str},{subj_str}')
" | while IFS= read -r NEW_ROW; do
        KEY="${NEW_ROW%%,*},"$(echo "$NEW_ROW" | cut -d',' -f2)","$(echo "$NEW_ROW" | cut -d',' -f3)
        awk -v key="$KEY" -v row="$NEW_ROW" -v hdr="$HEADER" '
        BEGIN { FS=OFS=","; found=0 }
        NR==1 { print hdr; next }
        $1","$2","$3 == key { print row; found=1; next }
        { print }
        END { if (!found) print row }
        ' "$FIXED_RESULT" > "${FIXED_RESULT}.tmp" && mv "${FIXED_RESULT}.tmp" "$FIXED_RESULT"
    done

    ACC=$(python -c "import json; print(json.load(open('$JSON'))['test']['balanced_acc'])" 2>/dev/null || echo "?")
    echo "  >> $NAME (overall): BalAcc=$ACC"
}

# ================================================================
# 抑郁症: MODMA(3ch) + OpenNeuro + TDBRAIN → HC vs MDD
# ================================================================
if should_run "depression"; then
    # 不设 MAX_CH，但降低 batch_size 防止 128ch 数据 OOM
    run_one "MODMA,OpenNeuro,TDBRAIN" "depression" "exp_multi_depression" 64
fi

# ================================================================
# ADHD: IEEE_ADHD + Mendeley + TDBRAIN → HC vs ADHD
# ================================================================
if should_run "adhd"; then
    MAX_CH="" run_one "IEEE_ADHD,Mendeley,TDBRAIN" "adhd" "exp_multi_adhd" 256
fi

echo ""
echo "=========================================="
echo " 全部完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
column -t -s, "$FIXED_RESULT"
