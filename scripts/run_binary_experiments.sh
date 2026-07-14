#!/bin/bash
# ================================================================
# EEG-FM 二分类批量实验 (抑郁症 + ADHD 分别建模)
#
# 用法:
#   bash scripts/run_binary_experiments.sh                            # 全部 6 组, 默认 linear 头
#   bash scripts/run_binary_experiments.sh --head mlp                 # 全部用 MLP 头
#   bash scripts/run_binary_experiments.sh --dataset TDBRAIN          # 只跑 TDBRAIN (depression + adhd)
#   bash scripts/run_binary_experiments.sh --diagnosis depression     # 只跑抑郁症 (MODMA + OpenNeuro + TDBRAIN)
#   bash scripts/run_binary_experiments.sh -d MODMA -g depression     # 只跑 MODMA depression
#   bash scripts/run_binary_experiments.sh -d MODMA -g depression --head cnn1d  # MODMA + CNN1D 头
# ================================================================

set -e
cd "$(dirname "$0")/.."

# ================================================================
# 参数解析
# ================================================================
FILTER_DATASET=""
FILTER_DIAGNOSIS=""
HEAD_TYPE="linear"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --diagnosis|-g)
            FILTER_DIAGNOSIS="$2"
            shift 2
            ;;
        --head_type|--head)
            HEAD_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: bash scripts/run_binary_experiments.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset, -d    数据集名 (MODMA, OpenNeuro, TDBRAIN, IEEE_ADHD, Mendeley)"
            echo "  --diagnosis, -g  诊断类型 (depression, adhd)"
            echo "  --head, --head_type  分类头类型: linear (默认), mlp, cnn1d, attention"
            echo "  -h, --help       显示帮助"
            echo ""
            echo "示例:"
            echo "  bash scripts/run_binary_experiments.sh                           # 全部, 默认linear头"
            echo "  bash scripts/run_binary_experiments.sh --head mlp                # 全部用MLP头"
            echo "  bash scripts/run_binary_experiments.sh -d TDBRAIN                # 单个数据集"
            echo "  bash scripts/run_binary_experiments.sh -g depression --head cnn1d"
            echo "  bash scripts/run_binary_experiments.sh -d MODMA -g depression    # 精确指定"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FIXED_RESULT="outputs/results/binary_summary.csv"

echo "=========================================="
echo " 二分类批量实验"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 结果: $FIXED_RESULT"
echo " 分类头: $HEAD_TYPE"
[[ -n "$FILTER_DATASET" ]]   && echo " 筛选数据集: $FILTER_DATASET"
[[ -n "$FILTER_DIAGNOSIS" ]] && echo " 筛选诊断:   $FILTER_DIAGNOSIS"
echo "=========================================="

# 如果文件不存在，写入表头
HEADER="dataset,diagnosis,head_type,balanced_acc,auroc,f1_weighted,subject_acc"
if [[ ! -f "$FIXED_RESULT" ]]; then
    echo "$HEADER" > "$FIXED_RESULT"
fi

run_one() {
    local DS=$1
    local DIAG=$2
    local BS=${3:-256}
    local NAME="$4"

    # 名称包含 head_type 后缀（linear 不添加以保持兼容）
    local SUFFIX=""
    if [[ "$HEAD_TYPE" != "linear" ]]; then
        SUFFIX="_${HEAD_TYPE}"
        NAME="${NAME}${SUFFIX}"
    fi

    echo ""
    echo "===== [$NAME] (head=$HEAD_TYPE) ====="

    rm -rf "outputs/checkpoints/$NAME"

    python models/train.py \
        --dataset "$DS" \
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
        --datasets "$DS" \
        --output_dir "outputs/results/$NAME" \
        ${MAX_CH:+"--max_channels" "$MAX_CH"}

    # 提取指标 (test split)
    JSON="outputs/results/$NAME/evaluation.json"
    # 数据集名缩写（兼容旧格式）
    DS_SHORT=$(echo "$DS" | sed 's/IEEE_ADHD/IEEE/; s/OpenNeuro_ds003478/OpenNeuro/')
    ACC=$(python -c "import json; print(json.load(open('$JSON'))['test']['balanced_acc'])" 2>/dev/null || echo "?")
    AUC=$(python -c "import json; print(json.load(open('$JSON'))['test'].get('auroc','?'))" 2>/dev/null || echo "?")
    F1=$(python -c "import json; print(json.load(open('$JSON'))['test']['f1_weighted'])" 2>/dev/null || echo "?")
    SUBJ=$(python -c "import json; print(json.load(open('$JSON'))['test']['subject_balanced_acc'])" 2>/dev/null || echo "?")

    # 写入 CSV（去重：同一 dataset+diagnosis+head_type 只保留最新）
    NEW_ROW="$DS_SHORT,$DIAG,$HEAD_TYPE,$ACC,$AUC,$F1,$SUBJ"
    KEY="$DS_SHORT,$DIAG,$HEAD_TYPE"

    # 用 awk 原地 upsert：匹配 key 则替换，否则追加
    awk -v key="$KEY" -v row="$NEW_ROW" -v hdr="$HEADER" '
    BEGIN { FS=OFS=","; found=0 }
    NR==1 { print hdr; next }
    $1","$2","$3 == key { print row; found=1; next }
    { print }
    END { if (!found) print row }
    ' "$FIXED_RESULT" > "${FIXED_RESULT}.tmp" && mv "${FIXED_RESULT}.tmp" "$FIXED_RESULT"

    echo "  >> $DS_SHORT ($DIAG) [$HEAD_TYPE]: BalAcc=$ACC  AUROC=$AUC  F1=$F1  SubjAcc=$SUBJ"
}

# ================================================================
# 实验定义
# ================================================================

# 检查是否需要跑某个实验（大小写不敏感，支持子串匹配）
should_run() {
    local ds=$1
    local diag=$2
    if [[ -n "$FILTER_DATASET" ]]; then
        # 大小写不敏感子串匹配：IEEE 可匹配 IEEE_ADHD
        local ds_lower="${ds,,}"
        local filter_lower="${FILTER_DATASET,,}"
        if [[ "$ds_lower" != *"$filter_lower"* ]]; then
            return 1
        fi
    fi
    if [[ -n "$FILTER_DIAGNOSIS" ]]; then
        local diag_lower="${diag,,}"
        local filterd_lower="${FILTER_DIAGNOSIS,,}"
        if [[ "$diag_lower" != *"$filterd_lower"* ]]; then
            return 1
        fi
    fi
    return 0
}

# ================================================================
# 抑郁症实验: HC vs MDD
# ================================================================

DEPRESSION_COUNT=0
if should_run "MODMA" "depression";      then DEPRESSION_COUNT=$((DEPRESSION_COUNT+1)); fi
if should_run "OpenNeuro" "depression";  then DEPRESSION_COUNT=$((DEPRESSION_COUNT+1)); fi
if should_run "TDBRAIN" "depression";    then DEPRESSION_COUNT=$((DEPRESSION_COUNT+1)); fi

if [[ $DEPRESSION_COUNT -gt 0 ]]; then
    echo ""
    echo "========== 抑郁症 (HC vs MDD) =========="

    # 1. MODMA — HC vs MDD (只取3通道, 快)
    if should_run "MODMA" "depression"; then
        MAX_CH=3 run_one "MODMA" "depression" 64 "exp_MODMA_depression"
    fi

    # 2. OpenNeuro — HC vs MDD
    if should_run "OpenNeuro" "depression"; then
        MAX_CH="" run_one "OpenNeuro" "depression" 256 "exp_OpenNeuro_depression"
    fi

    # 3. TDBRAIN — HC vs MDD
    if should_run "TDBRAIN" "depression"; then
        run_one "TDBRAIN" "depression" 256 "exp_TDBRAIN_depression"
    fi
fi

# ================================================================
# ADHD 实验: HC vs ADHD
# ================================================================

ADHD_COUNT=0
if should_run "IEEE_ADHD" "adhd";  then ADHD_COUNT=$((ADHD_COUNT+1)); fi
if should_run "Mendeley" "adhd";   then ADHD_COUNT=$((ADHD_COUNT+1)); fi
if should_run "TDBRAIN" "adhd";    then ADHD_COUNT=$((ADHD_COUNT+1)); fi

if [[ $ADHD_COUNT -gt 0 ]]; then
    echo ""
    echo "========== ADHD (HC vs ADHD) =========="

    # 4. IEEE — HC vs ADHD
    if should_run "IEEE_ADHD" "adhd"; then
        run_one "IEEE_ADHD" "adhd" 256 "exp_IEEE_adhd"
    fi

    # 5. Mendeley — HC vs ADHD
    if should_run "Mendeley" "adhd"; then
        run_one "Mendeley" "adhd" 256 "exp_Mendeley_adhd"
    fi

    # 6. TDBRAIN — HC vs ADHD
    if should_run "TDBRAIN" "adhd"; then
        run_one "TDBRAIN" "adhd" 256 "exp_TDBRAIN_adhd"
    fi
fi

echo ""
echo "=========================================="
echo " 全部完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
column -t -s, "$FIXED_RESULT"
