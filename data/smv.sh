#!/usr/bin/env bash
set -euo pipefail

# ===== 模板路径，用 {?} 作为占位符 =====
PATH1="/home/user/data3/daxger9/HMER/FiT/data/style{?}"        # 如果 path1 也要按 xx 建目录，也可以带占位符
PATH2="/home/user/data3/daxger9/HMER/pdf2img/img/0918-5k/0918-5k-style{?}"
PATH3="/home/user/data3/daxger9/HMER/pdf2img/img/0927-1w/0927-1w-style{?}"
PATH4="/home/user/data3/daxger9/HMER/pdf2img/img/crohme/crohme-style{?}"

# 预定义列表 xx=['3','4']
xx=('5' '6' '13')

for v in "${xx[@]}"; do
    # 把模板里的 {?} 替换成当前值 v
    p1="${PATH1//\{?\}/${v}}"
    p2="${PATH2//\{?\}/${v}}"
    p3="${PATH3//\{?\}/${v}}"
    p4="${PATH4//\{?\}/${v}}"

    # 在 path1 下创建 style{v} 文件夹
    # target_dir="${p1}/style${v}"
    target_dir="${p1}"
    mkdir -p "${target_dir}"

    # 把 path2/path3/path4 下的文件剪切到 target_dir
    for src_dir in "${p2}" "${p3}" "${p4}"; do
        if [[ -d "${src_dir}" ]]; then
            find "${src_dir}" -maxdepth 1 -type f -exec mv -f -t "${target_dir}" -- {} +
        else
            echo "警告: 目录不存在: ${src_dir}" >&2
        fi
    done
done