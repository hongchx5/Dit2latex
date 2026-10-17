# Patch 预算缩放与 batch 组包规范（v1）

本文档自包含，可直接作为实现依据。**按顺序实现，不要改写规则**；编号 `R1..R7` 是规则，`MUST` / `MUST NOT` 是硬约束。

---

## 0. 参数

| 参数           | 默认值    | 含义                                |
| ------------ | ------ | --------------------------------- |
| `max_patch`  | `192`  | 单张图的 patch 总数上限                   |
| `patch_size` | `16`   | 多少像素算一个 patch（16×16 像素 = 1 patch） |
| `min_height` | `64`   | 缩放后**高度**低于此值的图丢弃                 |
| `multiple`   | `8`    | 尺寸对齐粒度（VAE 下采样 8 倍）               |
| `batch_size` | `4`    | 每个 batch 的图片数                     |
| `drop_last`  | `true` | 桶内尾部不足一个 batch 时是否丢弃              |
| `seed`       | `42`   | 洗牌种子                              |

关系式：`patch 数 ≤ max_patch` 等价于每张图像素数约 `≤ max_patch × patch_size²`（192×256 = 49,152）。**这是缩放预算的物理含义，不是硬约束**——约束一律按 patch 数判定。

---

## R1. patch 计数

```
count_patches(H, W) = (H // patch_size) * (W // patch_size)
```

- `//` MUST 是**整除**（向下取整）。MUST NOT 用浮点除法或四舍五入。
- 例：`count_patches(104, 416, 16) = 6 * 26 = 156`。

---

## R2. 等比缩放（核心）

对每张原图 `(H, W)`，求满足以下三条的**最大**尺寸 `(H', W')`：

1. `H' % multiple == 0` 且 `W' % multiple == 0`；
2. `W' / H' ≈ W / H`（保持长宽比，按 `H'` 等比推 `W'` 后对齐）；
3. `count_patches(H', W') ≤ max_patch`。

**可以放大，也可以缩小**（原图 patch 数不足时放大到接近预算）。

### 算法：二分

`count_patches` 对尺寸**单调不减**，因此在 `H'` 上二分：

```python
import math

def count_patches(H, W, patch_size=16):
    return (H // patch_size) * (W // patch_size)


def fit_size_to_patch_budget(H, W, max_patch=192, patch_size=16, multiple=8):
    if max_patch <= 0 or H <= 0 or W <= 0:
        return H, W

    def size_at(hp):                      # 给定 H'，等比算对齐后的 W'
        wp = int(round(W * hp / float(H)))          # 四舍五入到最近整数
        wp = max(multiple, int(round(wp / float(multiple))) * multiple)  # 对齐到 multiple
        return hp, wp

    upper = int(math.sqrt(max_patch * patch_size * patch_size * H / float(W))) + multiple * 2
    lo, hi, best = multiple, max(multiple, (upper // multiple) * multiple), None
    while lo <= hi:
        mid = ((lo + hi) // 2 // multiple) * multiple
        if mid < multiple:
            mid = multiple
        hp, wp = size_at(mid)
        if count_patches(hp, wp, patch_size) <= max_patch:
            best = (hp, wp)
            lo = mid + multiple
        else:
            hi = mid - multiple
    return best if best is not None else size_at(multiple)
```

要点（MUST）：

- 返回的是**满足约束的最大**尺寸，不是第一个可行解。
- `W'` 由 `H'` 等比推导后对齐，MUST NOT 独立缩放两轴（会破坏长宽比）。
- `round` 用语言默认的"四舍五入"，与 `multiple` 对齐时误差 ≤ 4px，可接受。

> **长宽比是"近似保持"，不是精确保持。** `W'` 要对齐到 8 的倍数，因此与理想值最多差 8px。  
> 缩放后尺寸越小，这点误差的**相对占比**越大：`2929×485`（6.04:1）会得到 `552×88`（6.27:1），偏差 3.7%。  
> 这是对齐的必然结果，**不是缺陷**；不要为"比例必须严格不变"加断言，也不要因此放宽 `multiple`。

---

## R3. 丢弃

```
若 H' < min_height：丢弃该图
```

- 检查的是**缩放后的高度 `H'`**，MUST NOT 用短边、原始高度或宽度。
- 丢弃发生在**组 batch 之前**，被丢弃的图不参与分桶、不计入 epoch 长度。
- 丢弃数量 MUST 被统计并输出日志（含原始尺寸 → 能缩到多少）。

> 边界提示：该规则只约束高度。若数据含**竖条图**（H > W），缩放后宽度可能小于 64（如 3000×200 → 872×56）。这不违反规则，但下游若使用 LPIPS 这类需要短边 ≥ 64 的算子，需自行处理。

---

## R4. 分桶（按缩放后尺寸）

以 `(H', W')` **精确**分组（不可用区间合并、不可排序后切片）：

1. 样本数 `≥ batch_size` 的尺寸 → 独立成桶；
2. 样本数 `< batch_size` 的尺寸 → 并入"形状最接近"的**已成立桶**，距离定义为：
   ```
   distance((h1,w1), (h2,w2)) = |ln(h1/h2)| + |ln(w1/w2)|
   ```
   并入后这些样本在执行期会被缩放到目标桶的尺寸。
3. 若**没有任何尺寸**达到 `batch_size`（尺寸种类过多或数据太少）→ 整体退回 `batch_size = 1`。

MUST：每个 batch 只来自**一个桶**，因此 batch 内尺寸严格一致。

---

## R5. 组 batch

每个 epoch 重新生成 batch 列表：

```python
rng = Random(seed + epoch)
batches = []
for size, indices in buckets.items():        # 遍历顺序无关紧要
    order = list(indices)
    rng.shuffle(order)                        # 1) 桶内打乱
    for i in range(0, len(order), batch_size):
        chunk = order[i:i+batch_size]
        if len(chunk) < batch_size and drop_last:
            continue                          # 2) 尾部不足则丢
        batches.append(chunk)
rng.shuffle(batches)                          # 3) 所有 batch 全局打乱
```

- 打乱发生在 **batch 粒度**，不是样本粒度（样本级打乱与"batch 内同尺寸"互斥）。
- 每个 epoch MUST 改变 `rng` 种子，否则各 epoch 的 batch 组合完全相同。
- `drop_last` 对**每个桶独立**生效，每 epoch 每桶最多丢 `batch_size - 1` 张。

---

## R6. 执行期缩放

- 读取图片后按 `(H', W')` 做一次 `resize`：**放大用 BICUBIC，缩小用 BILINEAR**。
- 尺寸与目标不符时（如原图读取失败换图）MUST 强制插值到该 index 的 `(H', W')`，保证 batch 内不混尺寸。
- 归一化：`ToTensor` 后 `x * 2 - 1`，值域 `[-1, 1]`。

---

## R7. 训练与推理一致性

推理端 MUST 复用同一个 `fit_size_to_patch_budget`，且 `max_patch` / `patch_size` / `min_height` 与训练侧一致，否则 PSNR/SSIM 不对应训练时见到的分布。

---

## 禁止事项（MUST NOT）

1. 用浮点除法算 patch 数，或对 `H/patch_size` 四舍五入。
2. 用短边代替 `H'` 做 `min_height` 判定。
3. 破坏长宽比（两轴独立缩放）。
4. 跳过 `multiple` 对齐（VAE 会因尺寸非 8 倍数报错）。
5. 把不同尺寸的图 stack 进同一个 batch。
6. 先切 batch 再缩放（顺序反了会导致尺寸不一致）。
7. 用采样级打乱替代 batch 级打乱（两者互斥，且必须保留"batch 内同尺寸"）。
8. 丢弃发生在分桶之后（会造成桶内样本数与 epoch 长度不一致）。

---
