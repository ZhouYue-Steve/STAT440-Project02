# Project 2：Kilns 砖窑卫星图像分类

任务：根据 256×256 的卫星图判断砖窑类型（1 = zigzag，0 = 固定烟囱窑 fck），按测试集 AUC 评分（分数 = 2×(AUC−0.5)）。
训练集 1617 张（zigzag 占 64.4%），测试集 724 张。截止时间：12 月 9 日 17:00。

**比赛规则提醒**：比赛数据不能上传给 AI。所有脚本都在本机运行，只把汇总统计打印到终端。

---

## 1. 环境（阶段 0）

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

- 本机：Apple M1、8 GB 内存、macOS 14.6。
- **torch 固定为 2.6.0 / torchvision 0.21.0。** 新版 torch 2.15 在 macOS 14 上一用 MPS 就会崩溃（调用了 macOS 15 才有的 API），所以必须用旧版本。
- MPS 训练速度（合成数据，每折约 1294 张训练图）：

| 模型 | 分辨率 | batch | 每 epoch | 5 折 × 15 epoch |
|---|---|---|---|---|
| ResNet50 | 224 | 32 | ~50 秒 | ~1 小时 |
| EfficientNet-B0 | 256 | 32 | ~34 秒 | ~45 分钟 |
| ConvNeXt-Tiny | 224 | 16 | **显存不足** | — |

  ConvNeXt-Tiny 在 batch 16 时也会超出显存（MPS 上限约 9 GB，其中约 5.8 GB 被系统和其他程序占用）。如果要用，需要关掉其他程序，并改用 batch 8 + 梯度检查点，或者换到 Colab。

### 1.1 Windows + NVIDIA GPU（阶段 2 后半段起）

阶段 2 的后半段和阶段 3 改在 Windows 11 + RTX 4060（8 GB）上运行。同样固定 torch 2.6.0，但要装 CUDA 版（先装其余依赖，再强制换成 cu124 的 torch，否则 timm 会带进 CPU 版）：

```bash
uv venv --python 3.11 .venv
grep -v -E '^(torch|torchvision)==' requirements.txt > req_notorch.txt
uv pip install --python .venv/Scripts/python.exe -r req_notorch.txt
uv pip install --python .venv/Scripts/python.exe --reinstall-package torch --reinstall-package torchvision     torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124     --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

Windows 上把下文的 `.venv/bin/python` 换成 `.venv/Scripts/python.exe`（`scripts/*.sh` 会自动选择）。
在 M1 上验证过的结果在 CUDA 上能复现（例如 DINOv2-S@336 线性探针 OOF AUC 0.932，选到的 C 完全相同）。
CUDA 速度：lecture CNN + D4 增强 5 折约 19 分钟（M1 上 11 小时以上）；DINOv2-S@336 微调 5 折约 10 分钟。

## 2. 目录结构

```
project02/
├── data/            原始数据（不改动；解压后的图片和 train/data.csv 已纳入仓库，zip 不纳入）
├── src/common.py    路径、读图、读 folds、提交文件的生成与校验
├── scripts/         按编号顺序运行
├── src/dl.py        深度学习公用函数（D4 变换、读入全部图片、OOF/测试预测的保存、实验记录）
├── out/             中间结果（folds、特征、检查结果）
│   ├── features/    冻结特征（每张图 × 8 种 D4 变换）
│   ├── oof/         每个模型的 out-of-fold 预测（训练集）
│   ├── test_preds/  每个模型的测试集预测（5 个折模型的平均）
│   ├── logs/        训练日志
│   └── experiments.csv  所有实验的汇总（13_summary.py 打印）
└── submissions/     提交文件
```

## 3. 运行顺序（阶段 1）

每个脚本都在项目根目录下运行：`.venv/bin/python scripts/<脚本名>`。

| # | 脚本 | 作用 | 输出 | 用时 |
|---|---|---|---|---|
| 0 | `00_test_submission.py` | 测试提交文件生成与校验（应输出 `ALL SUBMISSION TESTS PASSED`） | `submissions/constant_0.5.csv` | 几秒 |
| 1 | `01_near_duplicates.py` | 完全相同、旋转/翻转后近重复的图片（pHash + 缩略图相关系数，取 8 种变换中的最小距离） | `out/near_dup_pairs.csv`, `out/nn_distance.csv` | ~15 秒 |
| 1b | `01b_shifted_overlap.py` | 平移后部分重叠的切片：对所有 270 万个图片对做 64px 相位相关初筛 | `out/overlap_pairs.csv`, `out/overlap_nn.csv` | ~1.5 分钟 |
| 1c | `01c_verify_overlap.py` | 全分辨率验证初筛出的候选：重叠区像素相关系数，与随机配对对照，并用人工构造的重叠切片校准 | `out/overlap_verified.csv` | ~5 秒 |
| 2 | `02_folds.py` | 固定的分层 5 折（seed 440）；如果已有不同的划分会拒绝覆盖 | **`out/folds.csv`** | 几秒 |
| 3 | `03_shortcut_check.py` | 简单图像特征的捷径检查、简单模型的 AUC 下限、训练/测试对抗验证 | `out/simple_features.csv` | ~25 秒 |

**提交文件一律通过 `common.write_submission(scores, name)` 写出**，它会自动校验格式：表头 `index,score`、724 行、按 index 升序、分数在 [0,1] 内。

## 4. 阶段 1 的结论

### 4.1 没有重复或重叠的图片 → 用普通分层 5 折
- 完全相同的文件：0 张。
- 旋转/翻转后的近重复：最近邻的 pHash 距离最小为 9（64 位哈希），而且整体是连续分布，没有“很近”的一簇。缩略图相关系数最高只有 0.876。在 pHash ≤ 12 的 49 对训练图里，标签一致率 0.61，和随机配对的 0.54 接近。
- 平移重叠：校准显示，在全分辨率下，≥50% 的真实重叠都能完整找回（相关系数 1.0）。64px 初筛能可靠地捕捉 ≥70% 的重叠。314 对候选配对的重叠区相关系数最高 0.57，和随机配对的最大值 0.62 没有区别。
- **局限**：重叠不到约 50% 的切片在初筛时可能漏掉；不同日期重拍、光照不同的同一地点，像素相关系数也会偏低。

### 4.2 5 折划分
每折 323–324 张，zigzag 比例都是 0.64，和整体一致。

### 4.3 没有发现明显的捷径
- 单个简单特征的 AUC 都在 0.48–0.55 之间。
- 简单特征 + 逻辑回归 / 梯度提升，CV AUC 为 **0.61–0.63**。这是 CNN 至少要明显超过的下限。
- 只用“图像属性”特征（文件大小、清晰度、过暗/过亮像素）：逻辑回归 0.50，梯度提升 0.575。信号很弱，而且可能只是反映了窑周围的地物。
- 图片编号和标签无关：相邻编号的标签一致率为 0.552，置换检验 p = 0.18；按编号分成 5 段，每段的 zigzag 比例在 0.63–0.68 之间。

### 4.4 训练集和测试集的分布一致
- 对抗验证（去掉编号后）：逻辑回归 AUC 0.505，梯度提升 0.548，基本无法区分训练图和测试图。
- 加上编号时 AUC = 1.0，这只是因为测试集编号都排在训练集之后，没有意义。

## 5. 阶段 2–3 的脚本

| # | 脚本 | 作用 | 输出 |
|---|---|---|---|
| 10 | `10_lecture_cnn.py --aug none\|d4` | 第 14 讲的小 CNN（PyTorch 重写），5 折；`d4` = 每个 epoch 每张图的 8 种旋转/翻转都训练一遍 | `out/oof/lecture_cnn_*.csv` 等 |
| 11 | `11_extract_features.py --model M --res R` | 预训练网络（timm）的冻结特征，每张图 × 8 种 D4 变换 | `out/features/<M>_<R>.npz` |
| 12 | `12_linear_probe.py --feat F[,F2,...] --mode t0\|mean8\|aug8 [--clf svm]` | 冻结特征上的逻辑回归（C 由内层 CV 选）；多个特征用逗号拼接；`--clf svm` 换成 RBF SVM | `out/oof/probe_*.csv` 等 |
| 13 | `13_summary.py` | 按 OOF AUC 打印 `out/experiments.csv` | — |
| 14 | `14_ensemble.py` | OOF 预测的集成（top-k 排名平均、贪心选择、logistic stacking），**嵌套 CV** 打分，写出最佳者的提交文件 | `submissions/ens_<方法>.csv`, `out/ensemble_weights.csv` |
| 15 | `15_finetune.py --model M --res R [--lr --llrd --epochs]` | DINOv2 端到端微调，5 折；最后一个 epoch 的模型 + D4 TTA | `out/oof/ft_*.csv`, `out/ft_*_curve.csv` |
| sh | `run_stage2_ablation.sh` / `run_stage3_heads.sh` / `run_stage3_finetune.sh` | 按顺序跑上面的实验 | `out/logs/*.log` |

所有 AUC 都是 `out/folds.csv` 上的 **pooled OOF AUC**：每张训练图的分数来自没见过它的折模型；超参数只在训练折内部选（内层 CV）或事先固定，外层验证折从不参与调参。

## 6. 阶段 2 的结论：冻结特征 + 线性探针

| 模型（mean8 = 8 种 D4 特征取平均） | 224 | 336 | 384/448 |
|---|---|---|---|
| ResNet50 (ImageNet) | 0.870 | — | — |
| EfficientNet-B0 | 0.850 | — | — |
| ConvNeXt-Tiny (in22k) | 0.875 | — | 0.880 (384) |
| DINOv2 ViT-S/14 | 0.890 | 0.932 | 0.927 (448) |
| DINOv2 ViT-B/14 | 0.922 | 0.950 | 0.952 (448) |
| DINOv2 ViT-L/14 | 0.942 | **0.958** | — |

- **DINOv2（自监督）明显好于 ImageNet 有监督的 CNN**，模型越大越好（S < B < L）。
- **分辨率从 224 提到 336 提升很大**（+0.03～0.04），再到 448 基本没有提升。256 的原图放大后，14px 的 patch 能覆盖更细的结构。
- D4 平均（TTA）稳定地提升约 0.03（t0 → mean8）；把 8 种变换当作增强样本训练（aug8）只再多 +0.002。
- 特征拼接（S336 + B224）0.945，好于各自单独使用；RBF SVM 比逻辑回归差（0.90–0.92）。
- 第 14 讲的 CNN：不增强 0.616，D4 增强 0.735。从零训练的小网络远不如预训练特征。

## 7. 阶段 3 的结论：端到端微调与集成

### 7.1 DINOv2 微调（`15_finetune.py`）
设置：336px，随机 D4 + ±10% 亮度/对比度抖动，AdamW（wd 0.05），layer-wise LR decay 0.75，1 epoch 预热 + 余弦衰减，stochastic depth 0.1，bf16，batch 16，10 epoch；用最后一个 epoch 的模型（不在验证折上 early stop），预测时做 8 种 D4 TTA。

| 模型 | 学习率（head） | OOF AUC | 每折 | 用时 (RTX 4060) |
|---|---|---|---|---|
| DINOv2 ViT-S/14 @336 | 1e-4 | 0.9952 | 0.999 / 0.998 / 0.996 / 0.996 / 0.993 | 10.5 分钟 |
| DINOv2 ViT-B/14 @336 | 1e-4 | 0.9771 | 1.000 / **0.623** / 0.999 / 1.000 / 0.995 | 29.5 分钟 |
| **DINOv2 ViT-B/14 @336** | **5e-5** | **0.9984** | 1.000 / 0.999 / 0.998 / 1.000 / 0.995 | 30.1 分钟 |

- **微调比最好的冻结特征（0.958）高出约 0.04**，分数从 91.6 提到 99.7（比赛分数 = 200×(AUC−0.5)）。
- ViT-B 在 lr 1e-4 下第 1 折**没有训起来**：训练 loss 一直停在 0.65 左右（约等于 64/36 类别先验的熵），只学会了输出基础比例。这只看训练 loss 就能发现，与验证集无关。因此把 ViT-B 的学习率统一降到 5e-5，**5 折全部重跑**（不是只重跑失败的那一折）。失败的那次记为 `*_lr1e-4`，保留在日志里，但不参与集成。
- 学习曲线（`out/ft_*_curve.csv`）：每折在第 7–8 个 epoch 进入平台期，之后不下降，所以用最后一个 epoch 不吃亏；D4 TTA 再加约 +0.003。

### 7.2 集成（`14_ensemble.py`，嵌套 CV）
用同一份 OOF 挑模型再打分会偏乐观，所以每种方法都在 4 折的 OOF 上选模型/拟合权重，在第 5 折上打分。

| 方法 | 嵌套 AUC | 折间 SD | 同一份 OOF 上（偏乐观） |
|---|---|---|---|
| logistic stacking（29 个模型的 logit） | **0.9984** | **0.0008** | 0.9990 |
| 贪心选择（结果只选了 ViT-B 微调） | 0.9984 | 0.0017 | 0.9984 |
| top-2 排名平均（两个微调模型） | 0.9978 | 0.0018 | 0.9978 |
| top-3 … top-6 | 0.9964 → 0.9910 | | |

- 单个最好的模型（ViT-B 微调）已经 0.9984，集成并没有在 OOF 上带来可测出的提升，差距（≤0.0006）远小于噪声：测试集公开/私有各约 362 张，AUC 的抽样误差约 ±0.002。
- stacking 的权重主要给 ViT-B 微调（0.43），其次是 DINOv2-L@336 探针（0.13），其余是一些较弱、但和它们相关性较低的模型。

### 7.3 提交文件
| 文件 | 内容 | 嵌套/OOF AUC |
|---|---|---|
| **`submissions/ens_stack.csv`** | **首选**：logistic stacking，嵌套 AUC 最高、折间最稳定 | 0.9984 |
| `submissions/ft_dinov2_b336.csv` | 备选：只用 ViT-B 微调（5 个折模型平均） | 0.9984 |

两个文件都通过了 `validate_submission`；测试集上 score > 0.5 的比例都是 64%，和训练集的 zigzag 比例一致。两者在 OOF 上打平，可以都提交，用公开榜看一下，但公开榜只有一半测试图，不要据此过度挑选。

### 7.4 局限
- OOF AUC 已接近 1，剩下的错误很少，几十张图的差别就决定排名。所以不同方法之间 0.001 级的差别不可靠。
- 微调只用了一个随机种子；多个种子取平均可能让测试预测更稳，但 OOF 上已经很难再看出提升。
