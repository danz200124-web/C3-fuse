
# C³‑Fuse+DL 项目实施 **超详细计划**（每一步含原因与原理）
> 目标：将图像与点云在复杂工程现场（隧道掌子面/边坡/露头）实现**鲁棒融合**，自动输出结构面/节理参数与工程报表。
> 范式：C³‑Fuse（跨注意力 + 圆柱BEV + 自适应门控） + 2D–3D 协同训练（对比学习 + 投影一致性）。

---

## 总览（四阶段、十五步骤）
- **P0 准备**：环境/仓库/规范（Step 0）  
- **P1 数据与几何**：采集与标定（Step 1–3），配准与可见性建模（Step 4–5）  
- **P2 学习与融合**：数据集构建与预训练（Step 6–7），核心融合网络实现与训练（Step 8–10）  
- **P3 工程闭环**：推理与工程统计（Step 11），评测与消融（Step 12），打包与验收（Step 13–15）

每一步均包含：【做什么】【为什么（原因）】【原理要点】【执行清单】【质控/验收】。

---

## Step 0. 工程化准备（环境、目录、版本与质量基线）
**做什么**：搭建 Conda/PyTorch 环境、代码仓库与标准目录；定义 Makefile 与配置模板。  
**为什么**：统一环境与路径，降低复现实验的摩擦；便于集成 CI/日志/可视化。  
**原理要点**：可复现性依赖“同一版本 + 同一随机种子 + 同一数据切分”。

**执行清单**
- Conda + PyTorch（>=2.1）、Open3D、OpenCV、NumPy、MinkowskiEngine（或 spconv）。
- 目录：`data/{raw,processed,annotations,splits,reports}`, `configs/`, `scripts/`, `models/`, `tools/`。
- 文件：`configs/env.yaml`（路径）、`configs/c3fuse_base.yaml`（模型）、`Makefile`（常用流程）。

**质控/验收**
- `python -c "import torch; print(torch.cuda.is_available())"` 为 True；
- `make preprocess SCENE=demo` 正常运行且产生输出结构。

---

## Step 1. 数据采集规范（多视图图像 + 近景点云）
**做什么**：制定相机/激光雷达或 SfM/TLS 的**采集脚本与命名规范**，并记录时间戳。  
**为什么**：时间同步与重叠视角是后续配准、可见性与投影采样的前提。  
**原理要点**：几何三角测量对**基线/视差**敏感；多视角可提升可见性与遮挡恢复。

**执行清单**
- 同一“窗口”至少 6–8 幅图像（70% 视域重叠），点云包含强度/颜色更佳；
- 命名：`scene_XXX_cameraID_YYYYmmdd_HHMMSSfff.jpg / .pcd`；
- 记录外业表（设备、姿态、距离、光照）。

**质控/验收**
- 抽检相邻视图的特征匹配数量 > 阈值（如 1000）；
- 点云密度在目标距离范围内达到预设（如 5–10 pts/cm²）。

---

## Step 2. 相机内参标定与去畸变
**做什么**：Zhang 法标定 `K` 与畸变系数，批量去畸变。  
**为什么**：准确的内参减小重投影误差，提升投影采样的定位精度。  
**原理要点**：针孔模型 + 径向/切向畸变；最小化角点重投影误差。

**执行清单**
- `scripts/01_calib_extract.py --board 9x6 --square 0.025` 生成 `cam_intrinsics.yaml`；
- `scripts/02_preprocess.py --tasks undistort` 写回处理后的图像。

**质控/验收**
- 标定 RMS < 0.8 像素；
- 去畸变后直线边缘残差明显降低（脚本输出前后对比）。

---

## Step 3. 时间同步与多传感器外参初始化
**做什么**：时间序列对齐（硬/软同步），外参 `T_{c←L}` 初始化。  
**为什么**：跨模态错位是融合失败的主因之一；同步可减少动态场景误差。  
**原理要点**：外参求解可视为最小化 `∑ ||Π(K, T X) - u||` 的非线性优化；时间偏置作为附加变量估计。

**执行清单**
- 记录或估计 `timestamp_offset_ms`；
- 以 DeepI2P 粗估外参（见 Step 4），将估计值写入 `extrinsics_init.json`。

**质控/验收**
- 抽取特征点重投影误差 < 3 像素；
- 视觉/点云主要边界在叠加图中大致重合。

---

## Step 4. 跨模态配准：粗对齐（DeepI2P）→ 精对齐（Colored‑ICP）
**做什么**：先用 DeepI2P 进行图像–点云粗对齐，再用 Colored‑ICP 对点云进行精配准。  
**为什么**：DeepI2P 弱化显式跨模态特征匹配；Colored‑ICP 融合颜色/光度项，细化位姿。  
**原理要点**：
- DeepI2P：将配准转化为分类 + 反投影优化，最大化正确视锥一致性；
- Colored‑ICP：最小化 `E = E_geo + λ E_color`，使几何与光度残差同时收敛。

**执行清单**
- `scripts/03_reg_deepi2p.py --weights weights/deepi2p.pth` → `extrinsics_init.json`；
- `scripts/04_reg_colored_icp.py --init extrinsics_init.json` → `extrinsics_refined.json`；
- 保存重投影可视化与误差直方图。

**质控/验收**
- ICP fitness ↑、RMSE ↓；
- 重投影中位数误差 < 2–3 像素。

---

## Step 5. 表面重建与可见性建模（Z‑buffer/网格遮挡）
**做什么**：Screened Poisson 重建连续网格，建立基于网格的可见性与遮挡判断（Z‑buffer）。  
**为什么**：可见性/遮挡是跨注意力采样与图像抬升到 BEV 的关键；网格提供稳定法向与几何正则。  
**原理要点**：泊松方程拟合有向点集；Z‑buffer 取最小深度保证正确遮挡关系。

**执行清单**
- `scripts/05_poisson_recon.py --depth 10 --linear_fit` 输出 `mesh/*.ply`；
- 生成每点可见视图 Top‑K 与遮挡掩码。

**质控/验收**
- 网格连通性良好、孔洞率低；
- 可见性掩码抽检与真实遮挡一致度 > 90%。

---

## Step 6. 数据集构建与标注（2D/3D）
**做什么**：整理训练/验证/测试划分；制作 2D 掩码与 3D 标签（可弱监督）。  
**为什么**：监督信号的“位置正确 + 可投影一致”决定模型上限；弱监督可降低人工成本。  
**原理要点**：2D→3D 投影标签与 3D→2D 反投影一致性构成互补监督。

**执行清单**
- `scripts/06_make_dataset.py --split_ratio 7,2,1`；
- 2D：像素级结构面掩码（空洞忽略）；3D：RANSAC 平面 + 人工复核或 2D 投影弱标注。

**质控/验收**
- 划分无泄漏（同场景不跨集合）；
- 标注一致性（双人交叉）κ 值 ≥ 0.75。

---

## Step 7. 预训练（提升少样本性能）
**做什么**：3D 主干进行 PointContrast 预训练；2D–3D 进行 CrossPoint 风格对比预训练。  
**为什么**：减少对大规模标注的依赖，强化跨模态对齐的共同潜空间。  
**原理要点**：InfoNCE 将正样拉近、负样推远；点云自监督增强几何特征的可迁移性。

**执行清单**
- `scripts/07_pretrain_pointcontrast.py --epochs 200` → `weights/pointcontrast.pt`；
- `scripts/08_pretrain_crosspoint.py --epochs 100` → `weights/crosspoint.pt`。

**质控/验收**
- 预训练损失稳定下降；
- 下游少量标注微调下 mIoU/F1 明显优于无预训练。

---

## Step 8. 融合网络实现（C³‑Fuse：跨注意力 + 圆柱BEV + 门控）
**做什么**：实现两条融合路径与门控融合。  
**为什么**：跨注意力提供细节级“软关联”，BEV 提供大范围上下文与高效计算；门控在模态退化时动态分配信任。  
**原理要点（含公式）**：  
- **跨注意力**（点为 Query，像素邻域为 Key/Value，加入几何偏置）：  
  \\[
  \alpha_{ij}^l=\operatorname{softmax}_{j,l}\!
  \big(\tfrac{(W_q g_i)^\top (W_k \bar f_{ij}^l)}{\sqrt d}+b(\theta_{ij},\rho_{ij},e_{ij})\big),
  \quad f^{\text{cross}}_i=\sum_{j,l}\alpha_{ij}^l W_v \bar f_{ij}^l.
  \\]
- **圆柱 BEV**（统一网格 \\((r,\theta,z)\\)）：点体素池化与图像抬升到同一网格，格上融合后回填：  
  \\[
  f^{\text{bev}}_i=\operatorname{Interp}\!\big(B^{\text{fused}},\,\mathcal{G}(X_i)\big).
  \\]
- **门控融合**：  
  \\[
  \gamma_i=\sigma(\mathrm{MLP}([c^{img}_i,c^{3d}_i,e_i])),\quad
  h_i=\gamma_i f^{\text{cross}}_i + (1-\gamma_i) f^{\text{bev}}_i.
  \\]

**执行清单**
- `models/img_backbone/*`, `models/pcd_backbone/*`, `models/fusion/{cross_attn.py,bev_cyl.py,gate.py}`；
- `tools/projection.py`（投影采样/可见性），`tools/cyl_grid.py`（网格与回填）。

**质控/验收**
- 单元测试：给定玩具数据，跨注意力/BEV 输出形状与数值范围正确；
- 梯度检查：各模块支持反向传播，显存与时间可控。

---

## Step 9. 训练目标与优化（多损失联合）
**做什么**：组合分割、跨模态对比、投影一致性、几何一致性与门控正则损失。  
**为什么**：让 2D/3D 在**共享潜空间**对齐（对比），在像素–点级保持**语义一致**（投影），在几何上**物理可解释**（平面残差），且融合**不过拟合单一模态**（门控）。  
**原理要点（损失）**：
- \\(\mathcal{L}_{seg}=\mathrm{CE}+\mathrm{Dice}\\)；
- \\(\mathcal{L}_{cm}=-\!\sum_i\log\frac{e^{\langle z^{3D}_i,z^{2D}_i\rangle/\tau}}{\sum_q e^{\langle z^{3D}_i,z^{2D}_q\rangle/\tau}}\\)；
- \\(\mathcal{L}_{proj}=\sum_j \mathrm{CE}(\Pi_j(s^{3D}),y^{2D}_j)+\sum_i \mathrm{CE}(\Pi^{-1}(y^{2D}),y^{3D})\\)；
- \\(\mathcal{L}_{plane}=\sum_k\frac{1}{|\mathcal{S}_k|}\sum_{X_i\in\mathcal{S}_k}|n_k^\top X_i+d_k|+\lambda_{\text{smooth}}\sum_{i\sim i'}\|n_i-n_{i'}\|_1\\)；
- \\(\mathcal{L}_{gate}=-\frac{1}{N}\sum_i[\gamma_i\log\gamma_i+(1-\gamma_i)\log(1-\gamma_i)]+\lambda_{\text{sparse}}\|\gamma\|_1\\)；
- 总损失：\\(\mathcal{L}=\lambda_1\mathcal{L}_{seg}+\lambda_2\mathcal{L}_{cm}+\lambda_3\mathcal{L}_{proj}+\lambda_4\mathcal{L}_{plane}+\lambda_5\mathcal{L}_{gate}\\)。

**执行清单**
- `configs/loss.yaml` 设置权重（默认：1.0, 1.0, 0.5, 0.5, 0.1）；
- `scripts/09_train_c3fuse.py` 训练（AMP + 余弦退火 + AdamW）。

**质控/验收**
- 训练曲线平滑下降；
- 验证集重投影一致性/法向残差与分割指标同步改善。

---

## Step 10. 超参与训练策略
**做什么**：确定批大小、学习率、网格分辨率、投影采样策略与视图 Top‑K。  
**为什么**：权衡性能与速度，避免显存爆炸与欠拟合。  
**原理要点**：尺度/视角多样性提升泛化；较高 BEV 分辨率提高远场/细节，但计算开销上升。

**执行清单（起步建议）**
- Batch 4–8，Epoch 100–120，AdamW(lr=1e‑4, wd=1e‑2)；
- 视图 Top‑K=3–5，投影邻域 k=3/5，多尺度 1/4–1/32；
- 圆柱网格 \\(N_r=128,N_\theta=256,N_z=64\\)。

**质控/验收**
- 资源监控：单卡 24GB 显存可完整训练；
- 训练 10 epoch 内，验证 mIoU 提升 ≥ 基线 3–5%。

---

## Step 11. 推理与工程后处理（平面拟合 + vMF 成组 + 报表）
**做什么**：阈值化分割 → RANSAC/最小二乘拟合平面 → vMF 混合在球面聚类法向；导出极点图/统计表。  
**为什么**：将学习结果转化为工程可用**参数与图表**；vMF 在球面上建模方向分布天然契合法向。  
**原理要点**：
- RANSAC 稳健估计平面；
- vMF 混合：\\(p(x)=\sum_k \pi_k C_3(\kappa_k)\exp(\kappa_k \mu_k^\top x)\\)。

**执行清单**
- `scripts/10_infer_and_post.py --post ransac_plane_fit --export_csv planes.csv`；
- `scripts/11_vmf_clustering.py --K 3,4,5 --select_by bic --plot stereonet.png`；
- `scripts/12_report_build.py --out report.pdf`。

**质控/验收**
- 平面残差（点到面 RMSE）≤ 0.05–0.10 m（按场景尺度）；
- 组均值法向与人工判读角差 ≤ 5°；报表要素齐全。

---

## Step 12. 评测指标与消融实验
**做什么**：制定指标与对照组，量化增益来源。  
**为什么**：识别“哪一块最有效”，支撑论文与工程验收。  
**原理要点**：对照实验 + 统计显著性（如成对 t 检验）。

**执行清单**
- 指标：重投影误差、mIoU/F1、点到面 RMSE、法向角差、间距/频率误差；
- 对照：仅点云、仅图像、早融合(PointPainting)、仅跨注意力、仅BEV、全模型、去掉门控/一致性/对比等。

**质控/验收**
- 全模型在召回、法向角差与 RMSE 上均优于所有对照；
- 关键模块的贡献可被数据支持。

---

## Step 13. 规模化与性能优化
**做什么**：加速可见性、体素池化与聚类；分块/多进程/GPU 并行。  
**为什么**：支撑千万级点云与多视图大场景的时效性。  
**原理要点**：稀疏张量/稀疏卷积、块处理（tiling）、KDE/DBSCAN 的 GPU 化。

**执行清单**
- 采用稀疏卷积骨干（MinkUNet）；
- 可见性与投影采样批处理；
- 聚类与统计流程 GPU/多进程化。

**质控/验收**
- 千万点场景**≤30–60 min/GPU** 完成全流程；
- 结果与小规模验证一致。

---

## Step 14. 交付物与可复现打包
**做什么**：生成一键脚本/容器、配置样例、示例数据与 README。  
**为什么**：便于交付、培训与后续维护。  
**原理要点**：不可变基础设施（容器） + 确定性随机种子。

**执行清单**
- Dockerfile/requirements.txt；
- `make all` 完成一键跑通 demo；
- 发布 `docs/模型说明(LaTeX).md`、`plan.md`、`report.pdf` 样例。

**质控/验收**
- 在另一台干净机器上 2 小时内跑通并复现主要指标。

---

## Step 15. 风险清单与兜底方案
- **外参与同步误差**：加入重估与鲁棒损失（投影一致性 + 门控）；严重时退化为仅点云管线。  
- **光照/粉尘导致图像退化**：门控降低图像权重；启用形态学或曝光均衡预处理。  
- **点云稀疏/遮挡**：提高 Top‑K 视图、扩大投影邻域、提升 BEV 分辨率；必要时启用多窗口拼接。  
- **显存与时延**：梯度累积/混合精度/稀疏算子；分块推理。

---

## 里程碑（建议排期）
- **M1（第 2 周）**：完成 Step 1–5（数据到可见性）✅  
- **M2（第 6 周）**：完成 Step 6–7（数据集与预训练）✅  
- **M3（第 12 周）**：完成 Step 8–10（模型训练、初版结果）✅  
- **M4（第 16 周）**：完成 Step 11–12（工程统计与评测/消融）✅  
- **M5（第 20 周）**：完成 Step 13–15（优化与交付）✅

---

## 命令“抄作业”清单（可直接运行）
```bash
# 预处理 + 粗/精配准 + 重建
python scripts/02_preprocess.py --scene data/raw/scene_A --tasks undistort,pcd_voxel_down,pcd_est_normal --out data/processed/scene_A
python scripts/03_reg_deepi2p.py --images data/processed/scene_A/images --pcd data/processed/scene_A/pointclouds/scene_A.pcd --intr data/raw/scene_A/meta/cam_intrinsics.yaml --out data/processed/scene_A/meta/extrinsics_init.json --weights weights/deepi2p.pth
python scripts/04_reg_colored_icp.py --src data/processed/scene_A/pointclouds/scene_A.pcd --tgt data/processed/scene_A/pointclouds/scene_A_ref.pcd --init data/processed/scene_A/meta/extrinsics_init.json --out data/processed/scene_A/meta/extrinsics_refined.json
python scripts/05_poisson_recon.py --pcd data/processed/scene_A/pointclouds/scene_A.pcd --out_mesh data/processed/scene_A/mesh/scene_A_poisson.ply

# 数据集与预训练
python scripts/06_make_dataset.py --scenes data/processed --split_ratio 7,2,1 --out data/splits
python scripts/07_pretrain_pointcontrast.py --pcd_root data/processed --epochs 200 --batch_size 8 --lr 1e-3 --save weights/pointcontrast.pt
python scripts/08_pretrain_crosspoint.py --img_root data/processed --pcd_root data/processed --pairs splits/train_pairs.txt --epochs 100 --batch_size 8 --lr 5e-4 --init3d weights/pointcontrast.pt --save weights/crosspoint.pt

# 训练与推理
python scripts/09_train_c3fuse.py --cfg configs/c3fuse_base.yaml --env configs/env.yaml --pretrain2d3d weights/crosspoint.pt --log runs/c3fuse_exp01
python scripts/10_infer_and_post.py --ckpt runs/c3fuse_exp01/checkpoints/last.ckpt --scene data/processed/scene_A --out data/processed/scene_A/pred --post ransac_plane_fit --export_csv data/reports/scene_A/planes.csv
python scripts/11_vmf_clustering.py --normals data/processed/scene_A/pred/plane_normals.npy --K 3,4,5 --select_by bic --plot --out_fig data/reports/scene_A/stereonet.png --out data/reports/scene_A/vmf.json
python scripts/12_report_build.py --scene data/processed/scene_A --csv data/reports/scene_A/planes.csv --stn data/reports/scene_A/stereonet.png --vmf data/reports/scene_A/vmf.json --out data/reports/scene_A/report.pdf
```

---

### 附录 A：配置模板（`configs/c3fuse_base.yaml` 摘要）
```yaml
img_backbone: resnet50
pcd_backbone: point_transformer
fusion:
  cross_attn: {layers: 3, heads: 4, dim: 256}
  unified_space:
    type: cylindrical_bev
    grid: {radial_bins: 128, theta_bins: 256, z_bins: 64}
loss: {w_cm: 1.0, w_seg: 1.0, w_geo: 0.5, w_soft: 0.1}
train: {batch_size: 4, epochs: 120, base_lr: 1e-4, warmup_epochs: 5, optimizer: adamw, wd: 0.01, grad_clip: 5.0}
```

### 附录 B：验收门槛（建议）
- 配准：重投影中位误差 ≤ 2.5 px，ICP RMSE 逐级下降；  
- 分割：3D mIoU ≥ 0.65（场景依赖）、F1 ≥ 0.70；  
- 几何：点到面 RMSE ≤ 0.05–0.10 m，法向角差 ≤ 5°；  
- 成组：vMF BIC 最优组数与人工一致，组均值角差 ≤ 5°；  
- 时效：单窗口（百万点）端到端 ≤ 30–60 min/GPU。

---

> 备注：若现场条件不允许高质量图像，可临时关闭跨注意力路径，仅保留 BEV 与门控（γ→倾向3D），保证最小可用；若点云稀疏/遮挡严重，则提升 Top‑K 视图与 BEV 分辨率并增强投影一致性权重。
