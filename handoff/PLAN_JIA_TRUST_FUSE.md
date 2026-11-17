# 甲：可信带融合实施计划

2026-11-16。只规划，不改训练、不重训、不覆盖 `ggs_pipeline` 里已有网格。通过线与 `TEAM_PLAN_2026_11_16.md` 第 5 节相同。

## 0. 要证明的事

同一套 `ggs_pipeline` 高斯、同一次中位深度渲染。训练里 NCC 误差不小于 0.9、或根本没有完成 warp 的像素，深度写成 0，不进入 TSDF。截断乘数仍是 4，体素仍是 0.002。对照是这套高斯已经提出的 trunc4 网格。

## 1. 先定死的四个选择

**近邻用最近的一个，不用训练里的随机抽样。** 训练在 `multi_view_ncc_views=1` 时，每次从 `nearest_id` 里随机抽一个近邻。提取若再随机，通过线会随种子变化。`nearest_id` 已按距离和角度排过序，提取固定用 `nearest_id[0]`。每个视角把参考图名和近邻图名写入记录。warp 公式、\(\tau=0.9\)、patch 大小 3、`pixel_noise_th=1.0` 与训练相同。

**掩码用强规则。** 像素保留深度，当且仅当它落在现有几何 `d_mask` 里，CUDA 的 `valid_mask` 为真，且 \(e=\mathrm{clamp}(1-\rho^2,0,2)<0.9\)。其余原本深度大于 0 的像素写成 0。`gt_mask<0.5` 的像素维持现有提取里已经置 0 的做法。没有算过 NCC 的像素算未通过，不保留。保留比例很低也不放宽规则，只把比例记下来，交给通过线判断。

**不重写一套 NCC。** 在现有 `PatchMatch` 上增加可选的稠密保留图返回。默认关闭时，训练前向与现在一致。提取只打开这个返回。禁止在 `mesh_extract.py` 里复制一份 warp。

**不覆盖旧结果，不改配准。** 新网格写到 `unified_geometry_gs/output_compare/ggs_trust_fuse/<scan>/`。`evaluate_dtu_mesh.py` 现在从模型目录读 `recon_post.ply`，并且要用该目录的相机做配准。给它增加显式的网格路径和输出目录，相机仍从 `ggs_pipeline/<scan>` 读取。配准算法保持 P5 那一版，这样 overall 能和旧 `vis/results.json` 相减。

## 2. 对照数字

控制组是归档的 `ggs_pipeline/<scan>/vis/results.json`，不是现场再提一次 trunc4。现场重提是乙的工作。甲开始评测前，这三个 JSON 必须在手。缺文件就先从 Release `research-assets-2026-11-15` 的最终清单里取回，不临时换对照。

高斯用 `ggs_pipeline/<scan>/point_cloud/iteration_30000`。加载参数与 P5 提取一致：`mesh_extract.py -m <pipeline> --trunc_voxel_multiplier 4 --voxel_size 0.002 --depth_max 8 --num_cluster 1`，只是深度在送进 `integrate` 之前被掩码置 0。

## 3. 步骤

**A. 资产与环境。** 确认最终 `ASSET_MANIFEST.json` 里有 scan24、105、122 的 30000 步高斯，以及三份 `vis/results.json`。工作区里没有就按清单取回，不覆盖已有证据。WSL 的 `uggs` 环境里，warp 扩展若导入失败，先按原方式重新安装，再提取。装不上就停，不改用别的相关实现。

**B. 单视角掩码，不建网格。** 只对 scan24 的第一张训练图（按 `image_name` 排序后的第一张）跑渲染和 `PatchMatch`。检查四件事：近邻是 `nearest_id[0]`；保留图尺寸与深度图一致；保留像素都满足 \(e<0.9\)；置 0 前后，正深度像素数等于保留数。把保留、NCC 拒绝、未能 warp 三类像素的比例写入 `ggs_trust_fuse/scan24/mask_preview.json`。这一步不看 Chamfer。

**C. 三场景提取。** 顺序固定为 scan24、scan105、scan122。每张训练视角都用自己的 `nearest_id[0]`。输出 `recon.ply`、`recon_post.ply`，以及每个视角的三类比例。中途不看评测结果来改 \(\tau\)。

**D. 评测。** 用改过路径参数的 `evaluate_dtu_mesh.py`，DTU 真值目录与 P5 相同。每个场景得到新的 `vis/results.json`。差值定义为新 overall 减归档 overall，负值表示新网格更好。s2d 同样相减。

**E. 图和判定。** 每个场景用步骤 B 规定的那张固定训练图：保留像素保持原渲染，被拒像素标红。图在算 Chamfer 之前就选定视角。`ggs_trust_fuse/verdict.json` 写入三场景的 overall、s2d、差值、保留比例，以及是否通过。

## 4. 通过线

三条同时成立才算通过：

- 三场景平均 overall 差值 \(\le -5\times10^{-4}\) mm。
- 三场景平均 s2d 差值 \(\le 0\)。
- scan122 的 overall 差值 \(\le 5\times10^{-4}\) mm。

任一不成立即为不过。不改阈值，不换近邻，不改成“只丢掉已经算过 NCC 且 \(e\ge 0.9\) 的像素”，不重训。局部截断是丙在通过之后才做的第二刀，甲不做。

## 5. 甲不做的事

不训练。不改 `train.py` 的损失。不处理 scan83、63、55 的展示图。不提取其余八个场景。不根据保留比例中途放宽掩码。


## 6. 审查修订：执行前必须补齐的约束（2026-11-16）

本节补充并修正前文，不启动实现。对应分工文件为用户提供的 TEAM_PLAN_2026_11_16.md。前述四项选择和三条数值通过线保持不变。

### 6.1 资产身份不能只看目录名

P5 脚本 run_research_p5.sh 的 resolve_model_dir 对开发组三场景可能实际采用 ggs_ex_trunc4，而非 ggs_pipeline；后者的点云目录也可能是链接。先逐场景解析真实路径，记录 iteration_30000 的所有模型文件、cfg_args、相机及归档 JSON 的 SHA256。确认旧 JSON 与该高斯及 trunc4 的对应关系，不能拿同名目录中另一轮高斯比较。清单、依赖或对应关系不齐即 BLOCKED；不要自行换对照。若最终 Release 清单尚不可用，遵循本计划的资产门槛等待，不用“附件看起来齐了”代替核验。

乙重提对照与归档不一致时，甲保留两套数值并暂停最终判定，组织者确认统一评测口径后再决定；不能一边看甲结果一边选择更有利的控制组。归档对照仍是本轮预注册主比较；新对照是复现诊断，改变主比较需单独记录，不能覆盖原表。

### 6.2 明确 patch 与 NCC 的真实语义

loss_utils.py 中 patch_size=3 对应 total_patch_size=(2*3+1)^2，即49个采样偏移，偏移还乘0.5；不能解释成普通3×3窗口。CUDA核还可能有自己的固定循环，必须在源码审查记录中分别写明，不能为凑一致改核。核返回 cc 已经是 cross²/(variance_r*variance_n+1e-8)，Python 使用 clamp(1-cc,0,2)，不能再对 cc 平方，也不要切换 ncc_exact_backend。灰度、法向归一化、内参、图像缩放、背景和采样坐标都沿用现有 PatchMatch。

patch_size=3、pixel_noise_th=1.0 只是当前计划预期值；先核验实际训练配置和参数来源，不因为默认值相同就声称历史配置已知。核验不符或缺失时记录并暂停，由组织者决定是否另立实验，不静默改计划。nearest_id[0] 是固定的排序候选，不应称为严格复现随机单近邻训练的逐次掩码；它是预先固定的提取策略。保存完整相机顺序、nearest_id列表、索引到图名的映射与选择规则，核对 nearest_id 指向的是同一 train_cams 索引空间，不对相机列表排序后继续沿用旧索引。只通过图名选择预览，不重排场景内部相机。

### 6.3 稠密返回必须显式重建，不能复用相关图当有效图

PatchMatch 仅在 d_mask 的 valid_indices 上调用 CUDA。现有 return_corr_map 以0填未评估位置，0并不表示可信或已经计算过。新增默认关闭的稠密诊断返回应包含 evaluated、geometry_valid、warp_valid、error、keep（均明确 H×W、dtype），用 valid_indices 散射回原图。error 未评估处用 NaN，keep 默认 false；nearest_cam=None、d_mask全空、单个有效点、CUDA全无效等分支也必须返回同形状的全拒绝结果，不因 squeeze 丢维。没有近邻就该视角全拒绝并记录；整个场景无可融合深度或网格为空时记 FAILED，不放宽。

定义 B=有限且正的 median_depth 且通过既有 gt_mask。最终 K=B & d_mask & evaluated & warp_valid & isfinite(error) & (error<0.9)。不要拿先置零的深度做 warp：保留原 render_pkg 计算掩码，在独立深度副本上用 where(K,原深度,0) 后送 TSDF。原无效深度和非有限深度也为0。未通过像素不保留颜色深度投票，不给保留深度额外置信加权。默认关闭时，不增加 NCC 调用或变动随机数、训练输出元组、损失及梯度；新增诊断仅 detach，不改变训练运算图。

### 6.4 统计与冒烟检查可机器核验

前文“三类比例”需补成互斥完备分组，分母统一为 B 的像素数：
1. keep：K；
2. NCC拒绝：B内已经几何有效、已评估、warp有效、有限error，但error>=0.9；
3. 几何/warp未通过：B内 d_mask失败、未评估、warp无效或error非有限的其余像素。
第三类再细分原因，不能全部误称为warp失败。另报原始H×W、B外mask数、非有限深度数。B为空时计数为0、比例为null并写原因，不能除0或伪造0%保留率。三类计数之和必须等于B计数。新深度非零数必须等于K计数；K处深度与原深度逐位一致，K外全0。

scan24固定预览在加载相机后按 image_name 选最小者（不改内部列表）。先连跑两次，选择、掩码哈希、计数一致；同时验证默认关闭的新 PatchMatch 与原版本的返回项、loss及同上游梯度在固定小输入上的一致性，不执行optimizer。验证大于1点、单点、空候选、无近邻、gt_mask边界、NaN情况。保存原深度、K、error、evaluated与warp_valid的无损数组和红掩码PNG；没有必要为了冒烟完整建网格。只要一致性或计数门槛失败就先修复并保留失败记录，不进入三场景提取。

### 6.5 配准与评测不是仅改一个文件名

mesh_extract.py 已有 --write_dir，优先复用，不重复增加输出接口。evaluate_dtu_mesh.py 还会写 recon_aligned.ply，并调用外部 dtu_eval/eval.py；必须让对齐网格、vis与日志全部位于新目录，模型/相机仍从旧模型目录只读加载。现有 os.system 可能吞掉子进程失败状态，应在新增路径中可靠检查退出码；失败或陈旧JSON不能参与比较。

保留 P5 的相机顺序、尺度与刚体配准、真值、裁剪、采样数量/随机种子与单位。固定 seed、记录依赖版本和完整命令；不要把每次重算配准称为逐位确定。核验 eval.py 的 s2d 字段定义及JSON实际字段名，不能把d2s误当s2d；核验坐标换算，不直接把高斯坐标米误当评测毫米。归档 JSON 和新 JSON 以未四舍五入浮点值做三场景等权平均，不用展示表的舍入值决定过线。

有效实验的三条门槛仍为 mean(delta_overall)<=-0.0005 mm、mean(delta_s2d)<=0、delta_overall_scan122<=0.0005 mm。有效但不过线记 FAIL；资产/配置/评测不齐记 BLOCKED/INVALID，不能算科学负结果。低保留率不放宽规则；空网格/评测失败只说明本次流程没有有效几何结果，不伪造无限Chamfer或零分。记录顶点、面、连通分量、保留率，辅助解释挖空；第三条质量门槛不能证明完全没有缺面。

### 6.6 交付与结论边界

新输出根 ggs_trust_fuse 若已有结果即停，组织者另给新run名，不能覆盖。按场景保存 input_manifest.json、resolved_config.json、camera_pairs.json、preview回执、per_view_metrics.json、原始/后处理/对齐网格、vis/results.json、评测日志和哈希。根目录交 verdict.json、三场景CSV、同固定视角红掩码PNG、执行环境/源码diff与最终摘要。图例区分GT外无效区和被可信带拒绝的有效深度；不为了展示选择新视角。记录耗时与峰值内存，不额外跑PSNR，不训练，不做局部截断第二刀。

新增实现经组织者审查冒烟回执后，再执行24→105→122，三场景齐后统一评测与判定。交回审查者的最小包：commit、diff、输入/输出哈希、完整命令/退出码、preview计数与数组、三场景新旧JSON和CSV、配准参数/评测seed、网格和图的路径、失败日志。缺证据的PASS不能接受。

不过线的准确结论是“在本三场景、固定近邻和冻结配置下，该强掩码未达到预设改进线”。不能据此推导“8mm就是最佳建面宽度”或“剩余误差不来自训练”；通过也只是开发组三场景经验结果，不是新算法或泛化证明。团队材料中的因果化表述应相应降格。
