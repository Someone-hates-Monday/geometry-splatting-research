# 重建实验审查：需补测与修正，暂不放行全量 trust

2026-11-17。依据本地审查包和实际工作树diff作独立审查。未修改生产实现，未运行GPU、提取或新评测。本轮模型身份不再BLOCKED；当前状态是 REVIEW_CHANGES_REQUIRED，不是科学FAIL。

## 已确认

- 三场景iteration_30000 PLY的大小与SHA256分别与审查包一致。这是文件身份核验，不替代本轮独立加载全部sidecar。
- scan24保存数组为581×777，B=249922、K=186846；逐像素重算 B & geometry_valid & evaluated & warp_valid & isfinite(error) & (error<0.9)，与保存K完全一致，K内深度有限。保留率0.7476172566。
- scan24新control JSON overall=0.3554418991327293，主比较没有使用旧trunc4 JSON。两次一致性记录存在；本轮没有重新跑GPU来独立验证两次渲染。
- diff采用稠密scatter、默认关闭开关、独立深度置0、新网格/评测输出路径，核心方向正确。

## 必须在全量前修正

### P1：评测的随机打乱实际上未固定

`dtu_eval/eval.py` 的 `shuffle_rng = np.random.default_rng()` 从系统熵创建独立生成器。`evaluate_dtu_mesh.py` 父进程中的 `np.random.seed(0)` 既不会跨子进程传播，也不能给default_rng无参调用设种子。打乱发生在点云降采样前，可能改变保留下来的点和距离统计。不能以现状判断5e-4 mm门槛，也不能据此认定现有控制数值错误或真实波动有多大。

给评测子脚本增加显式可选seed并透传（本轮预先固定0，历史默认路径可保留无参行为）。不改配准、密度、裁剪或距离定义；调用子脚本用sys.executable及明确脚本路径，避免走到另一Python。保存子进程seed、命令、依赖和结果。使用同一control网格在新的seeded评测目录重复两次检查一致性；必要时记录无法逐位一致的来源，不为通过而放宽门槛。

原control JSON、网格和日志保留不动。三场景control/trust最终均采用同一冻结seed的新评测口径，补充协议和新路径；原scan24 control 0.3554418991仅保留为此前未固定shuffle seed的诊断值，不混作新seeded主对照。无需因此重新训练，也无需为了重评重复提取未改变的control网格。

### P1：默认关闭路径缺少数值兼容证据

`smoke_jia_trust_fuse_scan24.py` 的入口包在torch.no_grad下，且只有arity断言；注释所写grad check并未实现。返回2项不能证明loss/梯度相同。

新增独立的小输入兼容检查，比较git HEAD中未改的PatchMatch与当前版本：默认调用和显式return_trust_maps=False。固定相同输入、上游权重与随机状态，独立构图；正常有效、无近邻、d_mask为空、单有效点、CUDA全无效均覆盖。正常有梯度案例使用torch.enable_grad，比较NCC/geo loss以及对depth/normal等对应可微输入的梯度，记录grad=None/零梯度是否一致。不要对无图零返回强行backward，也不要运行optimizer。保留默认旧返回选项组合的arity/内容检查。

优先逐位一致；CUDA非确定性需要事先声明容差并用原版重复基线说明，不能看过新旧差异后调容差。记录baseline源码SHA、当前源码SHA、实际加载扩展、输入哈希和逐项误差。两边若都在no_grad下或梯度全未请求，不算完成。

### P1：新输出还缺防覆盖保护

当前extract/evaluate仍对目标目录直接exist_ok和写文件。评测仅检查results.json存在，不能排除旧文件被误读。新增路径应拒绝已有结果、拒绝与model/control/旧证据目录相同或嵌套冲突的写入位置；每次用独立空run目录。分别保护metrics_json、网格、aligned网格和vis。记录子进程退出码并确认本次产生的有限合法JSON，而不是仅检查某个旧JSON存在。保留兼容旧入口所需的行为，不批量改历史结果。

### P2：边界与可解释性证据未齐

NaN/无穷/单点/空候选/无近邻、gt_mask边界需要机器检查。确保新增keep显式包含isfinite(error)，输出K满足记录公式；灰度、法向和warp不另写实现。每视角统计把几何失败、未评估、warp失败、非有限error分清，保留总三类互斥分组；不能只给一个geom_or_warp合计就声称已解释全部拒绝。

对已有预览的可视化注意NaN不会满足depth<=0，不能漏进图例；红图应与B和K定义一致。空网格检查应在post_process前也执行，写PLY时检查返回值，避免空三角形或写文件失败被当成功。

## 顺序偏离

协议要求scan24冒烟审查后才训105/122；目前三场景已训完，属于执行顺序偏离，需记录启动时间、脚本与原因。不为此丢弃已有模型或重训，不根据质量选择模型；冻结本次三份哈希继续使用。训练授权本身已有，但不能把未遵守审查顺序省略不报。

## 下一停点与给Cursor的提示词

本轮不授权trust_trunc4全量提取。请只修正上述兼容测试、显式评测seed/输出保护和边界诊断；不要重训，不提trust网格，不写PASS/FAIL gates。先更新重建协议，冻结评测seed0与新目录。完成原版/新版loss梯度对照、scan24两次掩码检查、同一control网格两次seeded评测，全部另存回执，不覆盖本审查所引用产物。提交diff和每条问题的证据/未完成项，再停下来审查。

通过补审后，才放行24→105→122的trust提取，并补齐105/122同模型control；三场景控制与试验统一新seeded评测后才计算原三条门槛。keep约74.8%只是诊断，不是几何收益或通过依据。
