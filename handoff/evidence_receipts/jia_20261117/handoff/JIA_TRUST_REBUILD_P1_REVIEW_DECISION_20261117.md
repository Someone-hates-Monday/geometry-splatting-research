# 甲重建 P1 补审结论与全量执行边界

2026-11-17。结论：主要数值阻断已解除，条件放行本次冻结模型的三场景配对提取；完成下列两项执行前自检即可开始，不必为这两项再等待一次人工停点。不是科学PASS；不启动新训练。本审查未运行GPU/全量提取，依据实际代码、回执及独立哈希核验。

## 证据

compat_receipt SHA256=f509aa49fe13aecea8bde86e48706de30723ff879e12745f2defba8afd0a365e；normal_enable_grad记录两边requires_grad=true，NCC/geo loss及depth/normal梯度均有数值、max_abs=max_rel=0，并非两边None。它支持此次固定输入和默认参数的兼容性，不是全部可能输入的证明。

seed0控制评测run1/run2 JSON均SHA256=52fe52f57374301e5fbc28b7c87193dcf938666dd6423e300eb48ad791a1eab0；overall=0.35539816541829106，mean_s2d=0.4064133534126079，mean_d2s=0.30438297742397413，eval_seed=0。旧控制JSON SHA256仍为d0d163417837912350cd4f69ec9965195cf5703e525d063c3960ce0032792d4b。新冒烟两次一致，B=249922、K=186846，拒绝细分计数完备。

## 开始前两项机械自检

1. 完整冻结源码。git check-ignore实测eval.py被根.gitignore的/unified_geometry_gs/dtu_eval/排除，当前git diff不能交付关键seed改动。只将dtu_eval/eval.py作为源码单独纳入版本或完整源码快照/补丁及SHA256；不要取消整个数据目录的忽略规则。jia_io_guard.py、compat和smoke等未跟踪辅助文件也要纳入。记录本次运行实际加载的源码哈希，loss_utils应与兼容回执中的current_loss_utils_sha256一致。若不一致，先定位差异，涉及损失数值或图的变动须重做对应兼容检查。
2. 严格校验seed字段。evaluate_dtu_mesh当前用results.get('eval_seed', eval_seed)会把缺字段默认为正确，None也可绕过。对本轮eval_seed=0要求结果显式包含整数0，缺失/None/不同值即失败；不要仅凭父进程命令推定子进程已执行新脚本。此处只改结果验证，不改metric算法；新增缺字段、None、不同seed、正确seed的检查即可，无需为此重新训练。

保护逻辑已审阅，显式路径已有产物会被拒绝。执行时仍只使用解析后的独立空目录，明确传model/旧control等forbid_root，避免符号链接/别名路径误指旧证据。P2尚未穷尽，应保留未测清单，不能声称单个正深度用例已保证CUDA实际恰有一个有效候选。

## 给Cursor的执行提示

读取本结论，先完成两项自检并保存完整源码身份回执。随后放行 trust_trunc4，按scan24→scan105→scan122，每场景使用已冻结的同一model，参数tau0.9、固定nearest_id[0]、voxel0.002、trunc4、depth_max8、cluster1均不变。使用全新输出目录，绝不覆盖控制组、旧smoke或任何历史JSON。

scan24先提取并做eval_seed=0评测，使用已有seed0控制结果作配对基线。105/122先补齐各自同模型的control_trunc4和seed0评测，再做trust和同seed评测。不能把scan24控制数值套用其他场景，不能把旧未设seed JSON掺入主表。每场景记录per-view计数、相机映射、输入模型哈希、源码版本、参数、耗时、网格/JSON哈希与退出码。

每个场景结束先检查模型未变、掩码计数完备、网格非空、评测字段有限且seed正确。异常即停止并保留失败回执，不扩大预算、不放宽规则。单场景质量不佳也不能改近邻/阈值；在技术有效且资源正常时按原协议完成三个场景后统一判定，避免挑好场景停止。

最终三条同时成立才PASS：mean(delta overall)<=-0.0005 mm，mean(delta mean_s2d)<=0，scan122 delta overall<=+0.0005 mm，delta=trust-control。用未舍入原数值，三场景等权，不用keep比例代替几何结论。失败/缺失与科学FAIL分开，不做局部截断第二刀。

交回三场景配对JSON/CSV、verdict、网格路径、固定视角红图、输入输出与完整源码manifest、命令/环境/日志/退出码及未测边界清单。完成后停止，由我审查最终结论；不自动扩其他场景，不重训、不上传新成果或改变论文结论。
