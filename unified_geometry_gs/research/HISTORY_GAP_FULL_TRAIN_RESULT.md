# 原始完整训练接入：24次更新完成，30k质量试验已启动

2026-10-15。本阶段是实质进展：从完整状态恢复的真实train.py，实际执行
NCC/几何损失、增密/filter和Adam；不再是fresh-Adam RGB-only smoke。
整个科研目标仍未完成，当前没有新的GT或留出图像改善结论。

只读预检确认common6999 checkpoint SHA为
6c9809bb0454af964192bfc79fa4b0f5155ca1f57fdff694c15b40af39c76ccf，
runtime sidecar SHA为ed74b165cb6155a0acb7f7ced77191841831ca70107a0b5b793d8f28a183aab6，
完整state digest为68b6c35a71323d882de1fc89516813cf7d1e1edb2e9b05d2103bb01552b0b840。
676277基元、752个原有+Inf opacity，42个训练视图及其队列/RNG/filter/增密统计均恢复。
这与之前49-trained PLY开发接入是不同起点；不把旧模型当新heldout起点。

隔离wrapper train_history_gap_branch.py通过作用域限定Tensor.backward hook，把真实
当前camera的128均匀采样loss加到原loss；不修改train.py/renderer/model。
原recipe NCC .6、geo .02、depth-normal权重0，7000开始正则，原30000 schedule。
原增密/pruning/reset/filter均保留。当前+Inf opacity mask在每次实际Adam前重新读取，
不冻结旧拓扑索引；普通Adam前后mask不变、opaque梯度0，其他参数/所有梯度/Adam有限。

history_gap_full_smoke_v1：disabled_a、disabled_b、computed_zero、weight.01四个
独立进程，各7000..7005六次Adam，共24次。四次完整状态digest、相机序列
0022/0033/0011/0019/0020/0005一致，zero与positive每步128像素ID相同；不消耗训练RNG。
实际train/test池42/7；NCC每臂6次，filter guards实际执行，test report0。
7000原增密后各677291基元。首个完整梯度在增密前保存，不能因此断言全部梯度在参数
替换之后仍进入第一次Adam。后续full wrapper会逐步记录optimizer处缺失gradient组。

首梯度相对L2，依位置/opacity/scale/rotation顺序：
disabled重复1.78034e-7、1.35135e-7、6.97151e-5、2.47993e-4；
zero对disabled1.57166e-7、1.50192e-7、4.62909e-5、3.00476e-4。
全部组满足预设max(1e-5,5*重复差)工程界；不是bitwise或统计等价。
positive对disabled分别32.8867%、15.2524%、14.8272%、22.7230%。
颜色组约2e-7重复量级，SG/appearance零差。原损失项首观测四臂相同。
相机和像素在变化，六步reg数列不可当固定目标的下降曲线，更不等于几何改善。

首区间从backward入口开始，省略forward且有冷开销；后续区间覆盖上次Adam后至本次
Adam结束，包含forward/NCC/原拓扑操作/capture/loss/backward/finite检查，另扣首梯度
产物传输写盘。没有把此计时当标准完整训练加速对照。
实际进程wall28.41/23.63/23.69/35.22秒，包括加载和产物；均exit0。
父guard把已知DiagnosticComplete写为exception/failed，外层仅在实际6步、状态与相机
guards全部检查后标记intentional diagnostic complete；未伪称父完整训练结束。
verify_history_gap_full_smoke.py独立重新读梯度并复算，63次来源检查、46产物哈希通过。
源/archive、所有原始梯度、logs、第一步和全部六步标量保留，没有改变原门槛。

后续已固定HISTORY_GAP_PILOT_PROTOCOL.md并启动history_gap_pilot_v1：
control关闭正则、history_gap权重.01，从common6999各跑到30000，23000次Adam；
最后30000反传/report/save、不做Adam。train_history_gap_pilot.py逐步JSONL，
每100步compact进度，首次实际query/CSR/public bit一致检查，最终完整runtime sidecar。
run_history_gap_pilot.py在两组完成后自动提取42-view mesh、按实际camera ID对齐，
official GT seeds0/1/2、TRAIN42/TEST7图像评价，保存判据与每项失败。
只是一对共享warmup的探索；场景有先前暴露，SfM可用所有视图；采样seed不是训练重复。
预设GT overall≤.99、两方向各≤1、TEST PSNR≥-.10dB/SSIM≥-.001、容量≤1.05、
实际全训练资源wall≤2；全部通过才推进独立种子/多场景/机制消融，失败不自动扩大调参。

运行真值是history_gap_pilot_v1/manifest.json、实际PID、子进程JSONL和退出码。
不要因聊天没有结论、manifest停留running或一次观察超时重启进程。
只在已知session/PID确实terminal或缺失后检查失败文件，再决定后续动作。
