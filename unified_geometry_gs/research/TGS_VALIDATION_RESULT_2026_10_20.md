# TGS 数学与固定输入 CUDA 核验结果

本轮授权：“恢复数学核验和小规模 CUDA 诊断”，不含完整训练。2026-10-18 的执行在
已完成主要核验后中断，2026-10-20 读回原始结果并完成 SH 缺陷复现、隔离修正和系数梯度检查。
这是可复核的数学/实现结果；尚未证明真实数据上的质量或效率收益，也没有完成新颖性审核。

## 1. 数学核验与具体透视反例

`verify_tgs_moments.py` 使用 SymPy 1.14.0 精确分数/符号化简，独立 NumPy 边距窗口积分，
以及 SciPy 1.17.1 提议证书。输入、收敛序列和结果位于 `experiments/tgs_subdivision_v1/`。

- 连续窗口质量为 `2 o A / ((sigma+1)(sigma+2))`，重心为 `(3g+sigma s)/(sigma+3)`。
  对 3-4-5 三角形，sigma 为 0.5、1、1.16、4、20 的独立二维积分均通过；
  sigma=0.5 的质量误差随每轴节点 64→128→256→512 从 2.30e-6 降至 4.59e-9。
  不能将首次 256 节点误差 3.66e-8 大于 3e-8 阈值当作公式失效；已保留收敛序列。
- sigma=1 的单视角 opacity-only 可行阈值精确为 `o <= 31/34`；
  极小最大质量权重为 `(17,13,15,17)/62`。双正交视角共享解要求 `t=-1/2`，违反非负约束。
- 同内参针孔相机的具体反例取先前推导的 `D=100 sqrt(7)`，父 opacity=1/2、sigma=1，
  真正投影世界边中点，未把透视下的中点误当作像平面中点。
  第一个预定候选 `D=10 sqrt(7)` 没有得到分离证书；不声称任意距离均不可行。
- 两个视角分别存在严格 `(0,1)` 内的解，数值约为
  `(0.5483534595,0.4201823092,0.4830906152,0.5483870968)` 及交换第2、3项的解。
  这由区间 Cramer 法验证，不是仅检查浮点残差。
- 共享系统写为 `Az=b, z>=0`。有理向量 y 的严格区间核验得到 `A^T y>0` 且
  `b^T y < -2.78158`，从而共享非负解不存在，甚至不需要子 opacity 的上界。
  `verify_tgs_perspective_certificate.py` 仅使用 Python 标准库和精确 Fraction 运算，
  每个平方根通过整数平方界包围；已在 Windows 独立执行，完全不依赖 LP 求解器。

证书进一步给出：对于任何共享非负子 opacity，两个视角中至少一个的理想连续 alpha 场
相对 L1 误差不小于 **0.0068915072956（约0.689%）**。这是保守的可证下界，不是最优误差，
也不是实际图像 PSNR/几何误差下界。推导与条件见 `TGS_SUBDIVISION_MOMENTS.md` 第8节。

关键原始文件：`math_results.json`、`distance_candidates.json`、`perspective_certificate.json`、
`perspective_verified_with_l1_bound.json`。后二者分别是输入证书和标准库重新验证的输出。

## 2. 原生 CUDA 与独立参考

固定主库 `685ff270a8273afc3a2b094ae3148dd0b98e31e9`；原生库
`6d61f4c0a571c635e37ffb740250706fa30a5541`；GLM
`5c46b9c07008ae65cb81ab79cd677ecc1934b903`。原生23个文件与429个 GLM 文件均逐个
按固定 Git blob 验证，清单在 `sources/tgs_native_diagnostic_v1/manifest.json`。
没有 pip 安装、包升级或替换 GGGS 扩展。构建在项目专用目录，名称与 GGGS 不同。

环境：WSL Ubuntu，Python 3.12.12，PyTorch 2.12.0.dev20260314+cu128，CUDA 12.8，
RTX 5090 Laptop，sm_120，gcc/g++11，MAX_JOBS=2。微型 GPU 算术与同步通过。
原生 forward/backward 均编译成功；本阶段只验证了下述有限前向和后述 SH 系数梯度。

`run_tgs_native_diagnostic.py` 完成18项固定输入，涵盖大三角形的父/四子、sigma1/1.16、
亚像素尺寸、opacity 阈值/cap、重叠早停、两个有限透视视角。使用常量白色/黑背景。
`tgs_native_reference.py` 独立实现像素点采样、原生 tile 矩形、阈值、剔除和早停。
5次重复的输出最大差异为0；参考容差事先设为3e-5，18项最大误差为 **5.13334e-6**。
这不是严格浮点误差界，也不覆盖任意输入或所有梯度。

大三角形 sigma=1：alpha 总和 1600.02468→1600.15012（约+0.00784%），
x重心 90.24854→92.74804 像素（偏移2.49950，与连续预测2.5接近），
相对 alpha L1 变化0.618618。总量几乎不变并不意味着空间场保持。
父内切半径1.5像素的例子：父 alpha 和为2.28335，四个子半径约0.75像素，均被剔除，
输出变为0。不能把这两类变化混写成“缺少总透明度”，也不能据此宣称真实场景收益。

`cuda_results.json`、18份 NPZ 和 `subdivision_diagnostic.png/.svg` 保存原始/参考输出。
图已目视检查；所有 alpha 面板共同色标，差值为 children-parent，无插值或局部增强。
首次构建收据记录旧模块哈希；增加 SH 修正所需 include 搜索路径后，原始源码模块重新
编译为另一哈希。各实验结果分别记录实际模块 SHA256，不将不同二进制混作字节相同。

## 3. 已复现并隔离修正的 SH 方向缺陷

原生 `forward.cu:410` 与 `backward.cu:293` 使用

    (glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)

C++ 括号内是逗号表达式，结果只保留 z，标量 vec3 构造器把它复制成 `(z,z,z)`。
同一 z 下改变 x/y 的三个输入，原始 CUDA 都输出同一 SH 颜色；与错误方向的颜色差仅
2.53e-8，与真实中心方向的最大 RGB 差为0.12168–0.13538。非零低阶系数直接区分了两种解释。
该缺陷作用于原生 SH 路径；主库 `convert_SHs_python=False` 为默认值。Python 预计算颜色
和 override_color 路径绕过这次原生 SH 调用，因此不能把所有配置都称为受影响。

隔离副本将两处改为 `glm::vec3(x,y,z)`，独立模块名 `tgs_diag_6d61f4c_shfix_v1`。
修正后三个输入与真实中心方向颜色的最大误差 **3.17853e-8**；SH 系数梯度与解析值的
最大相对误差 **7.08105e-6**。原始版本的系数梯度也符合错误方向，不能只修 forward。
没有验证几何、sigma、opacity、相机梯度的完整正确性；没有用修正扩展训练。

原始和修正结果分别在 `sh_direction_v1/results.json`、`fixed_results.json`，
生成源码、构建收据和可应用补丁同目录保存。原始归档未改。
初次诊断辅助函数原地归一化了记录用的中心数组，颜色比较不受影响，但中心记录错误；
已修正为复制后归一化，初始文件保留为 `results_initial_center_alias.json`，不作为输入中心证据。

这是固定版本上的代码缺陷，不是已证明的新科研方法。旧 checkpoint 可能已适应错误方向，
直接切换正确 renderer 不保证旧模型零步质量改善，不能用该方式替代公平重训对照。

## 4. 复现与当前边界

在 WSL、项目 research 目录中使用 `/home/someone/miniconda3/envs/uggs/bin/python`：

    python verify_tgs_moments.py
    python verify_tgs_perspective_certificate.py experiments/tgs_subdivision_v1/perspective_certificate.json --output experiments/tgs_subdivision_v1/perspective_verified_with_l1_bound.json
    python run_tgs_native_diagnostic.py
    python probe_tgs_sh_direction.py
    python probe_tgs_sh_direction.py --shfix
    python plot_tgs_subdivision_diagnostic.py

这里的 python 指上述完整路径，需在已准备的环境执行；backend 在自身进程内配置 Ninja/
CUDA/编译器 PATH，不改变系统环境。完整源码如需重新准备，使用
`prepare_tgs_native_diagnostic.py`（公开网络读取，缓存和固定 blob 检查，不运行下载的 setup.py）。
证书验证器可单独在 Windows 的标准 Python 执行。

`audit_tgs_validation.py` 已在 Windows 执行通过：核对452个来源文件，重新计算18份保存的
CPU参考和差异，复核严格证书与两版本各3个SH/系数梯度记录，并生成精确两行补丁与文件哈希。
`evidence_audit.json` 是保存结果的复核，不宣称再次执行了GPU或完成独立同行评审。

此前 GGGS 30k pilot 的几何失败结论保持。本轮没有新训练、真实 split-event、数据集图像/
几何评价或速度基准，广义科研目标尚未达成。原 LaTeX 文档未改，因此没有新编译或更换标签页。

## 5. 下一步的可证伪问题

首先取得来源可核实的真实 TGS checkpoint 或实际 split 前状态，按
`TGS_REAL_EVENT_PROTOCOL.md` 冻结输入和事件选择规则。对同一状态做原始/修正 SH ×
split前/后交叉渲染，把窗口变化、尺寸剔除、SH变化和排序变化分开测量；先不重训。
检查真实事件是否频繁、扰动是否足够大、数学诊断量能否预测实际像素变化。
只有通过该门槛，才比较受约束细分策略与已有方法，并另行确定训练预算。
若真实影响很小，应停止把该现象扩展成训练主线。新颖性仍需对照2DTS、ETS、DETRIS等原文。
