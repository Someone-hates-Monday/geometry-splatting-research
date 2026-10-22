# 生产FP32预处理探针：既有真实输入局部前向通过

2026-10-13。`production_packet_v3` complete、exit0；同一方案的v1缺ninja路径、v2误选conda CUDA
导致架构/头文件编译失败，均保留manifest、源码和错误日志。它们不是科学反例，也没有触发GPU训练。
v3按项目已有CUDA12.8/gcc11配置成功编译并执行；生产源码未修改。

独立扩展逐字提取生产 `render_forward.cu` 的 computeColorFromSHSG、computeCov2D、preprocessCUDA，
直接启动完整预处理内核，保留原中心投影epsilon、clip、conic逆、plane/width及culling。
用生产GaussianModel的filter scale/opacity和归一化rotation，SH颜色用预计算零值绕过。
这是实际CUDA操作的诊断扩展；不是已安装renderer二进制相等性证明。源与fast-math相同也不保证FMA指令一致。

原24射线的1992个selected输入全部保留，含46个clip，culling=0。
开发门槛仍peak相对1e-5、width相对1e-4(两者floor1e-8)、alpha绝对1e-4；无效不裁正。
生产UV恢复qmean/qbar后：

- NumPy double decode：0失败，最大p/w/alpha误差4.95286e-7、1.18496e-6、1.18154e-5。
- NumPy FP32 decode：0失败，最大5.53900e-7、1.29782e-6、1.18168e-5。
- Torch CUDA FP32 decode：0失败，同上述最大误差。

两个预定额外几何诊断臂也全部通过；其额外相机比率或double几何不属于仅有生产packet的部署方案。
原期望来自已独立80位原PLY复核的double世界系数；这次不再次宣称独立80位重算全部原参数。

`results/production_packet_report_v1.json` 核验全部源/archive、产物、生成文件、二进制和3原PLY哈希。
全部5976个CUDA输出分量重放逐位一致，并额外实际检查Torch CUDA的den/det/energy，0无效。
原采集脚本CUDA分类借用了NumPy诊断，此额外检查补齐GPU诊断范围；仍不是独立decoder实现。

既有真实输入局部通过不改变FP32合成压力失败，也不推翻同分布新4096构造拒绝两保护候选的结果。
当前没有完整renderer反向/共享梯度归约、运行成本、GT几何/图像改善或新颖性结论。
下一项需先完整CUDA/VJP与端到端成本，再做冻结模型affine/exact packet/direct factor三臂GT比较；
只有通过预定门槛才考虑训练。这里没有启动新训练。
