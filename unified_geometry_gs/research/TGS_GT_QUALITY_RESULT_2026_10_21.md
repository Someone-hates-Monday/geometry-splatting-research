# v11：GT身份已确认；全精确窗口路线未显示相称的固定权重质量收益

协议于2026-10-20冻结，2026-10-21完成图与封存收尾。本轮仍只进行固定输入诊断，没有优化器、完整训练、相机调整或调参。
最重要的结论是研究方向需要调整：coarse保持native64输出及此前的算子加速成立，但相对数倍更快的piecewise，其两张已确认GT的质量变化近乎为零且方向不一致。
此外，未修改源码的归档native对照在room固定权重上更好。需要先解释既有修正与公开权重的兼容性，不能继续把精确参考上的速度收益当作足够的学术成果。

## GT身份：由原始照片像素验证，不靠整数索引猜测

固定使用v10在输出前选择的两张test视图：flowers `_DSC9128`、room `DSCF4915`，同一iteration7000检查点、相同相机、SH3和有效推理mask1。
从[Mip-NeRF360官方项目页](https://jonbarron.info/mipnerf360/)所指的两份公开ZIP中，仅提取指定images_4/images_2原始JPEG；从[TGS官方模型入口](https://github.com/trianglesplatting/triangle-splatting#trained-models-and-rendered-test-views-on-mipnerf360)的archive中提取预测索引的GT。

- flowers：原始`flowers/images_4/_DSC9128.JPG`与公开GT `00011.png`，均1256×828；解码RGB全部字节相等，差异通道值数0。
- room：原始`room/images_2/DSCF4915.JPG`与公开GT `00031.png`，均1557×1038；解码RGB全部字节相等，差异通道值数0。

另核查固定源码的排序、modulo8 test划分、shuffle=False及枚举保存顺序。未做resize、裁剪、配准、EXIF转置或颜色/曝光变换。
目录元数据先获取，成员随后冻结；全部四个图像通过CRC/SHA。累计成功Range载荷7,370,982B，未下载两份共约17GB的完整数据包，也未上传研究数据。
GT所在ours_30000目录表示保存目录；已通过原始照片确认其身份，因此可用于相同照片的7000步权重诊断，不把它冒充30000步模型评估。

## 比较范围及度量

三种原定只读实现为oracle、coarse、历史piecewise。读取GT像素和分数前，另版addendum固定加入未修改源码的archived_native，以检查公开权重兼容性。
后者对应归档native commit6d61f4c及SO SHA9d974f09...，source_edits=false；实际生成公开权重的training commit仍未证明，不能将它称为已核实的训练环境复现。
四种实现均保留，不按分数剔除。它们包含不同的既有代码修正，不构成单一精度因素的消融。

只执行八次前向，输入、模型及相机哈希保持不变。新oracle/coarse的RGB、aux、半径、maxblend和路径逐字节复现v10保留输出。
所有raw RGB、辅助输出、PNG格式数组、浮点数组和逐点差分均保存。flowers少量raw颜色超过1，主/次定义都明确处理，未作选择性修图。

主度量按公开保存顺序执行float32 raw*255、+0.5、clamp[0,255]、uint8，再以/255的RGB计算MSE、PSNR、SSIM。
次度量使用clipped float RGB，避免PNG量化掩盖变化。SSIM固定11×11归一化Gaussian、sigma1.5、C1=.01²/C2=.03²、same-size零填充5、全RGB像素平均，无裁剪。
这是明确边界约定下的本地度量，不声称完整复现某个论文的全测试集评分。没有LPIPS或几何质量分数。

## 固定公开权重的结果

主PNG格式的PSNR / SSIM分别为：

- flowers：oracle=coarse **20.096742dB / .40391935**；piecewise **20.096153dB / .40394790**；archived_native **20.139036dB / .40623584**。
- room：oracle=coarse **29.690647dB / .79260139**；piecewise **29.693259dB / .79265974**；archived_native **30.871019dB / .80293880**。

coarse相对piecewise：

- flowers：PSNR **+.00058895dB**，SSIM **−.0000285491**。
- room：PSNR **−.00261213dB**，SSIM **−.0000583578**。

coarse相对archived_native：flowers **−.04229389dB**，room **−1.18037133dB**；两图SSIM也更低。
clipped float定义下结论相同：coarse-piecewise的PSNR变化为+.00060956/−.00260212dB；coarse-archived为−.04246111/−1.18195195dB。
因此差异不是由选择PNG还是浮点定义反转出来的。coarse/oracle质量相同是前向等价的必然结果，不是新的质量提升。

独立审计全部通过：用精确整数平方误差核验主MSE、Torch单独复现PNG转换、分块另算浮点MSE/PSNR、交换滤波顺序重新计算完整SSIM，并做3888次直接图块通道核验。
图`quality_views.png/.pdf`展示完整两视图，未选取有利局部；差分是coarse与archived_native的通道平均绝对RGB差，统一色标[0,.10]，饱和比例保留在metadata且图注披露。
人工视觉检查已确认标签、分数、色标与完整画面无裁切。

## 对研究主线的判断

v10在相同test视图的coarse/piecewise配对耗时中位比为flowers5.008、room3.876（不同会话的已封存成本证据，不在本轮重测）。
本轮两个视图没有显示能抵偿这一成本的固定权重质量提升。应降低“全精确窗口再加速”作为方法主线的优先级，保留其作为精度和梯度诊断基线。
这项筛选不是对整个TGS方法、真实几何或正确重训练结果的否定；两张目的选取的照片也不足以估计全测试集或总体效果。

尚未定位归档native与修正实现之间的主要质量变化来源。一个明确的待检验候选是旧源码SH中心构造中的C++cast/逗号表达式修正，但四方法之间的混合差异不能单独证明该因果归因。
源码修正可能在数学上必要，同时破坏旧权重对原渲染器的补偿；固定权重结果不能代替重新训练效果，也不能成为直接撤回正确修正的依据。

下一步先冻结单一代码因素的前向消融，对照未修改native、仅SH构造修正及后续几何/窗口版本，并加入事先选择的新相机，区分解释与新增验证。
对应数学问题是：改变SH评价中心后，是否存在能保持旧视角颜色函数的系数变换；若不存在，应明确其条件和不可表示范围，而不是假定“换成正确中心”会保留已训练函数。
之后围绕可保持函数的几何/属性再参数化或有证据的优化机制提出方法，并完成先验检索；当前不主张新颖性，不恢复完整训练。

总体目标仍未完成：学术新颖性、显著图像/几何收益、效率与质量的有效折中、跨更广场景及训练验证均待完成。v9/v10封存证据、原始源码归档和打开的median_depth_note.tex未修改。
