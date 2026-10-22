# 真实TGS checkpoint上的冻结细分诊断

2026-10-20。承接已完成的数学与合成CUDA验证，本轮已取得真实官方模型，并执行固定选择的
8个离线细分操作×4相机×3 appearance条件。没有优化器更新或训练。
这些是**真实发布模型上的模拟事件**，不是捕获的训练事件，不构成质量/效率提升结论。

## 来源与输入

固定官方主库README的[MipNeRF360预训练链接](https://drive.google.com/drive/folders/1YrH9IVU8QWgfnIg_i0iRHq_M9rNCNPyV?usp=sharing)
当前公开文件为`full_eval.zip`，15,832,586,976 bytes。通过正常公开下载表单和标准HTTP Range
取得ZIP中央目录，再按**查看渲染结果前确定的最小压缩传输量规则**，从18个候选中选择
stump/iteration_7000。并非按效果选择场景或迭代。

只提取模型、hyperparameters、cfg_args、cameras.json及一张发布GT图；新网络传输约163.58 MB，
无需下载15.8GB整包。范围缓存共40块，ETag/Last-Modified各自一致，所有ZIP成员CRC通过；
每个文件保存SHA256及HTTP/ZIP来源记录。没有验证整包SHA或作者签名，不声称知道训练时commit。

模型为784,056个真实三角形，active SH degree=3，sigma范围约[0.216805,3.532998]。
主模型SHA256为`9898fce495a105f0aa3fed9f0cbb181a68b1438c5eab046b62c9ab0f55a73620`。
张量与相机在CPU审计通过。主张量保持weights_only受限读取；原超参数的NumPy float32/64
标量经明确类型白名单读取，没有启用任意pickle执行。

按固定源码还原`sigma=0.01+exp(raw_sigma)`、`opacity=sigmoid(raw_opacity)`。
发布模型不保存训练mask，官方load()将raw mask重置为1，forward的有效STE mask为1；
本次按该**推理加载语义**执行，不声称恢复训练mask/优化器/历史随机数/选择状态。
同理，7000状态是保存时快照，不从文件名推定它对应训练循环内的哪次Adam/细分前后。

相机JSON是原COLMAP的4978×3300，不能直接当作实际渲染尺寸。cfg使用images_4和resolution=-1，
发布GT图头核验为1245×825，与缩小4倍的取整大小相符；FoV仍由原宽高/focal比重建。
125相机按名字排序、每8张留出，且JSON中test+train序列与固定源码一致：109 train、16 test。
选取各split中两个预先固定SHA256排序最小的相机：test `_DSC9269`,`_DSC9221`，
train `_DSC9258`,`_DSC9322`。train图像未逐个解码，使用同一已声明场景分辨率。

完整来源在`sources/tgs_pretrained_listing_v1/`及`sources/tgs_real_checkpoint_v1/`；
字段/相机审计在`experiments/tgs_real_event_v1/input_audit.json`。

## 固定选择、交叉条件与结果

在4相机的原始baseline上，114,642个父基元满足有效opacity>=.01且至少一视角原生外接半径>24px。
取其固定SHA256顺序前8个；该半径条件是固定源码split_size默认值的**离线代理**。
没有训练时完整历史image_size、importance/dead mask或抽样状态，不能称作真实训练选择。
这是程序在split结果计算前冻结的方案，不是第三方预注册或独立审查的选样。
筛选使用相机几何和checkpoint，不使用操作后的变化量。事件ID、全部候选ID及其哈希已在
任何细分渲染前保存。

每次只移除一个父基元，按原始顶点顺序生成四个世界边中点子三角形，复制sigma、opacity和SH；
逐字段确认其它基元不变。对同一状态比较原始SH/修正SH/常量白色三种条件的split前后。
三个appearance条件的alpha输出逐像素一致；其RGB扰动不能彼此直接解释为质量优劣。

3次重复原始渲染的最大差为0，事先设定像素变化阈值3e-5。
32个事件—相机组合全部保留：19个父基元通过原生预处理且有贡献，对应19个超过阈值的变化案例；
另13个为零变化。原始SH的最大单通道局部变化 **0.0477826**，最大**全图**MAE仅
**3.85572e-6**。不能把局部峰值当作全图退化，也不能把32个相关案例当作32个独立场景重复。

修正SH下最大局部变化0.0464182、最大全图MAE3.67698e-6；常量白色下为0.0169717和2.10339e-6。
这些只是不同appearance下的扰动大小，不是训练收益或显著性检验。
原始与修正SH的完整baseline本身可相差0.1618–0.4385，说明旧模型可能已适应错误shader；
禁止据此宣称修正旧模型必然提高PSNR。修正原生degree3与独立固定Python预计算SH路径
在4个真实相机上的最大差<=1.78814e-7，进一步核实了修正方向计算。

每个条件的差图保存包含所有非零变化的最小矩形，矩形外为已核对baseline；
指标以完整1245×825×3为MAE分母，不按ROI归一化。96份NPZ已由CPU重新计算全部指标，
并核对父/子参数、确定性选样、baseline切片及三种appearance的相同alpha。审计通过，
但它不是独立GPU重放或独立同行评议。

## 跨视角剔除反例与数学补充

事件2的parent16010，在测试相机0外接半径30px，进入候选集合；在训练相机2外接半径20px，
但内切半径仅约1.80092px。四分后四个子基元的原生半径全部为0。
重新投影的最小子内切半径约0.89062px，其余子半径也小于1；正深度、opacity和法向角度
检查通过，因此与源码的`inradius<1`剔除条件一致。该组合的局部RGB变化约0.00683135。
这证明合成的小三角形消失机制确实出现在该真实模型的一次模拟操作中；不能从1例估计总体发生率。

已另存全称证明：对正深度针孔投影，定义`kappa=max vertex depth/min vertex depth`，
世界中点四分的四个像平面子内切半径满足锐界

    min_j r_child_j >= r_parent/(1+kappa).

证明见`TGS_PERSPECTIVE_INRADIUS_BOUND.md`：角子由包含的相似三角形下界，
中心子由精确面积比例和Minkowski周长上界；并给出取等构造。
SymPy恒等式、1500个固定seed几何样本、kappa取1、2、10、100的取等构造及31个正深度实景投影核验通过。
标准凸几何引理已核读大学作者原始讲义；这些工具和数值一致性不证明该锐界具有发表新颖性。

在本次19个可见父组合中，18个满足理论上的`r_parent>=1+kappa`尺寸充分条件，
唯一不满足的是上述全子剔除例；该例kappa约1.03092，理论子半径下界约0.88677px。
这里只保证理想几何的内切半径，不能保证其它cull、浮点边界或未来未观察相机。

## 下一决策

本轮排除了“纯合成才有这种现象”的疑点，但单事件全图平均变化很小，尚不能确定它是有效主线。
下一步优先核读相关细分/三角形质量理论，检查锐界及其用途的新颖性；再评估基于可见性和
内切半径的细分选择能否保留相同新增容量，或采用更合适的局部拓扑/窗口过渡。
必须与同样新增基元数的对照比较，记录拒绝率，不能靠不细分获得表面上的保持效果。
排序与遮挡的因果贡献尚未单独识别。本轮没有使用GT评价几何/图像提升，没有重训或速度基准。

下一完整训练阶段仍超出“数学+小规模CUDA”授权；在此之前可继续公式/梯度/冻结counterfactual核验。
GGGS先前30k的负结果保持，研究总目标尚未完成。

复现入口依次为`extract_tgs_real_input.py`（已缓存时不重新下载模型）、
`audit_tgs_real_checkpoint.py`、`run_tgs_real_events.py`、`audit_tgs_real_events.py`、
`verify_tgs_inradius_bound.py`、`plot_tgs_real_events.py`。WSL Python路径沿用上轮报告。
`run_tgs_real_events.py` 检查到完整结果时拒绝静默重启，应先审计现有文件。
本轮源下载64593、CUDA55733、结果审计89276均exit0；其余命令也已终态，无训练等待。

图`all_event_effects.png`完整展示32个离散案例，不连接任意相机次序；
`real_event_example.png`明确标注为事后选取的最大局部峰值示例。
RGB显示仅为图示裁剪到[0,1]，指标始终使用未裁剪浮点值；两图已目视检查。
