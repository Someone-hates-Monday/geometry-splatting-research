# 修正根的完整参数链VJP：预注册诊断

2026-10-15。主集合为dynamic_root_v1原固定891条认证探针，不重新选易通过射线，含9既有异常。完整源节点51524均保留。对位置3、原log-scale3、原quaternion4、opacity-logit1共11参数的逐基元VJP进行检查；filter_3D、相机、实际query、贡献集合固定。

数学对象须明确：IEEE离散程序本身不具有通常的连续导数。本诊断用连续延拓，经过捕获的实际F32 mean/filtered-scale/quaternion/AMP值。mean直接变化；filtered-scale用原exp(2ell)+filter²的比值乘捕获scale；quaternion为捕获q加normalize(rawq)-normalize(base_rawq)；AMP为捕获AMP乘sigmoid/3D-filter补偿和屏幕补偿的基点比值。这些定义在基点严格保留既存物理根，包含原参数激活链与屏幕补偿的连续导数；不是声称原Float32字面程序的精确微分，也不把重新计算的理想幅值替换捕获AMP。

屏幕补偿定义取原GLM正文的连续矩阵公式：C=J A Sigma A^T J^T，J含原FoV裁剪，Gamma=.5(log(max(det C,1e-6F32))-log(max(det(C+kI),1e-6F32)))。使用实际F32常数kernel/focal/FoV limit。clamp或alpha cap恰好在边界者记不可微状态，不删行/称作通过。

实现比较：F64直接covariance逆的autograd；F64真实inverse-rotation因子的autograd，屏幕det以Gram叉积计算。数学目标相同。先在既存GPU内部候选、公共F32深度真实中心评价，再从原认证端点用实现自己的稳定signed-log二分细化（最多200步），参考根不作为该实现输入。根导数用IFT的-H_theta/H_t，公开z导数另除实际qnorm。scaled residual的scale detach，不能微分人工缩放；source AMP/energy前缀移位保留，舍入零不作为真根证据。

MP140独立world covariance inverse、解析矩阵链及MP240稳定残差/密度参考。独立MP根仅用于结果比较。SymPy核验R(q)四个导数及层partials；若梯度分母为0/非有限，明确记录未定义/不可表示状态。完整grad数组、各block误差、root误差、slope/条件数、状态、来源SHA保存。refined-factor门槛每条完整11参数VJP相对L2<=1e-5、每block<=1e-4；任何未通过保留，不取mean掩盖。public/candidate分支失败不删除，也不视作根证书失效。

此阶段是已观测开发源的数学/CPU连续延拓梯度诊断，不是完整CUDA训练后向、拓扑变化导数或GT/效率证明。下一步再验证GPU原参数接线与固定模型完整GT，不能仅凭本gate直接声称训练改进。
