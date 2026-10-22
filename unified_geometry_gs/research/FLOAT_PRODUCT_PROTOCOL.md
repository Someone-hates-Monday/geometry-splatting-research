# F32直接透射率充分条件：预注册诊断

2026-10-15。先固定此协议，再编译和记录结果。生产renderer/训练源码不改。

目的：检验原公共F32 median-z在固定真实世界Gaussian场中的误差能否由便宜端点区间认证，不把原候选当作正确根。来源为csr_decode_v1的实际query/AMP、csr_capture_v1全部CSR和原冻结模型。24视角全部像素；初始eligible但公共depth无效者独立计数，绝不称作数学无根。

每warp一像素，不限192节点。缓存低模式双精度有向几何正文保持；只替换alpha的exp为F32有向区间。每节点T：峰前sqrt(1-alpha*g)，峰后(1-alpha)/sqrt(1-alpha*g)，g=exp(-((t-p)rho)^2/2)。分支不确定取两个区间的包络；每节点及最终乘积限制在数学范围[0,1]。所有节点参与lane内顺序、warp树乘积。不确定/非法值传播[0,1]和失败标记，不能用舍入零确认符号。

exp仅非正输入，ln2邻接F32上下界；x>=-80用range reduction、degree8 Taylor和3*r^9/9!余项，r<=1才接受；x<-80用[0,2^-115]，由115*ln2_upper<80保证。所有1/j使用精确有理数邻接F32包络。x=-inf允许[0,0]，x=0为1。FTZ=false、FMA关闭，strict div/sqrt，不使用fast_math或标准exp的经验ULP界。数学论证依赖CUDA有向IEEE算术/编译器契约，不是形式化证明。

公共z的真实射线位置c=z*sqrt(qx^2+qy^2+1)用double有向区间。tau=0x1.0624dd2f1a9fbp-10严格小于1/1000；lo=round_up(c.hi-tau),hi=round_down(c.lo+tau)。只在lo<hi、所有输入有效、T(lo).lo>0.5且T(hi).hi<0.5时认证。连续严格递减Gaussian T保证唯一根在端点间，距真实c小于tau。不满足输出unresolved，后续仍需strict求根回退；此阶段不实现或声称完成该回退/梯度/训练。

输出每像素12double：code(1认证/0未认证/2无原候选)、lo、hi、center_lo/hi、Tlo_lo/hi、Thi_lo/hi、节点数、bad标记、公共z。无候选endpoint字段NaN。记录全部状态与多次CUDA事件，输出数组永久保存。不是完整renderer计时；已有cache/CSR/query成本单列，原renderer时间来自已冻结协议，不直接声称端到端加速。

独立探针：每view64个等距初始eligible像素（选择与认证结果无关）及原9异常像素，保留每条完整射线的源36字段和低模式系数包络。MP80/140独立世界逆矩阵计算全部节点p/rho/E/alpha与两端T，检查包络及认证严格符号。无候选也保留。scalar probe固定边界、邻接F32、等距[-80,0]与随机种子13017，包括下溢。若任何错误，保留失败批并停止采纳。24视角是开发集合；不构成GT/留出训练/新颖性证据。

官方算术依据：https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html 。失败与统计全保留，成功充分条件覆盖率不能推断未认证根错误。
