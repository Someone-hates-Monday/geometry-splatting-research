# 完整24视角的贡献项拓扑与容量核验

2026-10-14，观察完整计数前固定。short_certificate_v1的局部根最多192层，完整图像不能静默截断。这一步在原affine renderer实际执行位置计数，不用复制的CPU/独立CUDA eligibility表达式代替。

## 固定范围与改动

原median_cpu_scan37/55/63各8视角、iteration30000、全部原分辨率像素。核验原PLY和输入来源；Scene只在scratch读原模型。用同进程同源新编control与instrumented两臂，保持原RGB、alpha、normal、深度/斜率、排序、支持域、初始范围和5轮求根。未实现认证完整renderer/backward/GT，不进入新训练。

instrumented只在RGB实际接受位置增加uint32计数，在median首次迭代实际eligibility通过位置增加计数，并记录rsigma<=0数量。新增mdepth通道2..5分别为RGB接受数、首次median接受数、其中非ball数、初始T<=MIN_TRANSMITTANCE标记。原通道0..1保持；P<=2^24确保计数转FP32精确。新增计数没有固定容量上限、截断或抽样。

## 核验和度量

所有24视角必须输出：两臂原RGB/alpha/normal/depth/slope/radii逐位一致（含完整图像），计数非负整数且不超过P；初始eligible像素的median count与RGB count是否一致全部报告，若不一致保留位置/值，不用推测回填。全部初始eligible像素与最终valid像素分开，保留原invalid depth=0。

保存每像素完整计数/初始eligible/最终valid/两臂深度斜率，不保存新输入图片副本。报告完整直方图、分位数、max，超过192/256/512的绝对数和比例，nonball覆盖，按每视角、每场景与全部像素汇总。门槛针对计数与原输出的可用性，不把某个容量通过称为认证渲染或GT改善。

预热每臂3次，5对交替顺序各1次完整render，CUDA event+同步wall，峰值allocated/reserved/added allocated显存；记录所有数据而不按结果删视角。计时只度量计数插桩成本，不是方法/训练加速。原control与instrumented均是native forward，绕过高层appearance后处理；两臂相同调用输入，不把RGB等价扩大到后处理质量。

单视图冒烟仅定位ABI/编译错误，主输出必须全部24；源码、generated、binary、输入/原PLY哈希与逐视图终态留档。出现观察超时继续同一handle；首次error保留，不修改原模型或掩盖失败。

下一项根据真实贡献项分布选择动态名单/缓存和并行布局。完整认证、unresolved政策、世界VJP、跨场景训练/留出图像/几何及新颖性仍未完成。
