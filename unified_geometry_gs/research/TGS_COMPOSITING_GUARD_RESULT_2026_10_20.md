# Compositing guard v4: causal trace, implementation and expanded validation

本轮从两个真实深度异常像素得到逐项因果证据，并完成隔离的透射率分类版本。
所有本轮冻结范围内门槛通过，未训练、计时、更新模型或改动原库/LaTeX。
数值一致性进展不等于图像GT质量、加速或学术新颖性成立。

## 原因已验证

trace_v1在CUDA上使用原v3窗口函数，读取真实排序list和cache，四次原生端点逐位重现。
9258(116,742)在第338项发生early-stop分歧，adaptive停于338、oracle停于343；
9322(740,268)在第760项分歧，分别停于760/768。原有median项均相同。
省略项是triangle120791(z88.77187)与199338(z5.94180)，测试透射率位于1e-4两侧。
特别是第二像素当前alpha逐位相同，差别来自累计incoming T，不能只修当前窗口。
额外oracle深度权重、公共前缀差异、浮点累积残差在独立trace审计中分别保留。
详见 `TGS_COMPOSITING_INTERVAL_NOTE.md`，标准区间推导和硬停止不连续反例均注明假设。

## 实现与协议

冻结协议SHA256 `b71d2d71a7f94f737385c1effee7587f4d6a7666999a93d0a743432adbffeaf8`。
adaptive模块SHA `252b4826cce6fa69c538bd5245950480bca374cede8d71de5b76000d0752fe7d`；
oracle64模块SHA `2d1e3c9c2cdf71a731c7984cbe27e8c70852d4c99970afd17dc05b5bb18c7764`。

分类阶段传播outward T区间，若跨incoming .5或test T1e-4则标记整条射线；最终阶段使用
该标记选择double窗口路径，并供backward使用。保留skip-current停止语义和原阈值，未更改median定义。
分类阶段不写maxblend/图像，避免被否决的临时路径留下全局max贡献。
double边方程加法显式舍入；窗口power余量仍经验性，投影和tile guard也没有普遍IEEE证明。

代码检查发现最初16个新目标构造在float32分母上加EPS；冻结的111输入未覆盖，
在任何v4 native输出前追加16个配对的float64分母修正版。原数据、问题说明和addendum均保留。
扩展计划SHA `ccc30fee028f051bc29fc08cda8b09a778645fc684879c0334794064bf2e21df`，共127例。

## 固定案例与独立检查

原95输入全部保留，另32例提前终止/median阈值附近输入。每版本70680828个整图RGB像素，
两个版本共1286组VJP；逐像素contributor/median ID和radii一致，RGB/aux/maxblend及预设VJP门槛通过。
原95的独立行列式/重心坐标参考max RGB误差1.869081e-7；参测geometry VJP最大相对差约7.515e-5。
5184原FD探针原封不动执行，tile/alpha/CPU分支检查通过；FD数值误差中位7.1813e-5、
90分位.001542、99分位.033389、最大1.36864，不能宣称所有差分精度1e-3。

新阈值病例按预先声明的native oracle路径比较，CPU64分歧仍报告：4例median病例各2像素
CPU64/native median差2。RGB max差仅约8e-8。这说明离散分位数可放大极小阈值偏差，
v4没有恢复任意精度实数阈值决策，不得把native间一致写成CPU64所有辅助图通过。
原95的v3/v4 oracle RGB整图逐位相同；全部历史1087封存哈希及452源哈希检查通过。

## 真实模型结果

同一公开stump7000，784056三角形、SH3、1245x825，两train相机；RGB与weighted-depth
两种固定dense上游均执行。六类返回梯度全部有限，两个版本的RGB/深度VJP均在1e-3门槛内。
最大几何相对差：9258 RGB8.660e-6、depth8.090e-6；9322 RGB4.119e-7、depth6.283e-7。

9258/9322分别标记147/161条射线，占.01431%/.01567%；其中median84/120、early-stop63/41。
所有标记射线的RGB/aux/contributor端点与oracle逐位相同，整图contributor与median ID也逐位相同。
两个原异常像素的weighted depth现在逐位一致，flag均为early-stop类型2。
整图RGB max差均2.384186e-7；weighted depth max差降至1.525879e-5/5.722046e-6；
其余辅助通道max均<4.18e-7，median depth差0，maxblend差3.576279e-7。
完整真实图的v3/v4 oracle RGB、全部aux与radii逐位相同，舍入显式化未改变这些参考输出。

## 成本限制与下一步

少量射线被标记不代表只有少量新增工作。当前分类与最终渲染都遍历全图，
两相机window计数比合计约1.999884/1.999930；这是求值数比，不是时间比。
新增payload为144B/triangle，较v3多12B/triangle；另加4B/pixel标记，含对齐实际
geometry buffer259526464B、image buffer32868308B；不是完整显存峰值。
带counter诊断版本不计时，未证明本方法快于oracle或原生实现。

下一步应冻结无counter成本比较，先完成classification/final helper的phase一致性审查，
再剥离计数器测试正确性和native调用延迟；并研究更紧的alpha误差界或有界射线贡献暂存，
减少重复求值。所有优化须保留127整图、5184FD、真实两相机与阈值反例，并加入独立留出相机。
不能直接将v1的Gamma收缩套到新的高精度窗口而假设消去误差仍相同；这需要新的验证。
质量、训练收益、跨场景泛化、新颖性与最终学术目标仍未完成，整体目标active。

trace72907、guard编译5825、127案例98918、真实79321、FD18397、独立审计48572均exit0，无等待作业。
证据目录experiments/tgs_compositing_trace_v1和experiments/tgs_compositing_guard_v4。
