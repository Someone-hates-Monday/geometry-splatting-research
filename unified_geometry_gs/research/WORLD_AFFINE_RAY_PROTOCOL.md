# 同24条射线的世界/affine机制检查

2026-10-13，写于本检查运行前；在已观测24点单法向结果后追加，属于探索性检查。
查询射线不更换，仍为原3模型每个8保存视角的第一条有限旧中位根。

1. 原world audit的13源和全部24点字节必须匹配。原PLY hash再次核对。
2. 从全部世界Gaussian构造精确射线限制的名义double系数：
   a=d^T P d，p=−d^T P(o−mu)/a，sigma=1/sqrt(a)，
   alpha=amplitude*exp(−||(o−mu)+pd||_P²/2)。垂直残差直接求和，避免平方差消去。
   不做tile、alpha阈值、center near-plane或排序prefix裁剪；underflow数量保留。
3. 逐项用平方根vacancy及近/远侧公式构造从−infinity开始的T_minus(t)。
   同时显式记录T_minus(0)。相机出发的透射率为T_camera(t)=T_minus(t)/T_minus(0)，t>=0。
   不能把有限相机起点与−infinity起点混为一谈。
4. 在旧affine中位深度，检查世界射线的logV与原world点态logV一致（1e−8余量），
   世界−dlogT/dt与原点态sum|d dot b_i|一致（1e−7相对、floor1e−12）。
   对所有24例保留失败，不调阈值、删点或改原point记录。
5. 总剩余光学厚度>log2则用brentq求相机中位根；从[0,max(1,2t_old)]扩展，最多32次。
   xtol=5e−13、rtol=1e−13；最终logT_camera残差<=1e−8，不能仅信任solver success。
   有限性不足/括区失败/非有限参数均明确保留，不重新定义样本。
6. 保存旧根处world T/V、原affine T/V gap、world根/位移/宽度尺度、root处world T/V gap、
   起点光学厚度与所有名义underflow计数。汇总全部24点与各scan。

这个比较同时改变全世界基元集合、near-plane/tile/alpha/prefix规则、.99 clamp和affine形状，
不能只归因为透视线性化。无GT与新视角配对，不解释为像素/重建错误或方法收益。
世界射线系数不是当前生产渲染器的替代实现，不修改生产文件或启动训练。
代码/参数/日志归档，逐点保存进度，异常标failed，成功标complete。
