# fast_math下标准power调度：固定探针可用，尚非渲染器修复

实验目录experiments/tgs_standard_pow_dispatch_v1，2026-10-20。
v8的14个weighted-depth失败像素在标准powf回放中回到门槛内，但若直接在
--use_fast_math构建里把__powf写成powf，编译器仍会映射到fast函数。

本轮用显式experimental extern C device声明调用文档中的低层libdevice __nv_powf，
保留全局--use_fast_math/ftz=true，独立编译sm120；未改渲染器或模型。
这不是声称SDK头文件提供了该C++声明，也不保证别的工具链/架构可移植；这里只记录实际验证。
[CUDA12.8 libdevice文档](https://docs.nvidia.com/cuda/archive/12.8.0/libdevice-users-guide/__nv_powf.html)
提供其低层原型，精度说明仍指向先前已注明“非保证界”的函数表。

冻结4007个(u,sigma)：1176个v6网格，另2831个失败像素trace到达pair的输入，全部保留。
结果：named powf与__powf逐位一致；显式__nv_powf与fast在765项不同。
1106个“输入和输出均正常数”样本与已有标准函数参考逐位一致；2831个trace输入也全一致。
全数据仍有132项与严格标准参考不同，来自FTZ边界，保留所有差异；不能说低层调用
在fast_math下恢复了无条件IEEE行为或所有次正规输入。
SASS、输入、输出、原始参考来源hash、模块和独立审计已保存。

构建时cpp_extension早于CUDA_HOME赋值导入，实际使用/usr/local/cuda默认路径。
后续只读readlink和nvcc核验确认其指向/usr/local/cuda-12.8，版本12.8.93；
compiler_provenance.json保存编译器SHA及版本，未来构建须先设置环境再导入cpp_extension。
没有因这个路径问题重跑挑选结果。

原探针9399和独立审计均exit0，未训练、计时、评估GT或证明严格power界。
这只使“替换power并保留其它fastmath算术”成为可实施候选；v8整体精度仍失败，禁止性能推广。
