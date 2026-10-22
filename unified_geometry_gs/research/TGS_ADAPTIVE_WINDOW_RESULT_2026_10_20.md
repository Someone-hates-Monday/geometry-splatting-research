# Adaptive window v3: expanded image validation and real-model diagnostic

本轮取得了可复查的数值修复进展，未训练、计时或修改原库/模型/LaTeX。
95案例扩展验证通过；真实两相机RGB/VJP比较也通过，但加权深度尾部差异尚待解释，暂不推广。
这不是图像GT质量或学术新颖性证明。

## 冻结与实现

`TGS_ADAPTIVE_WINDOW_PROTOCOL_V3.md` SHA256
`fbabfa7e3d989291da1e16c757b4c98fb2f13480da2536e568578b25ccaf70d2`；
input_plan SHA256 `e4e0bb5e4bde8de58ded812db577b0c3ff41ef45559fc0186eecb9cbe9eba5a7`。
保留v2的全部79案例和失败，再加入16例小sigma、左右位置/正反绕序的边界案例，
没有根据CUDA输出选择采样，所有整图像素均参加比较。

三份隔离模块共享新缓存布局，均直接积累Gamma，未移植contraction来混淆比较：
arithmetic64只提高旧float系数的像素求值精度；adaptive按log窗口界回退；oracle64始终使用
从原始投影构造的double窗口系数。oracle64的SH、合成和梯度累积仍为float，不能称为整个渲染器真值。

模块SHA：arithmetic64 `e30b44e99f26bdb5cbd0e6b2bf096f90d3250e9d556f50da50b177f1bbf9a433`；
adaptive `18292f28af5b5967c3c4420c6e83d850eb38d2c36a8f3eea823f2cc4419154d2`；
oracle64 `7390fb64b9bbe0384c10704956d6e95904c4d2cd20706b164b12e665062d261a`。

标准数学推导：|delta alpha|<=.99*sigma*|delta log u|，在uhat>du时
<=.99*sigma*du/(uhat-du)。GPU用有向舍入构造相对double缓存的边方程误差界，
不确定的inside/正u或预算>1e-6时回退，并在cutoff/cap附近另加经验power余量2e-6。
这不是对投影、libdevice power及所有合成分支的普遍IEEE证明，详见log bound note。

增加每triangle10double系数+10float误差界+3uint计数，即132字节payload，另有分配对齐。
计数器会影响开销，不允许对该诊断模块计时。初次生成器匹配过宽在CUDA编译前失败，
generated_partial_1保留；修正后会话48077编译成功，未覆盖旧v2证据。

## 95案例结果及独立审计

95份整图，每版本68583676个RGB像素；三个模块共1545组VJP。
adaptive/oracle64全部通过原定3e-5前向和1e-3VJP门槛，arithmetic64有12失败，全部保留。
独立重心/行列式参考重算全部像素、Torch64重算全部采样VJP，原NumPy复步长结果相符；
adaptive整图最大绝对差1.869081e-7。参测VJP各组最大相对差约1.242e-4，仍不能一概写成1e-6。
所有532条既有封存哈希检查及452原始源文件哈希检查通过。
1210个存储系数误差界用精确有理数逐个验证。

两个旧边界失败整图：alpha_saturated从v2 max .442398降到adaptive8.619e-8；
early_stop_skips_fourth从.403816降到1.217e-7（此处为原CPU64参考数值）。
只转double像素算术的arithmetic64仍分别有.00313180/.00532578误差，说明缓存系数也必须处理。

继承的5184条FD探针完整执行且tile成员、alpha支持、CPU分段分支稳定。
归一化FD误差中位7.1813e-5、90分位.0015421、99分位.0333891、最大1.36864；
不能把支持稳定叫做所有FD都达到1e-3，未删除小导数或不利步长。
新增极近边界VJP按CPU64/复步长核验，未虚构可表示且不跨分支的几何FD。

## 真实完整模型固定输入

公开stump7000，784056三角形、SH3、1245x825；既有train相机_DSC9258/_DSC9322，
没有优化器、GT评估或性能计时。真实计划在输出前写入，固定dense RGB上游同前成本测试。
模型sigma min .216805、中位.934690，464951/784056(约59.3%)小于1，未使用虚构训练数据。

adaptive/oracle64两相机所有六类输出梯度均有限，radii逐位相同。
RGB max difference2.95043e-5/2.41399e-6，MAE5.33325e-9/6.69419e-9，无像素RGB差>3e-5。
几何VJP相对差8.63237e-6/3.57641e-7，scalar/SH各组也在预设1e-3内。
这些是两个数值实现间的固定输入比较，不是对所有真实参数的独立解析梯度证明。

高精度回退19702570/178132471=11.0606%，37595950/352762044=10.6576%。
分母是所有前向window求值对（包括之后outside/低alpha被丢弃），不是唯一像素或有效梯度对。
两个几何缓冲各250117696字节；这不是完整显存峰值或速度收益。

独立整图检查还发现：每相机各1个weighted_depth像素差>3e-5，
9258像素(116,742)为9.0383844 vs9.0403042，max .0019197464；
9322像素(740,268)为3.7840006 vs3.7840660，max6.53267e-5。
median depth两图逐位相同，不能把问题误说成median分支变化。
adaptive分别少5/8次window求值；alpha在这两点差2.16365e-5/1.09673e-5。
这是停止/合成路径的诊断线索，但尚未用逐contributor轨迹隔离，未宣布单一因果解释。

## 推进决定

95案例扩展门槛及两真实相机声明的RGB/VJP门槛通过；生产推广和性能计时仍为false。
下一步锁定两个weighted_depth像素，从缓存系数、实际alpha与incoming/test T逐项追踪，
核验是否跨early-stop边界，再设计跨窗口和合成的误差传播/不确定射线重放。
已有log bound note给出transmittance误差递推和当前skip-current截断尾部上界；
实现与新留出验证尚未完成，不能写成已解决。保持全部95案例、FD、两真实整图及其尾部像素。
完成这一步后才移植同一可靠前向下的contraction并测实际成本，不能直接根据11%回退预测加速。
最终质量、效率、泛化、投稿价值和新颖性都尚未成立，整体目标active。

编译48077、95案例25362、FD98816、独立审计57988、真实诊断38206均exit0，无待等训练或编译。
证据目录experiments/tgs_adaptive_window_v3，包括输入、生成源、模块、整图、梯度和独立审计。
