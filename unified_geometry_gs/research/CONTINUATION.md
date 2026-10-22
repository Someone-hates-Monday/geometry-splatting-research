# 继续研究时先读这里

## 当前入口（2026-10-21，v15隔离窗口与三像素排序机制完成封存）

先读 `TGS_WINDOW_SCENE_RESULT_2026_10_21.md`、`TGS_WINDOW_SCENE_PROTOCOL_V15.md`、trace附录与两份数学note。
上一轮v14及本轮v15均为实质进展，总goal仍active；没有新方法质量/效率或新颖性结论。
当前仍仅数学/小规模固定输入CUDA，未授权完整训练。本轮无optimizer/backward/benchmark。
复用v14全部32离线事件-真实捕获相机组合，不是新留出/捕获训练事件。13零case全部保留。
prepare exit0、isolated render exec1edbc1 exit0=64次tiny native前向；原模块9d974，白色预计算，无SH/新渲染器改动。
parent/child radii逐项复现v14，white RGB三通道literal，white-alpha<=2e-6，全图补丁重建hash通过。
CPU audit ea80b7 exit0：flowers isolatedalpha max.744209，room .436186；sceneG/isolatedalpha RMS中位数.0339351/.0151705。
此比混合颜色/透射等，不是遮挡百分比。离散pixel alpha sum变化不当连续全域积分反例。
固定槽位条件界T(c-B)(Q-P)是标准合成身份；passing不是同序证明，违反也需trace。empirical余量3e-5非FP32证书。
room/train case(event0,parent1526173)像素(1294,852)，(event5,parent59501)像素(455,198),(454,201)全部3例超界。
freeze_tgs_window_violations_v15 exit0保存全部3例及源hash，未选效果好的子集。其余30case过界不排除排序效应。
附录TGS_WINDOW_TRACE_ADDENDUM_V15.md先冻结额外3 full recaptures+18 tiny observer calls，无训练扩张。
trace prepare exit0、run19023 exit0；只编译新隔离CUDA observer，O3/use_fast_math sm120 gcc11，原生产模块未动。
3 full native recaptures全RGB/7aux/radii/maxblend/path hash逐位v14；6前后pixel端点RGB/alpha/last ordinal逐位原native。
packet从原tile sorted list与visible缓存提取，precomputed色从输入取，不读未写geom.rgb；所有external packets/order在canonical ID映射后literal同。
全3pixel无early stop，禁用stop端点literal不变；移动child回parent slot时child alpha不变。
actual peak→fixedgroup peak：.000512779→.0000113249；.005635105→.0000507683；.005895190→.000109889。
ratio .0220853/.00900928/.0186404（局部扰动减少97.8–99.1%，不是GT质量/全图收益）。
CPU exact audit552de9 exit0，3通道有理分解window/group、order、rounding严格恒等，1698个T RN检查；max rounding1.18e-7/2.71e-8/2.11e-8。
每pixel1个activechild分别cross8/19/15个active external候选；analyse c9feba exit0，exact centroid/swap identity通过。
数学note：同平面细分射线交点深度不变，corner centroid=g+(v_i-g)/2、中央=g；中心key可能cross；只充分no-cross margin非新算法。
2DTS v3已有ray depth/local resort、ETS已有kernel/filter；本轮primary联网核读和完整检索记录literature/window_scene_prior_work_v15.json，不能当新颖性认证。
所有图/CSV已生成并view_image目视通过，含32case和全部3mechanism pixels，明确零/条件界/事后限制。
主seal exit0：179files 30169756B 23source/dependencies，SHAa1354ac425e2a01c888d778b2b5a0481a1d518095b8d9f69f507953e11864e72。
主filename glob遗漏window_trace审计器和RNhelper，未改主封存：补充dependency_manifest_addendum.json另封3files 12399B。
补充sealer exit0，addendumSHAe771813309678bc6cd3e7281f585758d52590a1ea38eb96d7ba6271f2b676879，主清单/结果未变。
OUT=experiments/tgs_window_scene_v15，所有本轮作业terminal exit0，无待等待计算，不重跑/覆盖封存源/协议/数据/报告。
TeX仍161263B lastwrite2026-10-16未改；原模型/生产源码未改，未重hash整个v14树，只核验manifest和明确使用的旧文件。
下一项：全图实际射线深度排序基线与GT质量-成本对照，区分物理排序、parent-slot稳定诊断、原中心排序，必须以既有排序为直接先例。
3像素机制不能升级为全图主因，局部保持不能升级为更好训练。若全图质量/成本意义不足，停止TGS局部变体循环，转回GGGS理论及真实任务缺口。
旧train_maximin4、跨距离SH、GGGS30k负结果保持；总体科学目标尚未完成。

## 当前入口（2026-10-21，v14真实相机细分颜色分解已完成封存）

先读 `TGS_SPLIT_COLOR_RESULT_2026_10_21.md` 与 `TGS_SPLIT_COLOR_PROTOCOL_V14.md`。
上一轮v13是实质进展；本轮亦有决定策略的新证据。总体goal active，未有完整学术/方法成果。
范围仍数学/小规模固定输入CUDA，不含完整训练。本轮76前向、无backward/optimizer/benchmark/新编译/下载。
两公开7000模型flowers938101tri、room2100000tri，复用v10原捕获test/train各1相机；不是训练事件。
父选择opacity>=.01、任一相机radii>24，预先SHA顺序8/scene；候选65543/318571，未按split结果选样。
16单父中点四分事件*2captured cameras=32cases；各child复制sigma/opacity/SH，其它参数不变。
预计算物理中心p颜色（非legacy q），同一changed geometry上H继承parent颜色、M按child中心算色。
B原几何预计算baseline；G=H−B含所有非颜色变化，A=M−H纯颜色干预，D=M−B=G+A。
G尚不能分成纯window/culling/sorting因果项；所有平方能量保留signed2<G,A>，A/G是范数比非方差占比。
prepare exit0、render22346 exit0 exactly76calls；4baselinesrepeat literal，nativeSH-only maxRGB2.384e-7flowers/1.788e-7room，noncolor literal。
32H/M all7aux/fullradii/maxblend/contributors/tile literal同，model/camera inputs unchanged。
原CPU audit exec8d5c09 exit1在float(size-one np array)；不是gate失败；原源码/plan/trace保留。
v2driver只修复sigma/opacity .item()、新输出independent_audit_v2.json；repair plan beforeaudit，不重跑CUDA，不改门槛。
v2audit9689 exit0：全32fullarrays由patch+baseline重建并核对hash，midpoint/params/ID独立核验，SciPy SH颜色max1.950826e-7。
RGB能量identity maxresidual1.323489e-23；不是独立GPU重放/同行复核。
flowers16cases G0=6，room16 G0=7，全部13零情况保留且A也0；其余19 A/G最大.0893672。
有效A/G中位数flowers .0210259=2.10%、room .00322745=.323%；appearance-material门槛两场景均0/16。
Apeak>3e-5 flowers7/room2，Gpeak>3e-5 flowers10/room9；A最大fullMAE7.40461e-7/2.32911e-9。
继承父色反而使相对B总RMS稍大9cases（flowers6/room3）；没有GT评分，不是质量优劣结论。
这19非零case parent及四child均存活；13零case parent/allchildren均radii0，没有当前批次全子剔除witness。
不能据此前stump单例把本批主变化归为size culling；逐像素cutoff/window/排序仍未分离。
图split_components.png/pdf、32case CSV已生成并目视检查，零为真零，统一symlog/linear1e-9，全部案例保留。
sealer7291 exit0：217files 647155441B 18source/dependencies；manifestSHA8b7736cf7e650cc960a23c19d3660f1db0fde4ab51480321404465ee9aed5411。
OUT=experiments/tgs_split_color_v14。全部作业terminal，无待等待任务；勿重跑/覆盖封存源、协议、数据、报告。
TeX仍161263B lastwrite2026-10-16未改，原公开模型及生产训练源码未改。
决策：降低外观运输主线优先级，不扩大网络/SH阶数/训练。数学困难不等于真实任务值得优化。
下一项应分离单父/子窗口响应与完整场景合成/排序，在当前四子均存活的情况中找量级与机制，先对照已有窗口先例。
保留train_maximin4、跨距离SH、GGGS30k负结果，不重新包装/改门槛。若窗口/排序也无实际质量/成本意义，转回GGGS理论与真实任务缺口。
不要继续仅堆局部恒等式/数值诊断而宣称方法收益。任何完整训练仍不自动授权。

## 当前入口（2026-10-21，v13外观迁移数学/真实系数/距离压力已完成封存）

优先读 `TGS_APPEARANCE_TRANSPORT_RESULT_2026_10_21.md`、`TGS_SH_SHELL_THEORY_V13.md`、`TGS_APPEARANCE_PRIOR_ART_V13.md`。
上一轮v12为实质进展；v13亦已得到改变后续方向的证据，总goal仍active，没有完整论文/方法收益结论。
当前范围数学/小规模固定输入诊断，未授权完整训练；本轮全CPU，无新CUDA/optimizer/训练。
先例核读确认Scaffold-GS anchor方向距离、MCMC近似保持重定位、SteepGS理论等直接重叠。
检索logs保存于literature/appearance_transport_search_v13*.json；HF paper MCP UNAVAILABLE，部分PDF403/过大，使用arXiv原文。
这是单agent定向叙述性核读，不是完整数据库系统综述或无先例证明。skills/流程出处已在报告标注并联网核对。
新数学：固定半径完整球面仍有有限SH阶数障碍；精确degree1全谱b_n、L2最佳误差及投影uniform误差。
degree3→4泄漏B4=−12a(217a^4−364a²+143)/1001，一阶−12a/7；不能用degree1误差阶数推广真实SH3。
SymPy1.14+60位独立积分五个a均通过；另有事后跨半径过补偿定量解释，完整书面假设见theory。
CPU main probe v1 exec901357 exit1：requires_grad Tensor漏detach，尚无目标场输出，flowers ID已存。
保留v1脚本/plan/selection，v2新目录real_fields_v2只修复detach并核对原flowers IDs；不改样本/门槛。
v2 prepare exit0，probe75783 exit0，indep audit38492 exit0。两真实公开7000模型各64组SH3，构造球面与位移。
1536cases=2scenes*64primitives*3axes*4a；不是原捕获视图或真实训练密化事件，不作图像/几何收益声明。
degree3投影/复制组中位数比.454–.568；degree4/degre3 .0117–.2946，25vs16系数多56.25%存储。
全部真实checkpoint rows/ID/独立SciPy基函数/完整误差复核，max disagreement4.44e-16，双拟合网格max系数差约3.45e-14。
原距离压力190621 exit0，4608cases，固定系数不重拟合；是在观察主结果后冻结的事后敏感性，不是新独立留出。
原stress零复制误差分母产生Infinity/NaN汇总，原始逐case RMS全部finite；不使用原无效比值。
新audit_tgs_sh_radius_v13.py独立多项式SH复算所有4608原始RMS，exit0，max2.78e-17；strict JSON记录近零分母计数。
0.5R两投影无更差；2R三阶914/1536、四阶99/1536更差；4R二者全部1416非近零case更差，另120近零case保留。
4R组中位数误差比L3 2.537–2.721、L4 2.936–3.000，与小位移L4极限|k−1|一致；事后解释不当确认预测。
图distance_transfer.png/pdf及全case/summary CSV均已生成，所有4个位移保留、统一log2半径尺度、目视检查通过。
sealer exit0：66files 7316619B 17source/dependencies；manifestSHA43744a895b2a8eab4b983c7eb29a86afb9a312f10aad3bcce0a313381e586c05。
OUT=experiments/tgs_sh_shell_v13，全部作业terminal，无待等待计算；勿重跑或覆盖封存v13。
原PT输入readonly hash/抽样逐行核验，未重新哈希完整旧v12树，只检查v12manifest hash。TeX仍161263B lastwrite2026-10-16未改。
下一步：实际捕获相机的真实几何细分中分离alpha/coverage与SH中心变化误差，先判定颜色分量是否值得优化。
再对比显式半径条件运输、精确保留父外观中心与已有共享外观，核对直接先例/成本/链式导数。
固定单半径无条件系数转换不作为通用方案推进；不能为了新方法名把已知anchor/标准投影当创新。

## 当前入口（2026-10-21，v12 SH隔离、数学核验、审计与封存均完成）

优先读 `TGS_SH_ISOLATION_RESULT_2026_10_21.md` 和 `TGS_SH_CENTER_MATH_V12.md`。
仍仅授权数学和小规模固定输入CUDA；没有optimizer/完整训练授权。总体goal active，尚无充分方法/学术成果。
renderer63566、analysis18038、independent audit22787、forward-step45666全部exit0；不要重跑。
36前向=20主+16负对照。四视图原图/GT decoded RGB exact，全部隔离gates通过。
独立核验20主评分、9720 SSIM patch/channel；SH-only到geometry-fixed到piecewise五个保存前向字段全四图literal同。
SH-only相对原版PNG PSNR：flowers旧−.04288284065、新−.16199948371；room旧−1.17775919775、新−2.14525049799dB。
两fresh视图为输出前SHA规则选定；旧两图只能事后解释。所有非颜色输出和两种负对照逐位相同。
因此支持这四个冻结输入上的SH constructor软件干预效应；不证明实际training commit/重训练效果/GGGS失败原因。
SymPy exact反例、.0324866482两样本单通道最优uniform误差已核验；不是实际PSNR界或原创性证据。
梯度/curl仅孤立假定形式；尚未运行隔离颜色路径的CUDA反向核验。
图v1/v2保留排版问题；v3仅修复标签和横轴刻度，已view_image目视检查，作为交付图。
v2观察信息丢失后先查文件/pgrep，确认完成，未重复GPU。v3绘图exit0。
sealer8186 exit0：353files、3493857179B、28source/dependencies。
manifestSHA=91e0ddc12bd940eb774e3c3d2fea1c436e215452799503bcfcbcea8c9c166b3c。
本次核验新文件/明确只读依赖及v10/v11 manifest SHA，不声称重新哈希了全部19GB旧v10树。
所有v12本轮作业terminal；没有尚待等待的训练/渲染/绘图/封存任务。打开TeX未改：161263B，lastwrite2026-10-16。
后续新阶段新版本目录，勿覆盖封存v12协议、代码、数据、评分、报告。
下一步先一手文献核对appearance-anchor/shared-appearance/densification，再冻结区别于既有方法的外观保持/误差控制问题。
可考虑区分几何中心与外观中心，但目前只是假设：未实现、未证明收益、未建立新颖性，不把命名当创新。
下阶段仍先数学/局部运行时导数和固定输入验证，不自动扩大完整训练或重复旧计时。

## 历史执行中快照（已由上方v12完成记录取代，勿据此重启）

先读TGS_SH_ISOLATION_PROTOCOL_V12.md、TGS_SH_CENTER_MATH_V12.md；旧v11已封存勿覆盖。
prepare_tgs_sh_isolation_v12.py已exit0：forward/backward源与原版仅各1处constructor替换，编译receipt关键flags/commits一致。
4个视图：旧posthoc flowers_DSC9128/index11、roomDSCF4915/index31；预先SHA新视图flowers_DSC9136/index12、roomDSCF4755/index11。
新SOURCE=sources/tgs_sh_isolation_v12，新OUT=experiments/tgs_sh_isolation_v12；旧照片只读引用。
extract6811 exit0，4新图payload加metadata7009584B；全部4组old/fresh原图与GT精确RGB同，identity审计exit0。
SymPy1.14数学核验exit0：具体degree1色场有两相机同新方向、异旧颜色，色差.0649733，最佳两样本uniform色误差下界.03248665。
这不是PSNR界或实际质量因果证明；isolated assumed color-gradient Jq/curl结论不等于完整native梯度/训练结论。
execution_plan.json已在新GPU/quality前冻结原、新脚本和v11度量哈希。
render_tgs_sh_isolation_v12.py会话63566运行中，继续同句柄；至多20主+16负对照前向，无反向/优化器/训练/计时。
主5版本original9d974、SH-only4fa625、geometry955、piecewisef06、coarse81d18；primary pair SH3只准RGB变化，所有非颜色输出需literal同。
负对照degree0与constant precomputed RGB(.25,.5,.75)，original/SH-only完整前向需literal同；不作竞争模型质量评分。
先确认renderer终态，再分析、独立audit及报告/封存；若隔离gate失败，不归因或删相机。

## 当前入口（2026-10-21，v11质量结果已封存，下一步单因素SH兼容性诊断）

上一轮v10实质进展且已封存698文件19.2GB；本轮仍仅数学/固定输入CUDA，不含optimizer/完整训练。
先读TGS_GT_IDENTITY_QUALITY_PROTOCOL_V11.md及TGS_GT_CONTROL_ADDENDUM_V11.md。
固定旧v10两test视图：flowers_DSC9128/index11和roomDSCF4915/index31，不能按GT分数改相机。
官方Mip-NeRF360两ZIP目录catalogue会话32803 exit0，metadata919173B，不取整包；URL/HEAD/ETag/目录已存。
extract会话4527 exit0：两个原始JPEG+两个公开TGS GT PNG，四成员CRC/SHA通过，总Range缓存7370982B。
新sources/tgs_gt_identity_v11；新experiments/tgs_gt_quality_v11，旧v10不修改。
GT identity审计已exit0：两张原始JPEG解码RGB分别与GT00011/00031逐字节相同，0 channel differences，无任何几何/颜色调整。
质量候选固定4个只读SO：v9coarse、v5oracle、历史piecewise f06...、额外archived_native9d974...（未改源码但不是已证明实际training commit）。
4th对照在GT payload和分数前单独addendum冻结；用于已发布权重兼容性，不把不同数值/修正归为纯精度因果效应。
metric_selfcheck已exit0，解析常量图及PNG量化独立Torch检查通过；quality_plan.json已在首个渲染/评分前冻结源码哈希。
render58195 exit0，仅8次前向，输入hash不变；oracle/coarse新RGB/aux/path等逐字节复现旧v10。
analysis38457、independent audit22929均exit0；整数精确主MSE、Torch量化、全SSIM逆序滤波及3888直接图块通道核验通过。
主PNG8 PSNR：flowers oracle/coarse20.096741856，piecewise20.096152905，archived_native20.139035745。
room oracle/coarse29.690647251，piecewise29.693259383，archived_native30.871018581。
coarse-vs-piecewise仅+.00058895/- .00261213dB，SSIM两图都略降；不是抵偿3.9–5.0倍成本的质量成果。
coarse-vs-archived_native为−.04229389/−1.18037133dB；这是固定公开权重兼容性现象，不证明修正后重训练会失败。
必须降低只追native64精确参考速度的优先级。下一步定位具体代码修正造成的兼容性变化，而非再跑更多同类计时。
优先核验SH中心constructor别名修正（原C++cast逗号表达式）这一候选，需独立冻结消融/新增相机，不从四方法混合差异直接归因。
主PNG量化MSE/PSNR、次clippedfloat，SSIM固定11x11/sigma1.5/zero padding5；必须保留四方法所有结果。
绘图先前exec cell464在中断后不存在、pgrep无进程、无输出；确认后恢复CPU绘图，已exit0并视觉检查通过。
RESULT文件TGS_GT_QUALITY_RESULT_2026_10_21.md已写；seal会话30530 exit0。
v11封存164文件1038983743B，22个源/依赖文件，manifestSHA2b986d9073c4eb61b09670052a08b45b81827666e901b854d1931923ec9adc1d。
全体本轮任务已terminal；不要重跑或覆盖v11原图/渲染/评分/脚本。新阶段新目录。旧v9/v10及打开TeX未改。
下一阶段v12计划原native vs仅SH构造修正的source-only隔离，degree0与precomputed-colour负对照，固定旧2视图+预先SHA选新2视图。
数学问题：alias中心q=(p_z,p_z,p_z)换为p时，是否存在能对所有相机保留旧颜色的方向SH系数变换；需限定条件和反例，不当实际PSNR因果证明。
总体学术/方法目标仍未完成；完整训练未授权，不因小规模诊断通过就重启训练。

## 当前入口（2026-10-20，v10跨场景精度/成本通过并封存）

上一轮v9为实质进展，已封存，所有v9作业exit；总goal仍active。
本轮继续仅数学和固定输入小规模CUDA，不授权optimizer/完整训练。
先读TGS_CROSS_SCENE_RESULT_2026_10_20.md及AUDIT_REPAIR说明，再读两份冻结协议。
按预先listing最小transfer规则选择新室外flowers7000和室内room7000，已全部完成验证，勿重复获取/渲染/计时。
不是按输出挑场景，不换成较小room30000。官方public full_eval.zip只HTTP Range提取10个成员；650MiB上限。
extract会话71236已exit1：九成员已CRC+SHA完成，最后room PNG ZIP header发生RemoteDisconnected。
不是观察超时，已写acquisition_attempts.json。会话4335已exit0，原脚本跳过九完成成员补取了缺失PNG。
acquisition_audit已通过：全部10成员、总成功Range payload608581345B（非仅末次2086290B）。
CPU输入审计15753已完成并打印passed，967个v9封存哈希未变。flowers938101tri/1256x828，room2100000tri/1557x1038。
flowers相机test_DSC9128/train_DSC9169；room相机testDSCF4915/trainDSCF4697，均渲染前SHA选定。
CPU审计15753及GPU probe26656均exit0；全部4图初步gate true，完整VJP已保存npy。
独立precision审计5073 exit1：flowers train的不可见anchor原样重捕获比较失败，原脚本/失败/部分progress保留。
诊断65143 exit0：四图全部重捕获，仅flowers train25910个radii0的未初始化anchor不同；所有定义输出及可见缓存逐位同。
源码forward预处理可在anchor赋值前return；binning/backward只有radii>0使用。新v2审计仅anchors/rects按该mask，其他全查。
v2审计81110 exit0，四图full VJP/定义域缓存/独立oracle重捕获均通过；每图Fraction1632边，共6528。
audit_repair_plan.json保留新脚本及原失败/诊断hash。所有补充literal byte审计也通过。
不是更改模型/门槛/相机，v2progress独立新文件；TGS_CROSS_SCENE_AUDIT_REPAIR_V10.md说明。
v9 literal byte47494、v10 literal byte17152、v9 FD signed-zero sample审计全部exit0且zero differences。
发现普通array_equal合并正负零，新增实际字节审计（不改旧封存源），literal_bit_audit_plan.json预先冻结。
成本前已完成v10 literal byte审计88字段、v9 literal2948字段，及v9FD5184数组actualbit审计；均zero differences。
benchmark_tgs_cross_scene_v10.py会话8172已exit0：216pair、72blocks全部完成，输入hash前后相同。
分析首次提交因自动审批review额度限制未执行；实时get_usage_limits随后显示ordinaryUsageAllowed=true。
沿同一require_escalated审批路径仅重试分析一次，现已获准且exit0；没有切换执行通道或重跑计时。
独立成本审计已exit0：四图coarse/oracle配对中位数0.732411/0.754310/0.803530/0.783554，20k bootstrap已独立复算。
flowers省26.76%/24.57%，room省19.65%/21.64%；每相机median<=.90且upper<1工程门槛均通过。
历史不同数值piecewise仍快：coarse/piecewise比5.008/4.812/3.876/4.673。Torch峰值flowers增加15009792B、room增加33600000B。
被拒绝的functions.exec在首个exec_command即抛错，原计划后续CONTINUATION patch也未执行；本行记录已补。
科研图paired_cost.png/pdf已视觉检查通过，完整RESULT报告已写。sealer42110 exit0。
v10封存698文件19203065844B，39源/依赖文件，manifestSHA9d8b5e537c4110881a6ea1696c2c1f41a93ec77cb8ab4e5eba4006c5095ee463。
所有本轮作业已terminal，无待等待训练/GPU/审批。不要覆盖v10 source/scripts/data/reports；后续新版本新目录。
v10成本协议/benchmark/analysis/independent audit已提前写好，cost_analysis_plan.json已冻结。
全4图+独立precision及literal byte审计已在216pair计时前通过；v9是不同session，不混合或套用旧比例。
flowers实际PNG1256x828，而COLMAP width5025/4不可强取ceil1257；已按官方loadCam图像尺寸逻辑修正CPU输入审计。
这是渲染前输入修正，不改camera选择或精度门槛；TGS_CROSS_SCENE_INPUT_NOTE_V10.md有源码证据。
新sources/tgs_cross_scene_v10，不能重跑/覆盖.partial或existing manifest。CRC、Content-Range、SHA256核验。
原remote_tgs_zip helper只读导入，新range缓存位于新sources目录，不覆盖旧缓存或v9。
已按预先SHA选择每场景1test+1train，共4张整图，readonly v9coarse及v5oracle+各counter twins，RGB/depth完整VJP。
同门槛：forward逐位oracle；6组VJP有限且relative<=1e-5/tinyabs<=1e-6；全四相机pass后才可另冻计时协议。
GT PNG仅用于分辨率，映射未证实，不作PSNR/quality claims。下一轮优先做原图像素匹配+源码顺序核对GT身份，再冻质量比较。
本轮已只读核验Mip-NeRF360官方项目页https://jonbarron.info/mipnerf360/，包含官方Dataset Pt.1/Pt.2链接。
下一步可从官方原图有限Range提取所选images_4/_DSC9128及images_2/DSCF4915，与公开GT像素严格匹配，不能只靠test索引猜测。
目前尚未提取这些原图，不声称对应已证实。先验证链接/ZIP元数据、冻结有界获取及质量比较计划，再执行。
若质量不足以抵偿相对piecewise数倍成本，应降低全精确窗口路线优先级，保留其作诊断基线；目标仍是学术/实际价值而非仅精确参考加速。

## 当前入口（2026-10-20，v9精度/成本完成，单遍候选正向工程结果）

最新授权仅数学核验及固定输入小规模CUDA诊断；没有optimizer/完整训练授权。
v8及standard_pow_dispatch_v1已封存，勿覆盖。新证据 experiments/tgs_window_candidates_v9。
先读 TGS_WINDOW_CANDIDATES_RESULT_2026_10_20.md、TGS_COARSE_REJECTION_NOTE_V9.md，再读两份冻结协议。
新coarse=14float缓存、单遍可证粗排除、其余原生native64 window/alpha；standard=v8仅显式__nv_powf替换。
四模块构建41730 exit0，127案例94363 exit0（3858 VJP），binary82927 exit0。
real71510 exit0：七相机全部coarse前向RGB/7aux/maxblend/path/radii逐位oracle，最大VJP相对1.060913e-6。
standard在9213加权depth最大3.4332275390625e-5>3e-5，失败不可放宽；另外六相机pass。
相机旧六9258/9322/9269/9221/9213/9326；预先SHA选新test9293，不得将旧六称新留出。
fd_results.json完整7776记录（每base2592，复用方向不算独立7776样本）；旧70708会话已不存在，WSL ps无作业。
cache_audit.json已完整，两候选cache候选门槛true；未知前会话也已退出，无重复启动。
独立audit_tgs_window_candidates_precision.py会话31498 exit0，coarse通过、standard失败，aggregate false。
独立核实777旧hash和452原源hash；FD每base2592支持稳定，coarse/oracle误差max1.3686359，standard2.7011004。
支持稳定不是所有FD数值精确。cost协议/benchmark和分析审计脚本均提前冻结，仅coarse允许计时。
benchmark会话90159 exit0，378 measured F+B pairs +63warmups，126平衡blocks，模型/上游hash前后相同。
analysis及independent cost audit均exit0，coarse每相机paired墙钟比值
0.749259/0.740305/0.741011/0.735491/0.734012/0.737476/0.741558。
20k paired bootstrap全部上端<1，每相机<=.90工程门槛通过；单次未锁频1530–2962MHz会话描述。
比同语义native64参考省25.1–26.6%算子F+B时间，但比不同数值历史piecewise仍慢4.94–5.37倍。
Torch allocated峰值每相机增加12MiB（12582912B），不声称相对oracle省显存。
failure_census已exit0：standard旧14点均pass，但9213六个新depth失败，flag0；门槛未改。
coarse全部2592对FD正负端点样本逐位oracle。cache每候选747小case+11421真实选定边的Fraction审计。
图paired_cost_v3.png/pdf已视觉检查通过；前两次排版顶部裁切但保留，CSV三版哈希相同，不重跑实验。
seal_tgs_window_candidates.py会话60281 exit0，967文件4878229587B，38个源/依赖文件。
manifestSHA489230fa253ed3eca2e3a9a5ad1c5458bee2f3489f04a4337ee8c818fa7bb22c。
v9全套证据已封存，不覆盖其源码、脚本、结果或图。后续新版本新目录。所有本轮作业已结束，无等待训练。
下一步优先跨场景固定输入和剩余~5x历史成本的分解诊断；先冻结协议，再运行，仍不启动完整训练。
工程筛选门槛每相机paired median<=.90且20k paired bootstrap上端<1；这是单session描述，不等同研究目标。
不改原库、旧封存证据或打开的median_depth_note.tex。总goal仍active，GT/跨场景/训练/新颖性待解决。

## 当前入口（2026-10-20，v8失败已封存，标准pow调度探针可用）：下一轮实现单遍粗排除/标准power对照

实质进展：完成v8实现/全套诊断/14像素因果追踪/独立审计，失败不改门槛，仍goal active。
先读TGS_COMPENSATED_RENDERER_RESULT_2026_10_20.md及TGS_STANDARD_POW_DISPATCH_RESULT_2026_10_20.md。
v8封存760文件3932315637B，manifestSHA9111959a56dcabceba9ccebac94df3e42739998a8e51375a7a20e5513a16168f。
standard_pow_dispatch_v1封存17文件1475751B，manifestSHA8a3a239b4fa7a967c32cc216f86183eddf4d1f7ec6eeaf8d671cc3e8f6fd8b10。
这些source/helper/scripts全部不可覆盖；新版本新目录/脚本。

75781已exit0，precision_diagnostic_audit.json确认127/2572VJP/70680828pixels per版，5184FD支持稳定，
FD normalized分位[3.526e-8,7.642e-5,.001618,.03274,2.7011004]，非全1e-3；2161旧hash+452源hash未变。
总体all_gates=false/performancefalse：9213整图14depth像素±3.05176e-5>3e-5，其余5图pass。
6real最大VJP相对1.050625e-6，old-oracle基线与新oracle最大1.225754e-6；所有路径/median/radii一致。
全14无early-stop，原因是smooth α微扰触发T/weight/D_FMA舍入轨迹变化，不是v3/v4的停止分歧。
trace 14guard+oracle端点逐位、17692RN检查、Fraction三项分解严格相等；原失败及全部反事实保留。

标准power调度探针9399及独立审计exit0。4007输入=1176v6grid+2831全部trace到达pair。
--use_fast_math下named powf仍等于__powf；显式experimental extern C device __nv_powf与fast有765不同。
1106正常输入/正常输出子集与严格标准参考逐位同，全部2831trace也逐位同。
仍132FTZ边界差异，不是完整IEEE/严格power区间证明，也非新renderer修复。
低层C++声明只在当前CUDA12.8/sm120实验验证；官方libdevice文档是LLVM原型，不声称SDK头有公共声明。
probe较早import cpp_extension导致默认/usr/local/cuda；只读核实symlink->cuda-12.8、nvcc12.8.93及SHA
已存compiler_provenance.json。新builder务必先设CUDA_HOME再import cpp_extension，不必重跑旧探针挑结果。
precision/audit/trace/FD/baseline/dispatch全部有terminal exit，无等待作业。

下一轮避免只追小像素或只减double计数：真实实现并比较两种完整候选，不放宽旧门槛。
优先考虑“单遍可证粗排除+其余全部原生double window/alpha”：v7证实85.8–87.5%visited pairs可排除，
保留内部alpha与oracle同可消除v8浮点合成分叉，同时直接删掉已测负成本的全图preflight遍历。
无需不确定α时才需要的有界maxblend暂存/回滚；确定native路径可直接提交。尚未实现/计时，不宣称收益。
从v5复制新目录，保留n/b/Jacobian；每triangle可只缓存9hi plane floats+统一Efast+Bx/By+domain+reserve=14float，
而非v8 48float；原double coeff保持，coarse不确定/内部走原生oracle，重放flags恒0。这个布局只是方案需验证。
与“v8补偿窗口只替换__powf为显式__nv_powf”的完整版本作对照，后者仍带经验power条件且双遍历/大缓存。
前者着眼主要外部工作量和单遍成本，不把退回全double渲染器当新方法；需要实测端到端前/后向及显存。
两者都继承127/5184FD/6real/14坏点，新增未用camera/scene；六个已用相机名字见下节，不能继续称其新留出。
精度全通过后再冻结计时。质量/GT/跨场景/训练/新颖性和最终科学目标仍未完成，无optimizer/训练/原库LaTeX修改。

## 当前入口（2026-10-20，v8已集成但real9213深度14像素失败）：先封存并核验标准power修正

实质进展，非阻塞：已由v5隔离实现tile粗排除+hi/lo Dot2+紧max区间+唯一active误差，共用forward/back helper，
原projection/Jacobian/floatnormal不变，仍双遍历T区间guard，nearcap总oracle，cutoff距离RD。
协议TGS_COMPENSATED_RENDERER_PROTOCOL_V8.md；helper tgs_compensated_window_v8.cuh；
目录experiments/tgs_compensated_renderer_v8，尚未封存（需完成独立diagnostic审计和结果报告）。
stable_error10->48floats/tri，旧10保留，18split+16cache+2B+domain+reserved，增加152B/tri。
784056tri geom378702912B(原259526464)，image32868308B；不是显存优化。
四模块guard_diag/oracle_diag/guard/oracle编译71720 exit0，SASS72499 exit0，ADD6/4/0/0。
127原病例、2572VJP全部通过(29003 exit0)，5184FD(89810 exit0)全部支持稳定，数值FD还须独立复算。
6real9258/9322/9269/9221/9213+预先SHA选的新test9326，95082 exit0但real_results.all_gates=false。
6real路径/radii/median、RGB(<=4.173e-7)、全部6组RGB/depthVJP均通过(rel~1e-6)，counter/cache parity逐位相同。
只9213加权depth有14像素3.0517578125e-5>冻结3e-5；其他5图depth<=1.90735e-5。
不得放宽门槛或测成本。原2早停问题像素现flag2、RGB/aux/path逐位oracle。
flags各548/835/1182/1651/(查real_rows9213)/(9326=837)，比v5多因更多fastα带非零误差区间。
preflight double数例如9258=1864/178063370，render=376923/178132476(含整ray replay)，不可当速度。

depth_failure_plan.json固定全部14坏pixel，diagnose_tgs_compensated_depth.py 52749 exit0。
TGS_COMPENSATED_DEPTH_TRACE_PROTOCOL.md和trace_tgs_compensated_depth.py，20669 exit0；
depth_trace/results.json全部14guard+oracle端点逐位复现，packet/list/cache完全同。
全部14无early-stop、support/last/median不变，不能归因早停。
audit_tgs_compensated_depth_trace.py已exit0，17692RN检查，Fraction精确分解：
ΔD =精确floatα合成差 +T/weight浮点舍入差 +D_FMA残差差；三项精确和等于±3.05176e-5。
例如(772,1)：alpha−.6314e-6、T/w−8.6758e-6、FMA−21.2103e-6。
实际局部alpha最大差仅1.49/2.98/5.96e-8，不能由“小α差”推出深度绝对门槛。
7预声明counterfactual都保留：float-u+doublepow 14点全进门槛、max2.28882e-5；
native-u_float+fastpow仍5失败；float-u+标准powf14点全进门槛、max2.28882e-5；
native-u_float+标准powf全进门槛；Dot2[p,s]重建doublephi+doublepow14点depth/RGB全部逐位oracle。
这只是已失败像素的回放，不能宣称修复或新留出。trace不带use_fast_math、ftz=false，显式__powf保持fast，
0/1端点逐位证明该编译环境在本14packet复现原生。标准powf需要在全局fastmath渲染器中明确调度。

独立oracle/cache审计已完成：audit_tgs_compensated_baseline_diagnostic.py(从原baseline脚本派生，
仅去“guard全real先过”前置条件，所有实际比较未松)，49195已exit0。
baseline_cache_diagnostic_audit.json通过127全部可见747edges、6real各约1632edges(9213=1629)，
old v5 oracle整图/coeff/旧10error逐位相同，6real旧oracle RGB/depthVJP vs新oracle均<=1.226e-6。
原baseline_cache_audit.py及audit_plan.json仍保留未执行严格流程，不覆盖；diagnostic明确guard_gatefalse。
独立总审计正在/已执行handle75781（必须先poll同handle，不得重启）：
audit_tgs_compensated_precision_diagnostic.py，完整barycentric/FD/native/cache/source/SASS复算，
预声明保留9213 real gatefalse、禁止performance，不以assertfalse中断有用的诊断。
检查precision_diagnostic_audit.json及handle最终状态；原严格audit脚本未改，未有precision_audit绿色结果。

下一动作：完成v8诊断封存，不改变失败结果；然后建立新隔离版本评估更准power。
优先验证全局--use_fast_math下如何调用标准powf而不改变原preprocess/Jacobian算术，
可能显式extern C __device__ __nv_powf再检查SASS/固定grid，尚未实施，不要假定此C++入口可用。
查过主源CUDA12.8 libdevice __nv_powf官方文档：有LLVM prototype，区别__nv_fast_powf；
本地crt头grep未暴露C++声明，不能把LLVM原型当现成头文件API。
读https://docs.nvidia.com/cuda/archive/12.8.0/libdevice-users-guide/__nv_powf.html；无版本URL起初404。
WSL无/usr/bin/rg，曾因未保护regex管道读命令失败，后用grep(无匹配exit1)；未修改任何系统配置。
若用标准power，必须全127/5184FD/6real+14坏点+再新未用相机/scene全套验证，再成本。
不允许用trace14点过关当renderer通过，不将sigma.02域外FTZ控制冒充真实故障。
power严格界仍未证明；训练/GT/效率/新颖性目标未完成。无optimizer/训练/原库模型LaTeX修改。

## 当前入口（2026-10-20，v7真实ray工作量核验完成）：隔离集成粗排除/补偿及active区间

本轮实质进展。读 `TGS_REAL_WORK_RESULT_2026_10_20.md`、`TGS_REAL_WORK_PROTOCOL_V7.md`、
`TGS_REAL_WORK_ADDENDUM_V7_1.md`、`TGS_ACTIVE_EDGE_INTERVAL_NOTE.md`。
证据experiments/tgs_real_work_v7，62文件551969887B，manifestSHA
9f246c3a4638b29fdae09680f49a74a20f910679a4e2cd03617251b001490aee；627个v5/v6封存文件hash不变。
只读原v5oracle的geom/sorted list/ranges，实际clamped tile footprint统一界，沿oracle alpha/T路径统计，
候选alpha未进入图像。未训练、计时、原库/模型/LaTeX修改；goal active。
5完整相机1245x825，旧9258/9322/9269/9221加预先SHA选中的新test9213，同stump7000 784056tri SH3。
9213现在已用于设计；后续候选要再加未用camera/scene，不能反复称其新留出。

初版v7跑完5相机exit1：9258 path channel1的627差异恰等于全部K=0像素，observer uint=-1/native0。
原库float(-1)->uint转换与直接uint(-1)不同。RGB/7aux/maxblend/last ID全部相同，原失败results gatefalse保留。
v7.1独立source副本唯一改mid=__float2uint_rz(-1.0f)，另名模块，完整5图再核验全部端点逐位相同；
仅627哨兵改动，counts/cache/split/B/domain/samples和所有inputs哈希不变。不是原库bug修复或性能重测。
首次94559已exit1；修复67434 exit0；独立审计15403 exit0；数学/绘图exit0，无等待作业。

5相机visited=[178132476,352762052,404628221,446057503,435267740]共1816847992pairs。
coarse outside分数=[.85864084,.87546344,.86246388,.85804648,.85870662]，不是速度或唯一射线比。
coarse共1567264681，comp再outside860，insideunique249582269，edge不确定34，sign不确定148，domain回退0。
全frame统计中证书outside/inside/active无native矛盾；所有可见domain/footprint独立NumPy核验，
另有理数抽审8157edgecache+2560实际预定pair，全部通过。不是18亿对CPU有理数全审。
insideunique条件alpha预算通过后加经验2e-6+cap/cutoff margin，非outside对条件接受约71.85–75.82%。
所有预算通过alpha实测差未超allowance，但9221有2 cutoff分歧（cap0），margin必须保留；
未单录2项margin归属，不能称完整候选分支门槛已过。全部仍沿oracleT路径，未验证候选合成或VJP。

实际纳入贡献K而非n_contrib ordinal：中位[12,35,43,53,49]，max[186,197,186,188,214]。
K128溢出比例[.00838359,.00853742,.01163734,.03516417,.03968358]；K64约6.93/7.75/20.53/38.37/36.69%。
K256仅本五图无溢出，不是通用界。K=0的627像素也保留。K_survival.png/pdf/CSV已生成目视检查。
这支持考虑暂存128+溢出回退，但未实现/测local memory/spill/occupancy/ray replay成本。
观察器ptxas79regs/0spill只是observer，不代表暂存候选。

新数学补充已证明而未实施：每边[li,ui]则max边属于[maxli,maxui]。
若la>max其他uj，则a唯一active，phi只需Ea而非max(E0,E1,E2)；端点使用RD/RU，cap/cutoff距离要向下界。
v7已唯一后仍用Emax归一化，可能浪费精度。393个frozen insideunique样本区间核验通过，
max半径/active半径中位[6.10,6.38,7.83,10.09,7.50]，max3383；非均匀pair样本不能外推全frame速度/回退。
精确有理反例r=(-2^-20,-512,-256),e=(2^-40,2^-15,2^-16)：旧sign不确定、tight insideunique可证。
这是标准interval/max性质应用，不可当一般新定理/新颖性证明。

下一步优先真正隔离渲染集成（不要无休止只做observer）：以v5显式RN为基，实际tile粗界，hi/lo Dot2，
紧max区间及active error，保持inside/cap/cutoff/合成T区间和backward选择缓存；power仍经验条件不伪装严格证明。
继承全部127cases/5184FD/旧4real+9213完整RGB/aux/路径/6组VJP，再加新未用camera/scene；精度过后再计时。
双遍历仍负成本，可随后fused bounded list128，仅accepted ray提交maxblend，不确定/溢出重放，禁止提前global max回滚。
新数组不要覆盖已有float normals/backward Jacobian语义而未验证；缓存内存/不可避免64pow要实际量化。
数学/GT质量/跨场景效率和学术新颖性未完成；未授权完整训练。原v5效率负结果不变，不得从计数宣布加速。

## 当前入口（2026-10-20，v6补偿边方程条件包络通过）：下一项真实ray工作量及隔离集成

本轮实质进展。读 `TGS_COMPENSATED_EDGE_PROTOCOL_V6.md` 和 `TGS_COMPENSATED_EDGE_RESULT_2026_10_20.md`。
v5效率负结果保持；未重测成本、训练、更新原库/参数/LaTeX。
原文Ogita/Rump/Oishi Alg5.3/Prop5.5核对：Dot2误差u|D|+gamma_n^2*S，n=6，不是12。
hi/lo系数+独立delta，后验包络(u|r|+gamma6^2*M6+Delta)/(1-u)；另加binary64有序求值误差。
充分域：整数localxy<=8192，原c和非零hi/lo在[2^-60,2^20]，2^-83格点保证无下溢/溢出。
不是原几何投影的普遍包络，也不是新Dot2算法。未来域验证要防FTZ等零比较把次正规坐标误当整数。

冻结27712行：127例747个可见plane+4已用相机各512tri的6144plane，4人工坐标/plane，再148控制。
域内27701，域外11（仅控制）；人工坐标不是真实ray分布，4相机非新留出。
strict/fast_ftz独立CUDA12.8 sm120二版都运行exit0；GPU5090Laptop driver616.92 24463MiB。
各166206productEFT+138505TwoSum恒等式，1161行correctionRN，所有域内包络/证书通过，轨迹逐位相同。
259旧证据hash不变，27712缓存有理数审计，27701native64/CPU64端点逐位相同。
coarse正证书13506、补偿13751、补偿负证书13883，全部正确；不可外推为实际ray排除/回退率。
裸符号粗176分歧、补偿32分歧，32全部落于包络跨零，需double；最差相对108.085不可隐去。
最差P=153/4503599627370496≈3.397e-14,r=-3.638e-12,E=9.677e-10，正负证书都不通过。
真实cache人工nearline的补偿/粗绝对误差中位2.613e-9/9.844e-6；大坐标最大4.835e-4/1.295e-3。
完整分组/所有样本保留；包络变紧不等于成本降低。

重要新界限：CUDA12.8 Sec13.1说函数表误差非保证上界；4ULP标准powf不能证明严格幂函数区间。
fast_math重写named powf为__powf，不能混名；1176固定power格点mpmath100位及全量200位复核。
strict标准powf该有限网格<=1ULP；283active-normal项maxabs4.353e-8，intrinsic3.052e-7/5ULP。
这不构成全域误差保证。FTZ次正规u≈9.052e-39,s=.02控制，参考.173434256、float调用输出0；
float先经FTZ再castdouble的pow表达式也0，SASS有FMUL.FTZ乘1再F2F，cast不能恢复原输入。
不是v5真实renderer故障证据（高精度u由doublecoef直接算，lower>0回退），当前real最小sigma.2168
时此类u的alpha已低于截止，不能将合成.02控制解释为真实图像质量下降。
fastFTZ两个2^-140边控制裸包络失败，域外均拒绝出证；反例未删除。

下一项：按真实tile footprint预计算粗界，真实排序list上统计安全排除/补偿/double数，别外推人工坐标比。
然后隔离集成，active-edge区间排序、inside/cap/cutoff、合成T阈值证书和backward缓存一致性都要保留。
幂函数可先明确经验余量作可证伪候选；若要全数学证书，需log/exp有向舍入+截断余项，非powf表格。
双遍历成本仍待解决；保留有界贡献暂存另一方向但须先查真实有效贡献直方图与溢出/localmemory。
renderer候选继承127/5184FD/4完整相机，加新留出；精度通过再成本。
本轮没有重跑renderer整图/VJP/FD，不能在已有次数上重复计数。训练/GT/质量/效率/新颖性未达成，goal active。
证据36文件18489430B，manifestSHA95114628796d6dc005335732b11e63df450c91593d3e6a0369b2e4cfe0bae5c2。
原生无session直接exit0，审计11627已exit0，endpoint200位复核exit0，无等待训练/编译。

## 当前入口（2026-10-20，v5精度/留出通过但成本失败）：下一项核验补偿点积与粗过滤

本轮实质进展。读 `TGS_PHASE_COST_RESULT_2026_10_20.md`，证据experiments/tgs_phase_cost_v5。
显式RN锁住float window/T/合成，并保留double边显式RN；guard/oracle各diag与lean四版。
无counter保留相同缓冲且清零，只去计数atomic；SASS前向计数ADD6/4->0/0，back梯度atomic保留。
127原案例全部保留，四版2572VJP、70680828 RGB像素/版，diag/lean整图/cache/flag/ID逐位相同，
采样VJP差1e-5门槛通过。5184FD支持稳定，数值max1.36864，不能说全1e-3。
新32合成阈值例的CPU64/native分歧继续保留，native路径目标不是精确实数阈值。
1472旧封存hash+452源hash核验通过，precision_audit.json all_gates=true。
4真实完整相机train9258/9322+该方法留出test9269/9221，同stump7000，RGB/depth固定VJP全有限且通过1e-3。
RGBmax<=2.981e-7，depthmax<=1.526e-5，median/贡献者ID相同；test9269几何RGBVJP相对1.145e-4，
不许写成全1e-6。lean模块guard88cf61ec2a05232f12e9719daec6ed85fb0867e9ec64aaa1ddf8838eb996238b，
oracle e8aee33c860f346247a1e48a9fc61f70a7b035ef2ea55b800c5960280ad24ab8。

重要负结果：独立精度审计后已实际完成216交错native调用对，全部样本/输入哈希不变。
guard/oracle配对wall中位比9258=1.102742、9322=1.079620、9269=1.069572、9221=1.084729；
单次会话配对块bootstrap95百分位[1.0613,1.1441]/[1.0254,1.1026]/[.9974,1.1657]/[1.0174,1.1551]。
guard调用对中位66.88/107.75/128.36/140.27ms；oracle60.44/100.21/117.59/128.73；
历史piecewise9.18/13.58/16.67/18.26ms，数值前向不同，只作成本参照，勿宣称等价。
前向guard/oracle比约1.38–1.43，反向约.75–.79，子步骤收益不足以抵消整体成本。
GPU时钟1507–2820MHz、温度46–60C，未锁；区间描述本次会话，不是独立场景统计推断。
Torch peak约957–1002MB，历史834–870MB，不含driver/register/local总峰值；counter布局仍保留。
成本及bootstrap独立复算通过，图cost/cost_comparison.png/pdf已目视检查，所有数据保留，效率推广false。

下一优先项：先读原始EFT/compensated dot理论并核验误差界（不要凭记忆套Dot2常数）；
考虑每primitive按tile footprint预计算统一float边误差上界，用cheap float phi安全排除外部像素，
仅内部/不确定像素用hi/lo系数+TwoProductFMA/TwoSum补偿FP32，严格界不满足才FP64。
这有望减少常见FP64 pow和每像素大段误差界运算，但尚未实现/验证，不可写成收益。
另一方向为fused render+有界local贡献暂存、只重放不确定/溢出ray，注意maxblend不能先提交；
需先统计真实有效contributor直方图并核验local-memory成本，不能用n_contrib ordinal代替有效数。
新候选继承所有127/FD/4real门槛并加入新留出；不重复已有计时筛选好看结果。
未训练/GT/参数更新/原库/LaTeX修改；学术新颖性/最终质量效率收益未证明，整体目标active。
编译98001/案例38561/真实36142/FD50387/精度54526/成本42864全部exit0，无等待作业。

## 当前入口（2026-10-20，合成区间guard v4通过扩展检查）：下一步审查phase一致性并验证实际成本

本轮实质进展，读 `TGS_COMPOSITING_GUARD_RESULT_2026_10_20.md`、`TGS_COMPOSITING_INTERVAL_NOTE.md`。
CUDA trace_v1四个原生端点逐位复现，同排序/cache：9258(116,742)第338项early-stop分歧，
adaptive停338/oracle343；9322(740,268)第760项分歧，停760/768。median一致。
漏掉triangle120791深度88.77187/199338深度5.94180；第二像素当前alpha逐位相同，incoming T漂移跨eps。
trace独立audit将新增depth贡献、公共前缀权重差、浮点累积残差分离，因果解释已经成立。

guard v4分类传播outward T区间，跨incoming .5或test1e-4时标记整ray，最后一遍用oracle窗口并缓存flag给backward。
分类不写图像/maxblend，防止否决路径留下global max；停止语义与epsilon未改。
double边方程显式舍入，但power余量经验、投影与tileguard非普遍IEEE包络；不要宣称无条件数值证明。
adaptive模块252b4826cce6fa69c538bd5245950480bca374cede8d71de5b76000d0752fe7d；
oracle模块2d1e3c9c2cdf71a731c7984cbe27e8c70852d4c99970afd17dc05b5bb18c7764。
全部95保留，新增16合成阈值例(最初分母float32加EPS问题保留)及16配对double修正例，
在v4 native输出前扩展为127，input_plan_extended.json，不覆盖原111计划/输入，addendum详述。
每版本70680828RGB像素，1286VJP两版总计，全部声明门槛通过；全图contributor/median ID和radii相同，
所有flagged RGB/aux/contributor逐位相同。原95独立RGBmax1.869e-7，geometry VJPmax约7.515e-5。
5184原FD全部执行支持稳定，数值max相对1.36864，不可称FD全1e-3。1087旧hash+452源hash检查通过。
重要：4个新median病例各2像素CPU64/native median差2，两native仍一致；预声明目标是native oracle路径，
不是恢复精确实数阈值，所有CPU差异保留，不能将127 native门槛通过说成CPU64所有aux通过。

真实完整stump7000两train相机，RGB和weighted_depth两个固定VJP均完成且梯度有限。
标记147/161ray(.01431%/.01567%)，原两个depth异常像素flag2、现在逐位同oracle。
整图RGBmax差均2.384e-7，depthmax差1.526e-5/5.722e-6，median差0，maxblendmax3.576e-7，
整图contributor/median ID一致；RGB/depth geometry VJPmax8.660e-6/8.090e-6(9258)，第二相机<6.3e-7。
真实v3/v4 oracle RGB/全部aux/radii逐位相同。核心精度验证通过，成本/质量/新颖性仍未验证。
成本不能隐去：分类和最终render都跑全图，window求值数约1.9999倍(不是延迟比)；
geometry payload144B/tri，较v3多12B/tri；另加4B/pixelflag，实际buffer259526464/32868308B，非完整显存峰值。

下一步优先审查helper在分类/最终render中的phase一致性(当前只显式约束double边和，float fast算术也应确定)，
再建立无counter版本保留全部127/FD/real门槛，冻结交错成本测试，对比正确oracle和guard，并加入新留出相机。
研究更紧alpha误差界/有界ray贡献暂存以减少双遍历；当前尚未实现，别用少量flag预测加速。
不要盲移植Gamma收缩：高精度phi与float cotangent的消去误差须重新审查。
未启优化器/训练/GT评估，不改原库/模型/LaTeX；质量/速度/新颖性和整体学术目标未完成，active。
trace72907、编译5825、案例98918、真实79321、FD18397、独立审计48572均exit0，无等待作业。
目录experiments/tgs_compositing_trace_v1与tgs_compositing_guard_v4。

## 当前入口（2026-10-20，自适应v3解决窗口反例，真实尾部合成仍需追踪）：先分析两个weighted_depth像素

本轮实质进展，读 `TGS_ADAPTIVE_WINDOW_RESULT_2026_10_20.md`、`TGS_ADAPTIVE_LOG_BOUND_NOTE.md`。
v3保留79旧/压力/边界案例，再加16例，共95；三隔离版本arithmetic64/adaptive/oracle64，均直接Gamma。
从原投影构造double normal/offset/invr，新增10double+10float+3uint=132B/triangle诊断payload。
按|dalpha|<=.99*sigma*d_u/(uhat-d_u)和inside/正u误差界选择回退；power2e-6余量经验性，
double投影和onepixel tileguard也不是普遍IEEE证明；oracle64窗口double，但SH/合成/梯度累积仍float。
adaptive模块SHA18292f28af5b5967c3c4420c6e83d850eb38d2c36a8f3eea823f2cc4419154d2，
oracle64 SHA7390fb64b9bbe0384c10704956d6e95904c4d2cd20706b164b12e665062d261a。
95图每版68583676RGB像素，1545VJP三版总计；adaptive/oracle64所有既定门槛通过，
只转旧float系数到double求值的arithmetic64有12失败，说明必须同时处理系数和像素算术。
独立行列式/重心坐标整图与Torch64 VJP审计通过；adaptive整图max1.869e-7，VJP组max约1.242e-4。
532旧封存hash+452原源hash未变，1210缓存误差界有理数逐项通过。
5184原FD全部执行，tile/alpha/CPU分支稳定；数值归一化误差max1.36864，不能说FD全1e-3。
最初生成器once匹配2处，在CUDA前退出，generated_partial_1保留；修正后编译48077 exit0。

真实stump7000完整784056tri SH3、两train相机1245x825，固定参数denseRGBVJP，没训练/GT/计时。
adaptive/oracle64六类梯度全有限，radii相同，RGBmax差2.95043e-5/2.41399e-6，几何VJP相对8.632e-6/3.576e-7。
回退11.0606%/10.6576%，分母是包括outside的window求值对，不是有效贡献/唯一像素。
sigma min.216805，中位.93469，464951/784056<1；每几何buffer250117696B，不是完整显存峰值。
重要未解：两相机各1个weighted_depth像素差>3e-5：9258(116,742)差.0019197464，
9322(740,268)差6.53267e-5；alpha差2.16365e-5/1.09673e-5，adaptive少5/8次window求值。
median depth逐位相同，不要误归因median；需逐contributor incoming/test T轨迹来证明early-stop因果。
`real_independent_audit.json`保持production_promotion/performance=false；不能只引用RGB门槛通过。

下一步：固定上述两个真实像素，读取tile排序list和cache，追踪实际alpha、incoming T、test T与停止位置；
按log note递推T误差，设计uncertain-ray double-window重放并把前向选择缓存供backward，
保留95案例全部整图/FD及两真实相机新尾部反例。尚未实现，不要写成已验证。
当前skip-current在testT<eps时停止，其当前+尾部权重最坏上界eps/(1-cap)，不是eps；
include-current改语义需单独实验，不得伪装数值修复。正确性扩展后再移植contraction、去counter做成本比较。
目录experiments/tgs_adaptive_window_v3，编译48077/固定25362/FD98816/审计57988/真实38206全部exit0，无等待作业。
授权仍仅数学+固定CUDA，不启优化器/训练，不修改原库/模型/LaTeX；质量/效率/新颖性未证明，目标active。

## 当前入口（2026-10-20，稳定窗口v2局部门槛通过但整图出现小sigma边界反例）：暂不计时

本轮实质进展。先读 `TGS_STABLE_WINDOW_RESULT_2026_10_20.md` 和 `TGS_SMALL_SIGMA_STABILITY_NOTE.md`。
隔离v2采用double投影、整数anchor局部float32顶点/边、area/perimeter r、alpha>=1/255同心支持AABB、
有符号floor和exclusive上界+1并外扩1pixel；仍不是普遍IEEE覆盖证明。direct/contraction两个模块SHA
eb0b40a5d9c910707e84faaadf3d05487c3068ae1db9502398c9b9a623195450 /
c55b352ed3b02a290a18e88e3587dbecf2f803160cbf0a212a96172f33bcf6d0。
冻结协议/输入后79案例(18旧+48压力+1有理+12旋转裁剪变深度)，902VJP和5184FD记录均完成；
所有既定采样前向/VJP/支持稳定门槛通过，两个新版本整图RGB/aux/radii逐位相同；六旧失败全恢复。
采样前向max9.683e-7，geometry max1.286e-4(aux distortion)，48压力前向max1.223e-7。
GPU保存rect末尾解析每次以tile实例总数校准，FD逐primitive/pixel参与集合+alpha支持稳定；
12新案例全部2986有效整数采样无漏tile/alpha。原生FD精度不是全1e-3：最差归一化2.701，已逐条分解
native-minus-CPU和CPU有限步长/可表示路径项，所有探针保留，不能把分支稳定冒充FD精度通过。

重要：独立门槛审计后又对18旧案例整图CPU64检查，16图max<=1.71e-7，
alpha_saturated和early_stop_skips_fourth各39像素仍>3e-5，max .442398/.403816。
`independent_audit.json`的protocol=true原记录保留，`promotion_decision.json`明确当前
performance_measurement_allowed=false，未计时/训练。不能只引用79个采样通过来推广。
sigma=.01999999955时，像素(176,44)/(188,35)原始精确phi=-377/200000010，
u=377/8000000000，80位alpha=.71354536876，两点严格在内且tile都有，GPU alpha却0/.72367668。
独立有理审计还证明量化顶点三角形仍包含两点，缓存float normal/offset精确phi都=-45/16777216，
所以不能单独归因vertex量化；像素乘加也破坏精确切向不变性。原证书解释偏窄，以独立audit分解为准。
0<s<1窗口仅Hölder，近零导数无界；局部远离零的条件数可用，不能把全局充分预算误说成所有像素需要227位。

下一步保留全部79案例、两张完整图及精确边界证书，隔离double边方程求值与double/hi-low系数，
从原始几何区分投影、缓存、像素算术误差；设计带inside/cap/cutoff/early-stop分支证书的选择性高精度候选。
不要只将错误float缓存转double就声称恢复原窗口，不得夹sigma>=1/删边界像素掩盖。
扩展正确性通过后再比较同一前向direct/contraction效率，尚未证明质量/速度/新颖性收益。
目录experiments/tgs_stable_window_v2；原v1封存293哈希及452源哈希未变。
编译88211、执行88151、独立审计77012及单三角形证书全部exit0，无待等训练/编译。
当前授权仅数学+小规模CUDA，未更新LaTeX/原库/模型/优化器；整体目标active。

## 当前入口（2026-10-20，收缩v1整体门槛失败并发现真实tile覆盖缺陷）：先修共同前向再比较效率

本轮为实质进展。先读 `TGS_RADIUS_CONTRACTION_RESULT_2026_10_20.md`、`TGS_TILE_COVERAGE_NOTE.md`。
v1冻结协议未改，SymPy恒等式核验通过，隔离contraction模块SHA11153f8a6628d2412b433472ec9b56dbcd9e758929847242d4fe6f600ae0fdbe。
18旧场景全部通过、前向不变；48压力×8正负VJP、两版本共804VJP与4608FD探针步长对照均保留并独立审计。
v1和旧piecewise同样有6失败：stress_02/15/29/30/40/42。五例VJP组仍<=1e-3，失败为执行脚本预先加的
CPU前向abs<=3e-5检查（差约3.7e-5至6.0e-5）；stress40还真正失败理想窗口VJP（geometry约.9302）。
v1 performance_measurement_allowed=false，未做性能计时或训练。不能删案例、改冻结协议或以只测旧场景宣布成功。

stress40主因是共同前向漏tile。旧smooth标记只含CPU窗口/贡献和native radii，漏了tile成员关系。
补充审计按单三角形样本支持掩码发现每版本4个FD探针对跨tile参与集合；原字段保留并明确其范围。
不能把这些FD当同一光滑分支；缺失像素的零梯度可以是截断实现的正确导数，却不是理想窗口导数。

精确有理证书：三角形(317/8,124),(365/8,124),(317/8,257/2)，r=1.5、sigma1、o.5，
alpha>=.01支持maxY=128.44，旧floor((maxY+15)/16)=8应为floor(maxY/16)+1=9。
像素(40,128)应alpha7/120，却输出0；邻像素(40,127)=.125且radii5，排除整primitive剔除。
仅改四处tile上界的新隔离版本恢复alpha.05833435，VJP误差1.048e-6；同一冻结stress40八VJP恢复到<=1.313e-4。
这只是因果诊断版本，不改原v1失败结果。模块SHAe42169eaad7765fa238b7660029d1e8517975825cacaadef4780e837d8503aa8。

真实完整stump7000固定参数、两train相机，仅改四处上界：DSC9258新增可见88，35585像素delta>3e-5，
maxRGBdelta.07668/全图deltaMAE4.736e-5；DSC9322新增176，78300像素delta>3e-5，max.15611/MAE8.899e-5。
不是GT误差或质量收益；alpha也有少量下降，未拆分早停/算术原因。之前control整图与缓存逐位相同。
全部证据在 experiments/tgs_radius_contraction_v1，tile_coverage子目录有证书/因果干预/真实整图及独立审计。
编译62834、v1执行66235、仅上界编译65494、真实前向88759及其它诊断均exit0，无训练/编译等待。

下一步：用新版本协议保持全部66旧/压力输入及有理反例，明确native tile支持稳定检查；
建立整数采样保守覆盖，并审查预处理.01与kernel1/255双截止、负坐标边界；五个小前向偏差需逐项数值归因，
评估局部坐标/高精度预处理后，再在同一正确前向下比较直接Gamma与重建Gamma。新设计尚未实现，别写成已通过。
仍仅数学+小CUDA，不启动优化器/完整训练，未改原库/模型/GGGS/LaTeX，整体目标active、学术新颖性/最终收益未证明。

## 当前入口（2026-10-20，真实成本/参与频率已测）：下一项为局部坐标重建半径VJP候选

本轮为实质进展。先读 `TGS_COST_EXPOSURE_RESULT_2026_10_20.md`。
完整公开stump7000、784056三角形SH3、两个固定train相机1245x825，control/geometry/piecewise
前向RGB/aux/radii相同、六组完整VJP有限。108个交错计时调用对：geometry反传配对中位比
1.06764/1.03723（约+6.76%/+3.72%），piecewise1.05756/1.05212；调用对约+3.2%-5.2%。
这是含分配和.item同步的native API延迟，不是纯kernel或完整训练速度；单会话未锁时钟。
Torch分配峰值三版本相同，835161088/852922880 bytes；不含寄存器/驱动等显存。
目录 experiments/tgs_backward_cost_v1，原始108样本及描述性bootstrap区间保留。

独立counter构建通过四个1像素校准，仅测频率、不计时；实际backward有效贡献对：
DSC9258 23229103对，饱和5、精确/容差多max86；DSC9322 39664028对，饱和1、多max133。
合计62893131对仅6饱和、219max平局（约0.0954/3.4821次每百万对）。
这不是唯一像素数/非零梯度数/完整训练发生率，不能用频率推断损失影响。
counter前向不变，六组VJP相对差<=6.616e-6；每三角形整数计数与当前forward及旧缓存支持集合相同。
目录 experiments/tgs_branch_exposure_v1，模块SHA5465d6655ea84246c1d5701662e736edc1c92b62c62e28ab4afaef26780a866d。
benchmark23776、counter编译80600、counter执行28562均exit0，审计均exit0，无等待训练或编译。

主线因此优先普遍窗口链式项，cap/max与normal_floor作为正确性保障，暂不宣传实际质量收益。
新可证伪优化候选：`TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md` 已冻结，尚未实现/数值验证。
用边VJP Gc=sum beta*(pixel-center)、B=sum beta，以Gamma=-sum[n·Gc+n·(center-edgepoint)*B]
重建log-radius系数；法线链改用Gc+(center-edgepoint)*B，省去逐像素额外Gamma atomicAdd。
避免直接全局坐标收缩的大数相减，但浮点可靠性未证明。下一步先符号/独立数值核验、隔离实现，
重跑18冻结场景及48位移/细长压力场景；保持全部病例和既定精度门槛，失败不得删案例。
性能门槛/范围详见协议，不能把代数方案或预计速度当成果，不能修改已冻结协议原文件。
仍在数学+小CUDA授权内，不启动优化器/完整训练；整体学术目标active、质量/效率收益与新颖性未证明。

## 当前入口（2026-10-20，分段导数与辅助图核验完成）：下一项应测实际开销/触发频率并准备公平训练协议

本轮为实质进展。先读 `TGS_BRANCH_AUX_RESULT_2026_10_20.md`、`TGS_BRANCH_AUX_DERIVATION.md`。
新增9场景1458探针步长对照，确认geometry_fixed另有三个分段/次梯度问题：饱和参数梯度非零、
max两边平局无权重相加破坏同心缩放零方向、normal叉积范数<1e-4时导数分支错误。
新piecewise模块保留前向，选精确cap导数/首条精确max边/正确normal分段导数：
饱和参数梯度归零；平局方向0.283808524→1.806e-8；normal_floor几何误差36.539%→2.239e-7。
普通alpha/depth/normal/median及固定早停通过；distortion VJP误差约1.085e-4，原生FD约1.43%，
不能把所有辅助图宣称到1e-6或把全部FD称为同精度。max点只检验不变性，不宣称唯一普通梯度。
两版本9份完整RGB/aux图逐像素相同；另9份旧RGB/SH/真实子集回归也全通过，原前向图不变。
独立NumPy复步长、两两distortion公式审计通过，旧封存79条哈希未改。
模块SHA f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852，
目录 experiments/tgs_branch_aux_v1；控制和piecewise结果均保留，勿覆盖或重复完成的执行。

重要负证据：真实stump7000的784056三角形中3203落入normal_floor（约0.4085%），
但两个已查train相机DSC9258/DSC9322中这些三角形的cached visible/contributing均为0。
不能用该合成反例解释当前两视角质量，暂不把它升为主研究方向。
opacity>.99全局939，两个视角有贡献151/160，但这只是实数cap必要条件，未测实际cap像素频率，max平局次数也未测。
固定主仓默认normal权重1e-4/distortion0/mesh起点5000，但checkpoint未保存这些字段，不能当作实际训练配置。
原生检查78831、新编译/检查4448、九旧场景回归及CPU统计均exit0，无训练/编译等待。

下一步优先测一般窗口修正的真实场景反传成本，或用冻结计数方案量化cap/max平局实际参与；
准备区分SH修正、像素Jacobian、半径项、分段/代理梯度和学习率补偿的公平训练对照。
先不启动优化器/完整训练，仍在用户授权的数学+小CUDA范围，整体目标active、质量/效率收益与新颖性未证明。

## 当前入口（2026-10-20，遮挡/SH耦合及真实子集VJP过门槛）：下一项是饱和/边界/辅助输出

本轮为实质进展，先读 `TGS_MIXED_REAL_BACKWARD_RESULT_2026_10_20.md` 及同前缀DERIVATION。
只读加载前轮已编译控制/修正版 .so，旧40个封存文件未改。
三层同几何×常量色/SH1/SH3、非零背景，810探针步长对照；修正几何误差<=3.722e-7。
Torch float64及独立NumPy复步长参考一致到5.334e-16；预设h=.003的原生FD误差约1e-4。
真实stump7000两个旧train相机DSC9258/DSC9322，各4个重叠三角形、3像素、128裁剪保持像素尺度，
另720对照。控制几何误差约98.4%，修正后8.168e-7/1.638e-6；sigma/opacity<=1.420e-6、SH<=1.161e-6。
真实FD中间h=.0001误差0.3044%/0.09892%，勿误称其精度也到1e-6。
独立审计重算原始采样损失/实际float32扰动/梯度组误差/全部门槛；checkpoint tensor审计确认所有保存参数逐位来源。
五份完整RGB在控制/修正版间逐像素相同。真实裁剪仅子集，无GT或test/优化器，不是整场景质量测试。
实验目录 `experiments/tgs_mixed_backward_v1`、`experiments/tgs_real_backward_v2`。
真实v1在requires_grad张量转numpy时中止、未调用CUDA；失败及原脚本保留，v2显式detach只读快照后完成。
合成13816、真实52041及CPUcheckpoint审计均exit0，无待完成训练或编译，勿重新等待这些句柄。

下一步在已授权数学+小CUDA内：单列alpha饱和、max ties、早停导数约定，区分精确导数和可能有意的代理梯度；
再核验depth/normal/median/distortion输出。不要宣称整个backward已正确或直接启动完整训练。
之后训练协议须控制SH修正、像素缩放、半径项和学习率补偿，真正测试图像/几何/效率收益。
本轮未改模型/GGGS安装/原native归档/LaTeX，未改变失败maximin策略结论，学术新颖性/收益仍未证明，整体goal保持active。

## 当前入口（2026-10-20，几何 VJP 缺项确认并隔离修正）：下一项是合成/SH混合及真实子集核验

本轮有实质进展，先读 `TGS_BACKWARD_RESULT_2026_10_20.md`、`TGS_BACKWARD_DERIVATION.md`。
固定 native 6d61f4c，在四个光滑单三角形RGB场景，以预计算常量颜色隔离几何，
原几何VJP误差98.9%-99.0%；仅补像素Jacobian后44.1%-49.3%；再补内切半径导数后<=7.424e-7。
不是只差学习率：最优标量拟合仍约30%方向残差。sigma/opacity联合VJP误差<=5.413e-7。
前向四张整图三版本逐像素相同。528参数/步长对照全部保存并通过NumPy独立闭式导数审计。
有限差分h=.01误差约2.3e-5至3.3e-5，较小h有FP噪声，勿宣称FD也达7e-7。
证据 `experiments/tgs_backward_v1/summary_audit.json`，脚本probe_tgs_backward.py、audit_tgs_backward.py，
隔离生成器tgs_backward_fixed_backend.py；原来源452文件/GGGS安装/模型/LaTeX未改，未训练。
scale_only27868、geometry_fixed22131均exit0；原control完整命令exit0，无待完成训练或编译。
初次矩形fixture gap断言中止记录已保留，安全调整测试点后完整重跑，不要把它当renderer失败。

当前修正只覆盖未饱和、唯一active edge、固定贡献集合的RGB几何分支，不称全部反传已正确。
下一步：固定排序多三角形+非零背景、SH对几何混合VJP，再做冻结真实模型确定子集的VJP诊断；
饱和/边界约定单列，暂不改旧模块，不启动优化器/完整训练或复活失败的maximin候选。
已授权数学+小CUDA有效，整体学术目标仍active，尚无真实训练收益/新颖性结论。

## 当前入口（2026-10-20，固定容量候选已否证）：停止train_maximin4 v1，准备完整反传核验

本轮为实质进展。先读`TGS_TOPOLOGY_RESULT_2026_10_20.md`及同日prior-work文件。
冻结协议/训练几何评分后，在16个新事件（一般8+压力8）、4相机（三角形每次均+3）比较
red4、经典world_longest4、train_maximin4。新增test `_DSC9213`,`_DSC9326`此前未本地模型渲染。
192规则视角案例×2appearance=384记录，272实际split调用、56已标别名案例，CPU审计通过。
maximin13/16选择red4；压力train全子丢失red/longest/maximin=8/2/7，压力test=1/0/1。
两个预设gate都false，不能以训练几何分数更高替代目标成功，不继续该版本训练/搜权重。
结果在experiments/tgs_topology_v1；来源审查确认最长边不是新方法，锐界新颖性仍未证明。

下一必要检查：固定输入下TGS几何/sigma/opacity反传是否与有限差分一致；目前只有SH系数梯度过检查。
使用现有隔离模块与branch远离阈值的输入，保留全部差分步长/边界和失败证据；不预设一定有bug。
这在已授权数学+小CUDA范围内，不包含完整训练。勿重复旧前向或用全拒绝细分当作方法收益。
CUDA27040、审计83845/34607、文献归档30530均终态exit0；无训练等待。总科研目标未达成。

## 当前入口（2026-10-20，真实输入门槛已跨过）：stump7000冻结诊断与内切半径锐界

本轮为实质进展，先读`TGS_REAL_EVENT_RESULT_2026_10_20.md`。
从官方15.8GB ZIP用标准Range只提取约164MB必要内容，真实模型784056三角形，
SHA9898fce495a105f0aa3fed9f0cbb181a68b1438c5eab046b62c9ab0f55a73620。
不要再写“没有真实TGS checkpoint”。输入在sources/tgs_real_checkpoint_v1；
结果在experiments/tgs_real_event_v1。没有优化器/训练mask或训练commit证明，只按官方load推理语义。
相机JSON4978x3300为COLMAP大小，实际images_4渲染1245x825由PNG核验，勿再次按原尺寸重建。

固定hash选择8模拟事件×4相机，3appearance，共96保存条件；32组合中19有像素变化、13为0。
原始SH局部峰值0.0477826，但最大全图MAE仅3.85572e-6，不能宣称显著质量下降/提升。
事件2/parent16010在test0尺寸30px、train2内切半径1.80092px，四子在train2全剔除。
独立CPU重算指标/参数/选样通过，degree3修正SH与Python路径差<=1.79e-7。

新数学补充`TGS_PERSPECTIVE_INRADIUS_BOUND.md`证明正深度针孔中点四分
min child inradius>=parent inradius/(1+kappa)，常数锐；1500数值样本及31实景投影核验。
它只给尺寸充分条件，不保证其它剔除/浮点边界，不构成新颖性或训练收益证明。
下一步应审查该锐界先行工作，并在固定新增容量下检验细分选择/拓扑/窗口候选，
不能靠全拒绝细分伪装成方法收益。排序/遮挡尚未因果拆分。

所有新句柄终态：下载64593、CUDA55733、审计89276均exit0；无训练等待。
本轮无完整训练，GGGS旧负结果保持，LaTeX未改，整体科研目标未完成。

## 当前入口（2026-10-20，验证执行完成）：数学证书、18项CUDA、SH缺陷修正

上一执行turn为实质进展，18项前向已exit0；本轮补完3输入SH方向缺陷复现及隔离修正，
两版本系数梯度均与各自解析方向相符。完整结果先读 `TGS_VALIDATION_RESULT_2026_10_20.md`。
math_results/cuda_results/sh_direction_v1原始与修正结果均已保存。有限同内参相机D=100sqrt7
有严格有理区间Farkas证书，且给出至少一视角连续相对L1误差>=0.0068915073的保守下界。
18项CUDA最大参考差5.13334e-6；小父r=1.5px分裂为四个被剔除子；大三角形场变化但总量近守恒。
SH固定版本forward410/backward293把中心误作(z,z,z)；隔离修正颜色误差<=3.18e-8，
仅SH系数梯度相对误差<=7.09e-6。原始归档/GGGS安装/LaTeX未改，未训练。
Windows证据审计已通过：452来源、18保存输出、严格证书及两版本SH记录；见evidence_audit.json。
所有本轮句柄均终态：源准备9712、首次编译59999、原版SH6948、修正版SH3379均exit0；
18项CUDA执行及数学命令已exit0。不要继续等待这些句柄。

不要重启已终止训练或重复18项来代替下一门槛。下一项是来源可核验的真实checkpoint/事件，
按 `TGS_REAL_EVENT_PROTOCOL.md` 冻结选择与因子拆分。现有结果不是实际数据质量/效率收益，
新颖性/独立同行审核未完成，广义目标仍未达成。用户已授权数学+小CUDA；下方分析hold为历史。

## 当前入口（2026-10-18，用户已恢复执行）：数学核验与小规模 CUDA

用户明确回复“恢复数学核验和小规模 CUDA 诊断”。这覆盖下方旧的等待范围状态；
按 `TGS_SUBDIVISION_AUDIT_NEXT.md` 第1–3项执行解析核验、有限透视反例证书、
固定输入的隔离 TGS CUDA 诊断。此前本地 WSL/CUDA 授权继续有效，不再重复询问。
本轮范围不含完整训练、lambda 搜索或真实数据收益结论。整体学术目标仍未完成。
进程、结果与证书将在执行后记录；此条授权记录本身不是验证结果。

## 当前入口（2026-10-18，阻塞审核）：等待执行范围选择

已重新读取当前待办、证明限制与GGGS负结果。上一轮为实质进展；本轮未增加研究结果。
自发出异步范围问题的goal turn、随后论文来源核查turn，到本turn，同一“先不进行尝试”
限制和未回答状态已连续出现三轮。前两轮仍完成了授权范围内的静态工作，不能将其误记为
空转；现在下一必要步骤是已列明的数学证书和渲染核验，依赖用户选择恢复范围。

目标尚未达成：TGS推导缺独立/数值核验、真实事件证据和质量/效率验证，GGGS原pilot失败。
当前不继续增写未验证结论或重试已读取的来源来替代这些必要步骤。按阻塞审核申请将目标
设为blocked，实际状态以goal工具返回为准；完整研究目标与待办保留，不记为complete/paused。

恢复入口：用户回答已有异步问题后按其范围执行 `TGS_SUBDIVISION_AUDIT_NEXT.md` 第1–3项；
“仅数学”不得进入CUDA，“数学+小规模CUDA”不包含完整训练。此前WSL/CUDA权限授权有效。
自动goal消息不能替代该选择；不要重复问题或重启任何终态进程。

## 当前入口（2026-10-18，论文来源补齐）：窗口定义确认与高sigma极限澄清

上一轮完成针孔推广/文献落盘，是实质进展。本轮在尚未收到执行范围选择时，继续只读来源
核查，取得作者站10页 `sources/tgs_conference_2026/held26triangle.pdf`，session23967 exit0；
不要再等待该句柄。SHA256 9c7ffb52f1040dfba78a9a47292b68d359c2d8248d9b1a1d238bf46c0787a85a。
本地pypdf文字读取+pypdfium2页面预览并目视3–5页，与固定代码核对核心窗口/四分规则。
详见同目录READING_NOTE.md和receipt.json。作者稿不等于已与IEEE出版社版本逐字节比对。

`TGS_SUBDIVISION_MOMENTS.md` 第2节增加极限澄清：固定峰值I_sigma的积分趋零，
I_sigma dx弱收敛到零测度；归一化密度才趋delta_s，平均距离恰为3J/[A(sigma+3)]。
这是手工数学补充，非实验或新颖性认证。下载/解析公开PDF不属于研究模型的数值验证。
前述“会议相关正文始终未取得”在核心方法段范围已过时；完整版本差异仍未审查。

执行范围问题仍待用户回答，不能自动启动数学核验、CUDA编译、渲染或训练，也不要重复问。
本轮无新活动研究进程，既有GGGS负结果保持，LaTeX/canvas未改，整体目标未完成。

## 当前入口（2026-10-18，中断恢复）：同内参针孔推广与先行工作补齐

上个有执行的goal turn属于实质进展：`TGS_SUBDIVISION_MOMENTS.md` 已保存同内参有限针孔
相机的存在性推广，而文献记录与入口未写完。随后空白中断没有可等待的工具句柄；不要重启
旧训练。本轮已读回证明、修正列向量记号，补齐 `TGS_RELATED_WORK_UPDATE_2026_10_16.md`。

推广使用固定K=I、两组旋转/平移相机，真实透视投影后统一缩放写一般面积比与矩约束；
用非负解的质量界取得紧致性，将假定共享解序列送到已不可行的正交极限。每个视角单独
的严格内部可行解由可逆3x3系统连续延续。它证明存在有限距离、同内参配置的冲突；
没有给具体D0、误差幅度或数据集发生率，仍非独立审核/数值证书/CUDA结果。

新核读的ETS已有边界核/尺寸剔除分析，DETRIS已有视角依赖与重心坐标讨论，Revising
Densification已有中心克隆修正的非全像素保持结论。不能把这些现象当作首次发现。
下一执行包写在 `TGS_SUBDIVISION_AUDIT_NEXT.md`；用户此前“先不进行尝试”限制仍有效。
本轮只改研究Markdown；未启动数学程序、渲染、编译或训练，LaTeX/canvas未改。目标未完成。

2026-10-18 已通过异步问题请求用户选择下一阶段范围：数学+小规模CUDA诊断、仅数学核验、
或继续仅分析。问题源于用户此前“先不进行尝试”的限制，不是再次索要WSL权限；尚未收到
答案。后续自动goal续接不能替代用户选择，也不要重复提问。按到来的答案更新执行范围。

## 当前入口（2026-10-16，多视角补充）：停止扩展 opacity-only 通用修正设想

`TGS_SUBDIVISION_MOMENTS.md` 第6节新增条件性多视角不可行性推导。两个视角均保留世界中点为
像平面中点、同sigma且只调共享非负opacity时，总量+重心同时保持要求对应边长比例gamma相同。
证明由w_i=(1+2t)lambda_i-t相减得到；不依赖子opacity<=1。3-4-5对应顶点反例取父o=1/2，
各视角单独严格可行，但共享解要求t=-1/2。给出同尺度正交相机和不同焦距比例针孔构造；
不能直接推广到固定内参的一般倾斜透视。未执行数值/符号验证、编译或训练，未改生产代码/LaTeX。

本轮完成这个有限的理论补充后应汇报并收束：先保留可审核的零/一阶矩与可行性草稿；下一阶段
若用户恢复尝试，首要任务是识别真实细分事件的窗口/剔除/SH排序/optimizer影响，而非立刻训练
opacity补偿或继续history-gap权重搜索。独立证明审核、新颖性和真实收益均未完成。
OpenReview会议PDF返回验证页、Oxford存档Internal Error，最终版与arxiv v1差异仍未解决。
当前无新启动进程；用户“先不进行尝试”限制仍有效，研究总目标尚未达成。

## 当前入口（2026-10-16，静态审查后）：空间矩与 opacity-only 可行域草稿

先读 `TGS_SUBDIVISION_MOMENTS.md`。上一轮为实质进展；本轮在用户分析阶段限制内继续
推导，未运行数值/符号实验、CUDA或训练。源归档和旧结果保持，LaTeX/canvas未改。

理想连续TGS窗口：透明度重心mu=(3g+sigma*s)/(sigma+3)，一般三角分区的偏移
=sigma/(sigma+3)*(sum A_j*s_j/A-s)。像平面中点四分特例d=3sigma/(4(sigma+3))*(g-s)，
并得L1下界2M||d||/方向宽度。世界中点在一般透视下不是像平面中点，不能泛化特例。
固定形状/同sigma，只调4个子opacity同时保总量与重心的可行条件被化为一个精确区间；
高opacity可能无解。草稿含3-4-5三角形sigma1的精确分数例子：o<=31/34才在[0,1]内可行，
不是运行过的合成实验。独立证明审核、新颖性、真实影响仍未验证。

新增主要先行：2DTS arxiv2506.18575v3（2026-04-04）已含重心坐标窗口、解析积分补偿，
不能把这些基本思路称为新方法。旧v2与最新v3相关段已读，未审其完整源码/性能。
TGS作者站会议PDF本次Internal Error，版本差异未解决。不要自动转入训练或继续lambda搜索。
本轮无活动进程，整体研究目标尚未达成。下方静态审查和30k负结果仍有效。

## 当前入口（2026-10-16，策略讨论后）：TGS 静态审查；用户暂不尝试

用户要求“根据已有工作给出下一步策略，先不进行尝试”，之后要求继续之前任务。
本轮继续的是文献/源码静态审查，未启动 CUDA 编译、渲染诊断或训练。保持分析阶段限制，
不能用 goal 自动续接自行恢复实验。先读 `TGS_SUBDIVISION_STATIC_AUDIT.md`。

实质进展：发现旧 triangle_thin_layer_audit_v1.json 已 pin 主库685ff270/native6d61f4，
最新入口“从未pin”的理解不正确。本次 current main 同一commit，19选定来源195687B及
commit/tree元数据归档 `sources/tgs_subdivision_static_v1/manifest.json`，blob身份全部核对。
首次树SHA假设错误已修正；审批额度失败后同路径恢复；下载60404网络短读exit1，
补缺27807最终exit0，没有任何训练被重启。不要继续等待这两个终态句柄。

解析推导：任意二维非退化三角形的归一化边距u满足area{u>=t}=A(1-t)^2，
所以alpha积分=2oA/[(sigma+1)(sigma+2)]；不相交三角细分复制o/sigma已保持总积分，
但内部新边导致局部重分布。不是CUDA验证、独立新颖性或质量结果。故不能直接做
“总透明度缺失补偿”；下一问题须拆开窗口/采样剔除/中心SH与排序/optimizer状态。
固定源码还有小于1像素内切半径剔除、中心view-z和中心方向SH；实测发生率未知。
旧缓存未找到equilateral_regularizer疑问已解决，固定版loss_utils.py确有函数且返回面积。

MCMC v3已核读近似relocation/中心校正/切片积分及Adam段；TGS+/MeshSplatting相关
方法段已读，完整实现未审查。仅定向文献核对。LaTeX与现有canvas未修改；完整30k负结果
保持，整体目标仍active且未完成。下方入口为历史，不覆盖用户本次分析阶段限制。

## 当前入口（2026-10-16）：完整30k pair结束，GT门槛失败；下一独立问题TGS细分

先读 `HISTORY_GAP_PILOT_RESULT.md`、`HISTORY_GAP_RELATED_WORK_2026_10_16.md`、
`TGS_SUBDIVISION_AUDIT_NEXT.md`。本goal turn是**实质进展**：同一session7120核实完整
16阶段均exit0，不重启旧训练；controller/control/candidate均已terminal。**不要再等或重跑。**
各23000 Adam，共46000；完整状态/逐步相机和sample IDs/最终cp与sidecar/原始评价独立审核
session51212 exit0，见 `results/history_gap_pilot_audit_v1.json`，290 sources/319 inputs。
最终GT overall .3307163221→.3315905070mm，ratio1.0026433，d2s/s2d各ratio>1；
TEST PSNR+.004528dB/SSIM+.0008505；容量+.233%；训练wall+31.98%。geometry三门槛失败，
总pilot_gate=false；不据微小图像增益声称成功，不扩大lambda搜索或直接上多场景训练。
这是一对探索共享warmup，不是统计显著普遍退化；原数学C1/MP/warp核结果仍保留。

只读事后机制检查 session34419 exit0：42train×512pixels×2final模型，
同18344非空ray D²均值.177667541→.093169198，降47.5598%，GT仍未改善。
84个NPZ独立审计 `results/history_gap_posthoc_audit_v1.json` exit0。全loss大部分含empty常数，
code2只tail不是no-root证书；不能声称识别因果机制。原模型SHA不改，无新训练。
审计开发中F64重算F32/假定NCC=23001以及名义图片尺寸错误已按实际source/PNG纠正，
没有改实验判据；9001两臂ncc/geo0与empty-mask提前return相容，未存mask，保留范围限制。

新问题**尚未开始实现或实验**：Triangle Splatting midpoint细分保持面积/区域是否保持
实际alpha/渲染？已读arXiv2505.19175v1 §3.1–3.3和官方main triangle_model.py：
_update_params约592–614行中点四分，复制opacity/sigma；main未pincommit，未编译运行。
下一步先pin官方TGS+native及许可证、对照实际forward窗口/cap/filter/排序，检索TGS+/MeshSplatting/
MCMC守恒densification原文；再固定小型精确公式/实际CUDA对照，不把面积不等于alpha的简单代数当原创。
光学扰动也不自动否定某未定义Markov概率/证明GT损害。见 NEXT完整条件；不能跳过直接训练。

同一LaTeX已新增full-pilot负结果/事后loss分解/先行范围，v44内置编译成功；同一canvas38页展示负结果（仅语法检查）。
生产train/renderer/model和已归档协议/source不改；整体目标仍未完成，用户WSL/CUDA授权继续有效。
下方running句子全是历史，不覆盖本段。

## 当前入口（2026-10-15 full recipe后）：24更新通过，30k配对正在运行

先读 `HISTORY_GAP_FULL_TRAIN_RESULT.md`、`HISTORY_GAP_PILOT_PROTOCOL.md`。
本goal turn为**实质进展**：common6999 CP/sidecar完整digest及42训练池CPU预检通过；
原train.py完整NCC .6/geo .02（depth-normal0）、增密/filter保留，四独立臂各6步共24次Adam。
allgroup computed-zero首完整梯度过预设repeat-noise工程界；首步在增密前保存，不伪称每项梯度穿过参数替换。
初始676277基元+752原Inf opacity，原增密后677291；mask动态读取、Adam前后不变、finite/binding/RNG/相机guards通过。
source/gradient独立audit：63来源检查/46产物，重算首梯度，结果 `results/history_gap_full_smoke_audit_v1.json`。
主full-smoke session90037已exit0，预检1149已exit0。**不要重跑已完成smoke。**

**现有活任务**：`research/experiments/history_gap_pilot_v1`，exec session **7120**。
controller `run_history_gap_pilot.py`已启动，control子PID **401**曾经ps确认Rl+、正常推进；
后续PID/阶段以实际manifest/ps为准，不能拿该旧PID当永久真值。
先poll同一session7120，再查manifest/current_stage、子JSONL与实际PID；观察超时不等于terminal。
固定两臂control/weight.01各从common6999到30000，完后**controller自动完成**mesh+官方GT3seed+TRAIN42/TEST7评价。
完整结果尚未得到，不声称质量/效率改善。control运行期间严禁改其source_sha256中任何文件！
这些已归档sources含new pilot wrapper/runner、旧recipe函数、train/scene/renderer/GS源码与评价函数，
不含本CONTINUATION、LaTeX、canvas或结果说明。即便只是修wrapper也应先等terminal保存失败证据，不可中途改代码。
失败没有自动重试；完成后从真实paired_quality.json独立复算，再决定多种子/多场景/消融，不自动调大范围。
短诊断首区间漏forward/含冷开销，不用于完整速度声明。最终runtime sidecar由新wrapper保存。

同一LaTeX新增完整recipe诊断小节，内置v43编译成功；同一研究canvas新增37页（原有36页保留），仅语法核验。
未开新tab，生产源码/原模型不改。旧147-view GT负结果不变，整个目标仍未完成。用户已授权WSL/CUDA，不重问。
下方为历史入口，新段优先。

## 当前入口（2026-10-15现场adapter后）：真实共享梯度与36个短步完成

优先读 `HISTORY_GAP_ADAPTER_RESULT.md`、`HISTORY_GAP_FULL_TRAIN_NEXT.md`、`results/history_gap_adapter_checkpoint_v1.json`。
本goal turn实质进展：LiveAdapter实际GPU query/AMP/CSR采样→Float64图→原GaussianModel共享Float32参数，
三场景各0000现场source/raw/ID/图像bit对照过，MP loss shared梯度误差≤4.69e-8，块≤7.40e-8。
V1 session89078原all-finite检查遇原opacity+inf停止；V2 session30049误取depth梯度当loss参考失败；
V3 session22299正确loss参考后完成：两disabled/zero/.01各3步×3scene共36更新，optimizer/RNG/finite/原参数恢复过。
原PLY3431/1908/2443个+inf opacity保留为固定opaque坐标，梯度0、mask/值不变，非C1-infinity声明；其他参数/所有grad/Adam有限。
sample164/162/167实际empty0，37现场全空＋另mixed saved-real fixture（97796 exit0）分母/零梯度过。
独立audit45504 exit0：20来源/35产物/实际二进制及回放/scatter/分母/mask/controls/steps，输入额外哈希已记录。
零控制是noise-aware工程界，非bit/统计等价：37 scaling差1.054% vs disabled repeat.678%；zero参数差max8.43e-6。
reg首→末.163931→.138692/.244935→.224625/.123730→.104279，RGB最后稍高于control，不能声称质量改善。
真实诊断wall positive111.31/85.66/64.38ms vs disabled-repeat70.71/47.25/31.30；capture24–42ms。
原LaTeX v42内置编译成功，同一看板36页仅语法检查。生产源码/原PLY不改；本次是fresh-Adam RGB-only memory smoke，非完整训练。

下一步**原完整train.py recipe＋严格42/7＋common6999 checkpoint/runtime sidecar**，按HISTORY_GAP_FULL_TRAIN_NEXT.md先完整状态恢复和zero短步，再配对pilot。
不要继续无止境优化capture而不验证科学效用；现有完整GPU CSR可用于短pilot但成本全计。old49-trained PLY不能作真正heldout起点。
正确MP损失参照为history_gap_raw_v1/reference；world_root_vjp_v1/reference是depth导数，只用其中source/raw/IDs。
整个目标仍未完成，旧147-view GT负结果与所有失败保持。用户已授权WSL/CUDA，不再问。
以下历史入口不覆盖本段。

## 当前入口（2026-10-15 warp后）：loss并行优化完成，下一步训练adapter

优先读 `HISTORY_GAP_WARP_RESULT.md`、`HISTORY_GAP_TRAIN_ADAPTER_PLAN.md`、`results/history_gap_warp_checkpoint_v1.json`。
本goal turn实质进展：独立warp CUDA核、同process串行交替对照、MP原门槛全过、独立产物审核完成。
warp_v1 session99775编译/计时后汇总dict重复键exit1，sourceZIP/traceback保留；
warp_v2 session72993复用同一核重测固定协议exit0。第二批compile_wall是加载时间，非编译耗时。
实际native experiments/history_gap_warp_v1/generated/history_gap_warp_79797068f72e.so，
SHA b95d0172b690670318456986daf3defdecac151386e7a408a313e0c777b27aa1。
完整GPUgeometry/loss/rawback median串行79.11893ms、warp11.99309ms（ratio.151583，6.597倍组件加速）；
仅核72.75982→5.91307ms。10轮交替，前2轮warmup，8次全用于median，非完整训练速度/置信区间。
两版本各原891/51524全部过，64合成、2缺根、12特殊（含513/1025必要最后基元）全过。
原9异常/3零loss保留；MP max整射线9.143e-11、块1.720e-10。invalidrho0拒绝。
独立audit session33454 exit0核111来源/17产物/binary/所有loss梯度、MP引用、exact长射线质量与时序。
原LaTeX v41内置编译成功；同一看板35页仅语法检查；没有训练启动、生产源码/模型不改，旧GT失败不变。

**下一步真实GPU采样/共享ID训练regularizer adapter**。不要再优化串行核或重新跑已完成对照。
按HISTORY_GAP_TRAIN_ADAPTER_PLAN.md读取实际capture/query/AMP API；新loss调用graph `history_gap_gpu_v2.Engine`、warp native。
固定camera/采样，raw gather到原Float32共享参数，避免CPU per-iter copy；先disabled/零权重、render输出、scatter/optimizer绑定、短步与完整迭代成本。
非finite/empty处理明确，失败不传播NaN或奖0；之后才预设配对训练与GT/heldout/容量门槛。
用户已授权WSL/CUDA，不重问；整体目标未完成。以下历史入口不覆盖本段。

## 当前入口（2026-10-15再后续）：GPU loss与完整raw伴随已完成

先读 `HISTORY_GAP_GPU_RESULT.md`、`HISTORY_GAP_GPU_NEXT.md`、`results/history_gap_gpu_checkpoint_v1.json`。
本轮为实质进展：CUDA独立分支/内部求根/系数伴随+GPU raw链实现，原891/51524节点全部过MP原门槛。
最新graph `history_gap_gpu_v2.Engine` 继承 `history_gap_gpu.py`，CUDA源 `history_gap_gpu.cu` 已冻结不要改，下一核复制新文件。
V2 GPU整射线max9.143e-11、各块max1.720e-10，877内部/11tail/3zero，无丢失，仍不是区间认证符号。
初版新32只有28过，4理论零块舍入失败已保留；V2叉积能量+kernel0精确零导数后原32回归及seed13019另32全部过。
最终64合成max4.583e-14，两个真实无根更新/10特殊coefficient测试全过。合成非新真实场景，不声称GT。
V1 session34235 exit0；V2 session14507 exit0；最终confirm_v4 session66333 exit0；profile和独立审核exit0。
confirm_v1 JSON失败、v2最终console失败，v3从已保存结果恢复28/32，不能说这些失败不存在；原数组/manifest/源码ZIP保留。
原LaTeX v40内置编译成功；同一看板34页仅语法检查。独立results/history_gap_gpu_audit_v1.json核全source/binary/数组/MP引用。
V2 native路径experiments/history_gap_gpu_v2/generated/history_gap_d6be97bb45e5.so，SHA406b06756bb62361cb4a57777bbba3b2815c5ef85661bee933c6dfe2c15008e4。

下一步**先warp并行化串行loss/seed核**：V1拆分warm72.699ms在核；V2完整891 forward+rawback79.067ms，成本尚高。
保持所有891/64/特殊样本及原门槛，不再调block分母或丢失败。串行binary保留参照，交替同process计核及完整raw链成本。
之后GPU训练adapter，固定采样/共享真实ID，零权重同状态对照，处理empty/nonfinite；不每步做CPU capture/transfer/冷启动。
未启动训练，原模型和生产源码未改。旧147视角GT失败仍不变，整体目标未完成。用户已授权WSL/CUDA，不重问。
以下“当前/下一步”为历史；本段优先。

## 当前入口（2026-10-15后续）：完整raw链、真实共享更新、无根时C1延拓

当前优先读 `HISTORY_GAP_RAW_RESULT.md` 与 `results/history_gap_raw_checkpoint_v1.json`。
上一goal turn是实质进展；本turn也完成五个实验及独立审核，不是等待/状态复述。
raw42177、raw_micro89864、shared33823、shared_micro23825、extension77335均exit0。
factor raw891/891过MP140/240，max整射线4.54e-11、各块≤2.54e-10；dense890/891失败。
原方向35/42，7失败独立MP微步全过；保留原失败，不把缩步当成新确认。
共享真实24007 Gaussian，6278重复，max137次，zero-control891 bits相同。
full/geometry步长.001降低固定平均loss约.08392%，不是GT；.01各1无根，原均值未定义保留。
共享原FD约5.35%失败；全891后续h1e-9误差8.58e-6过，h1e-7和1e-11仍失败，主点MP支持局部截断+舍入。

已证明/实现/核验C1无根延拓：Ω>log2保留原loss，否则L=(log4-Ω)^2。
质量边界H(maxp)<log2邻域使有根侧也等于该公式，故整个正系数域C1。
16随机/16双侧及精确单层通过；两个原缺根更新在MP确属Ω<log2，延拓loss及raw FD误差≤2.4e-14。
helpers `history_gap_extension.py`、`history_gap_raw.py`、`world_root_vjp_v2.py` 及各协议固定在experiment来源ZIP。
原LaTeX原位v39编译成功；同一看板新增33页，语法检查而非SDK/runtime。原GT失败仍不变。

下一步是GPU批处理loss及伴随，包含新无根延拓，保持source-preserving raw链/真实共享IDs；
先全891和新确认输入对MP/CPU梯度及完整成本，再决定配对训练，不绕过原失败或把本地下降等同重建改善。
可从动态root source/缓存诊断借鉴，但不依赖粗公共F32中心（原仅264/891），要独立精细root和log-alpha稳定密度权重。
不改/覆盖既有源、模型、失败或GT结果。当前无新训练在跑，用户已授权WSL/CUDA无需再问。
以下历史入口/“下一步”不覆盖本段；整体研究目标未完成。

## 当前入口（2026-10-15）：平方历史差证明与稳定CPU系数梯度核验完成

先读 `HISTORY_GAP_MATH_RESULT.md` 和 `results/history_gap_math_checkpoint_v1.json`。
history_gap_math_v3 complete/exit0(session57182)，891射线/51524节点全部过原梯度门槛，
独立55来源/53产物及全数组审核通过；原9异常保留。v1 NaN停止、v2的108块失败及原源码保留。
数学上L=log(V/T)²在所述连续系数开域C1，奇异median上梯度0；一般不C2。
分离薄表面大loss但指数小梯度反例已证明/MP核验，因此不能据此进入训练。
稳定Float64整射线max相对L2为5.694e-12，各块≤1.151e-11；包含额外log-alpha输入。
公共F32深度中心仅264/891过，非完整raw链/CUDA/训练证据。
原LaTeX原位v38内置编译成功；既有看板新增平方历史差页，语法检查无错，非SDK/runtime验证。

此前完整GT阶段（FROZEN_CERTIFIED_MESH_RESULT.md）已完成147视角/18官方seed评估，
数值修正后的三scene均值overall仅约0.0006245%下降，且各scene s2d都变差，质量门槛失败。
勿重复该批或把小变化说成显著进步。其完整快照见after_frozen_certified_mesh_v1.zip。

下一步：先检验此loss真实raw参数链的方向导数及受约束小步下降覆盖，尤其高loss弱梯度区域；
通过才实现CUDA/配对训练，否则保留数学补充并改查几何一致性目标。总目标未达成。
用户已允许本地WSL/CUDA，不重新询问；当前三次math主句柄均终态，无训练启动。
出现聊天断流先读终态/同一进程，不重启旧任务。以下各节为历史记录，其“最新/下一步”不覆盖本入口。

2026-10-14。研究目标尚未完成；真实质量/效率与学术新颖性尚未证实。先查实际进程和产物，禁止因聊天中断重启已有训练。

## 最新：设备原参数预算、自主求根与条件回退固定开发通过

`DEVICE_ROOT_RESULT.md`。device_budget_v1 complete/exit0(session54256)：7322项两路径58576系数包围全过，110×2 mass分类全过。
device_root_v1 complete/exit0(session60003)：不输入CPU候选/bracket/MP根，106finite全认证（low76/high30），4no-root全正确。
max误差1.361838888e−5 camera-ray，门槛仍1e−3；已观测fixed seed属于开发，不称独立新确认/像素发生率。
独立verify_device_root MP240 exit0：106新端点对、4408使用中系数、精确宽度及4无根全过，56+14源、9+7产物、10+11generated核验。
event预算0.441504ms/7322双路径，求根+证书705.173584ms/110模型，合并705.568237ms；不含host transfer/full renderer，无加速采用结论。
原生产训练未改，当前句柄全终态。下一步先固定单一优化：预计算严格级数的1/k有理倒数区间，替代循环动态double division；同一110模型核验输出与event，不改目标/样本。之后完整24视角成本/范围/选择及独立backward，通过才做多场景训练/留出/GT。
原LaTeX原位v27编译成功，hash见results/latex_compile_status_v27.json。用户已允许WSL/CUDA，不重复询问授权。

## 当前：条件精度 CPU 证书与锁定确认通过，尚待 GPU 实现

**后续最新：** `STRICT_INTERVAL_RESULT.md`。共享数学核心不用standard exp/log/log1p/expm1，因为CUDA12.8说明其ULP边界不是保证界。
严格range reduction/series与ln2有理常量核验完成；host 774scalar+212端点全过，strict_interval_cuda_v1 complete/exit0(session25852)，212端点全过。
GPU中位event222.9916504ms/212端点，包括输出allocation/fill、kernel，不含原参数预算/选择/求根/transfer，不能声称效率收益。
GPU端点使用CPU保存区间/bracket；完整device预算/mass/offset/回退/求根未实现，GUARDED_CUDA_PROTOCOL整体仍未过。
独立strict_interval_audit_v1核验两组archive/产物与generated、212scaled参照、精确常量；host后未执行CUDAwrapper追加same-device guard的已知源漂移已记录，header/cpp/exe一致。
两次只读WSL自动审批timeout后，用户明确允许本地WSL/CUDA诊断，实际compile/run成功；权限阻碍已解除，不重复询问授权。
下一步便宜充分认证与device预算，严格核心作为难点参照，完整成本对照后再扩张。当前上述诊断终态，原LaTeX原位v26内置编译成功，hash见results/latex_compile_status_v26.json。

`GUARDED_PRECISION_RESULT.md`，固定协议 `GUARDED_PRECISION_PROTOCOL.md`。
guarded_precision_v2 complete/exit0(session64382)，全部旧7010项区间核验通过；18旧+88新有限根全有证书，4无根全分类正确。
低路径80/110（76有限+4无根），30有限根回退；最大误差1.361841855e−5 camera-ray单位，非GT/world mm。
独立MP240 audit complete/exit0(session50127)：106对端点符号/精确有理宽度、4408系数、4无根、56源archive/9产物均通过。
v1 complete/exit0(session37988)保留，81个括区宽度因端点舍入超2e−3最多6.67911e−16；v2仅向内舍入并保存完整残差区间，输入和门槛不改。
53位CPU区间无界指数不是IEEE FP64 GPU；预算/精确偏移/指数尾部符号尚需device实现，不声称完整renderer/backward/效率/GT或新颖性。
下一步先锁定GPU尾部符号和预算策略、实测完整成本；通过才做24视角整图、梯度及跨场景训练与留出评价。不再筛选缓存组合。
当前上述三CPU句柄均终态，无新训练启动；原LaTeX原位v25内置编译成功。

## 前一阶段：一致QR主候选拒绝；合法F32世界的根系数别名证书完成

`COHERENT_FACTOR_RESULT.md`。coherent_factor_v2 complete/exit0(session91229)，7010项/18根模型×3臂全保留。
固定primary=QR_mixed(6F32 U+3F64 mean，48B)在9真实异常根全过，max4.99316e−5 ray；
但旧4096局部失败12、9临界双层根失败5，整体FAILED。不开始新seed确认、整图计时或训练。
QR64缓存仍将系数转F32，临界根失败4，不换主候选宣称通过。
v1参照c−b²/a相减断言失败/session16490 exit1已保存；v2仅改参照为残差二次型，源码archive保留。
9个quaternion inverse符号身份通过；coherent_factor_audit_v1核验42源/archive、12产物、21030误差向量和54根记录。
两个合法F32世界输入给相同正确舍入的F32(p,rho,alpha)，有向区间证书r_A<.512/r_B>1.488，
故仅依据这个系数接口的任意单输出统一根误差下界>.488。只限制该接口，不限制原世界数据/full metadata，非完整renderer或GT结论。
下一项为质量裕量不足的条件精度/误差区间及薄层联合极限，不继续手调缓存组合。所有当前CPU句柄已终态。
原LaTeX原位v24内置编译成功，最新源hash见results/latex_compile_status_v24.json；现有研究视图v2语法检查通过并同步。
首次UI数学比较符号语法错误及stage/guard/check均保留，未写入用户视图；检查器rows24为历史常数，非本页数据项数，不代表SDK/runtime核验。

## 最新：全部9个异常点的真实采集、独立根及CUDA重放完成

**后续当前已完成（2026-10-14）：** `EXACT_OUTLIER_COEFFICIENT_RESULT.md`。
全9×8混合模型complete：有效alpha-only复现大差，全替换与alpha-only差≤9.58250e−7 ray；
peak+rho-only≤9.58099e−7。不是单独训练opacity参数的归因。
世界参照world_v1 complete：772原PLY字节核验，MP80/140两构造、36根复查通过；
raw-physical/saved-activated/共同co.w三参考均factor更近8/9（packet更近case0），非GT或一般排名。
gap_v1与独立MP180审计complete：18无内部peak区间/126点检查，case1/2/5两臂反侧充分界，6项inconclusive保留。
当前原LaTeX原位v23编译成功，源hash以`results/latex_compile_status_v23.json`为准；现有研究视图新增有效alpha/world参照页，语法检查通过但不声称SDK/runtime完整核验。
原world任务在启动前因协作额度失败；主代理确认无活process后接手session15996，已exit0。两个audit亦exit0。
旧会话中断不得解释成CUDA死锁，也不恢复/重跑已完成mix。
当前下一项为一致precision-factor能量/相机均值构造与条件误差界，再用锁定旧输入和新构造确认；不扩大失败packet训练。

- `exact_outlier_capture_v3` complete/exit0：8受影响视角全图旧输出位模式重放通过，17581候选/772 accepted保存。
  固定新rebuilt binary，历史binary环境漂移失败保留；不声称历史内部状态恢复。
- `exact_outlier_root_v1.json`：18/18保存F32系数均精确判定有限根。最坏点观测−.18208014 ray、
  表示根差−.18209892 ray、余项+.00001878635。全9余项绝对最大8.9645251e−5 ray，
  相对观测差最大2.7124813%。不是false finite，也不能归为单纯F32求根乘积错误。
- `exact_outlier_solver_v2` complete/exit0：18/18原depth/slope位重放；FP64 log-T仍有大差。
  原实际MIN_TRANSMITTANCE=.45、SAMPLE_RANGE=.4、SPLIT8/5轮；v1 .5 gate勘误和归档保留。
- `EXACT_OUTLIER_RESULT.md`为本轮汇总。原LaTeX原位v22编译成功。当前这些执行句柄均已终态，
  不因聊天恢复重跑。用户要求现有步骤后先简报，本轮先收尾与快照，不启动额外训练。
- 后续先锁定全部9点/772项做alpha/peak/rho混合机制分离，再独立raw PLY高精度参照。
  两F32模型都不是真值声明；无GT质量、普遍频率、因果或新颖性结论。首版完整图像/2.6倍耗时失败保持。

## 上阶段（2026-10-14）：全24视角否定首版exact packet；不扩大该实现

- `exact_perspective_smoke_v1` complete/exit0，单图原门槛通过；只替换median T内系数，
  legacy affine selection/RGB/normal/init保持，grad/sample/integrate/backward显式禁止。
- `frozen_exact_render_v1` complete/exit0（句柄59928已终态），3场景×8视角、10834488像素全部保存。
  三臂全finite，RGB/alpha/normal全24逐位保持，B/C有效mask全同，p99 depth差至多1.90735e-6。
  **8/24视角失败，共9个像素depth差>1e-3，最大.1796405315；数值门槛FAILED。**
  packet/factor GPU时间比GM2.6001822/max2.6573840，wall GM2.5999135；效率门槛FAILED。
  不做GT改善解释或扩大训练；全图结果不能被1992局部或单图smoke通过替代。
- report_v1重放17源/archive、50产物及全图/计时算术；assets_v1核验3binary、51生成文件、
  652原源及3原PLY。`FROZEN_EXACT_RENDER_RESULT.md`保存具体门槛/异常位置和限制。
- `EXACT_PERSPECTIVE_BACKWARD_AUDIT.md`、chain_v1完成：局部因子VJP96构造/1536分量及6数学root通过；
  无全CUDA/NCC/normal后向结论。`TRIANGLE_THIN_LAYER_AUDIT.md`补齐TGS原文与固定代码，
  sigma锐度不等于射线厚度；分位敏感性有明确先行研究，薄层双裕量新颖性仍未证实。
- 下一安全动作：对全部9个超阈值像素作明确事后失败机制重建，读真实候选/前缀与系数，
  用高精度根区分表示、条件数与FP32/有限bracket；不得凭割线斜率小就宣称已找到因果机制。

## 上一阶段：精度候选拒绝，实际CUDA局部前向通过

- `packet_input_protection_v1` 64组合×2120输入complete；全部6360旧FP32系数逐位重放一致。
  最小通过mask8/21仅属选择结果，不能认定稳定。锁定候选的独立seed13017/4096同分布构造
  `packet_input_confirmation_v1` complete：mask8失败13、mask21失败11；FP32基线失败21、
  rounded+double失败11、全double失败0。因此两个保护候选均拒绝，不继续搜组合。
  `PACKET_INPUT_PROTECTION_RESULT.md` 和 report_v1完整保存；全部误差重放，24失败并集80位复核分类不变。
- `PACKET_CONDITIONING_THEORY.md`：有限den/cross/peak/width/energy界与条件logT/median界，
  2120数值检查、5符号身份通过；不是全场景中位数证书。当前直接round qmean的8输入布局存在
  同FP32字节、合法SPD的双点反例，接受区间间隔下界约4.92775e-7；仅此表示的统一相对精度限制。
- `production_packet_v3` complete、exit0：复制完整生产preprocess CUDA内核的独立扩展，
  同1992真实输入含46clip、0cull，production UV+CUDA FP32decode全通过原局部门槛。
  report_v1重放5976个输出逐位一致，额外真实CUDA den/det/energy无效0；3原PLY哈希核验。
  非已安装renderer二进制/完整backward；旧合成失败不撤销。v1/v2构建环境错误保留，未产生科学结果。
- 定向贡献审计 `CONTRIBUTION_AUDIT_2026_10_13.md`：IFT/exact ray/Schur等与已有工作重合明确。
  薄层两种裕量及梯度反例可能是更实质的理论候选，但新颖性未证实；Triangle原论文技术核验尚缺。
- 用户要求完成现有步骤后先简报，当前不展开新训练或新分支。下一计划：完整CUDA/VJP及真实成本，
  再预登记冻结模型affine/exact packet/optimized direct factor三臂GT质量/覆盖率/成本对照；
  未过门槛即不扩大训练。报告须同时说明已有留出PSNR仅+.0901dB且SSIM下降，目标仍未实现。

## 上一阶段记录：输入精度候选被独立构造否定；生产源码CUDA探针完成

- `packet_input_protection_v1` complete：64组合覆盖1992真实+128构造，旧FP32基线6360系数逐位一致。
  开发集包含关系最小通过组合为mask8(qmean)和mask21(plane/conic/camera_geometry)。
  保护精度的通过并不单调，不能把开发集选择当稳定性证明。
- `packet_input_confirmation_v1` complete：锁定这两个候选后，用新seed13017生成4096同分布构造。
  FP32基线21失败、all-rounded double decode11失败、mask8有13失败、mask21有11失败、全double0失败。
  两候选均拒绝，不重新搜索组合或放宽旧门槛。`PACKET_INPUT_PROTECTION_RESULT.md`记录完整结果。
  `packet_input_protection_report_v1.json`重放135680+20480次输出误差算术，24个失败并集独立80位重算后分类不变。
- `packet_conditioning_v1` complete：5符号身份、2120有限扰动界数值检查通过；只有双点量化反例使用区间证书。
  同一FP32原型packet可对应两个合法真峰位，原1e-5接受区间分离；仅限制该表示/输入域的统一保证。
  另给出条件性logT和中位深度误差界，尚无真实全场景深度证书或质量收益。
  见 `PACKET_CONDITIONING_THEORY.md`；不能扩写为所有CUDA metadata布局都不可能。
- `production_packet_v3` complete、实际句柄66454退出0：逐字提取生产预处理函数构建独立CUDA探针，
  同1992输入(含46clip)全部保留、cull0；UV恢复的FP64/NumPy FP32/Torch CUDA FP32求值均通过局部门槛。
  CUDA求值max peak/width相对误差5.5390e-7/1.2978e-6、alpha绝对1.1817e-5。
  **是复制生产源码的独立扩展，不是已安装renderer二进制或融合kernel；无backward/GT/速度结论。**
  采集时CUDA内部合法性标记沿用NumPy诊断；后续report_v1已补齐真实GPU内部量与全部产物复核。
  v1缺ninja PATH、v2误选conda CUDA导致编译失败，均保存；v3按已有CUDA12.8/gcc11配置运行成功。
  不能把构建失败当科学失败，也不覆盖旧记录。
- `CONTRIBUTION_AUDIT_2026_10_13.md`已完成定向原文审计：一般透视正确求值和Schur分解非新概念；
  薄层深度/梯度所需不同裕量是候选理论内容，新颖性仍未确立。Triangle原论文/实现尚未完整核验。
  当前应收束局部组合搜索，先决定完整冻结模型A affine/B packet/C直接因子exact对照的可行性，
  完成必要CUDA前后向与支持域检查后再执行GT质量/成本对照；不自动开始新训练。
- 原LaTeX保持v19成功版本；本节新增结果尚未写入该文档，不声称已编译最新研究内容。

## 上一阶段：含中心裁剪的packet身份与全部1992输入核验完成

- 新 `clipped_packet_v1` complete、exit0，旧v1源码/结果不改。
  `clipped_packet.py`恢复qbar基底的Schur参数，Gram行列式用非负分解求能量。
  6个一般SymPy身份与seed13016的128构造通过，其中112触发clip。
  同1992个selected输入全部通过，包括旧46个域外输入；无换样本、无fallback。
  `results/clipped_packet_report_v1.json`从原PLY独立80位重算全部1992，含46clip，
  具体误差、来源/产物计数以实际JSON为准；仍非CUDA/VJP/质量或速度结论。
- 本轮最初WSL健康检查因自动审批服务403未执行；原样一次重试已获审批并exit0，
  perspective_reparam_health_v2仍complete+空匹配进程。不能把服务故障归因为GPU。
  Windows原生128数值检查曾跳过符号，overall_gate_pass=false，不能当完整通过；
  当前WSL版已完成符号检查。旧native源码/结果zip保存了符号Q构造括号修复前版本，
  该旧版符号从未执行，不谎称发生过符号测试失败。
- **clipped_packet_precision_v1已complete、exit0，整体开发门槛FAILED。**
  全1992真实rounded FP32 packet通过；全71712 raw FP64梯度分量通过，
  三输出max scaled errors=(4.19e−7,1.23e−7,2.40e−7)<1e−4；无clip边界输入。
  128合成中case2峰位相对5.29059e−5>原1e−5，不能删例或改判通过。
  `CLIPPED_PACKET_RESULT.md` / precision_report_v1：233源/archive、51产物、
  全行/梯度算术重放；80位decode已rounded packet仍峰位误差1.71395e−5。
  单纯提高求值精度不足；该例真alpha≈2.3005e−5712，物理相关性有限，但原失败保持。
  下一步研究输入精度保护/相关性误差界，并另外核验真实生产FP32预处理与CUDA；不训练。
  两AD图共用raw变换/库、逐输入独立复制，不证明共享基元atomic或全renderer反向。
- 当前LaTeX已原位补入完整clip证明、全覆盖复核及保留FP32负例，内置编译v19成功；
  hash见 `results/latex_compile_status_v19.json`。不打开替代文档或新标签。

## 上一阶段：匹配因素分离与unclipped条件性packet

- **matched_ray_factors_v1 complete、exit0，不重跑。** 24投影与24射线数组bitwise匹配原CPU，
  全192个3因素混合根和所有路径中间根有限。69源/archive、50产物、3PLY、24membership核验；
  原log_fields独立重解192根，最大差2.6823e−13。所有路径总和误差<=4.34e−19。
  `MATCHED_RAY_FACTOR_RESULT.md`：典型eligibility影响大于width，尾部peak变化突出，
  中位相消29.1519%；只是固定选择射线、Shapley/路径分配，非因果或GT质量。
- **perspective_reparam_v1 complete、exit0；独立80位原参数检查exit0。**
  `PERSPECTIVE_REPARAM_RESULT.md` / `THEORY.md`：unclipped、kernel0、P/Sigma互逆条件下，
  用现有affine plane线性项、t0、s0、conic能量与相机几何精确恢复世界ray p/sigma/alpha。
  9精确符号身份和64个预定SPD构造通过。原24射线1992 selected输入均保留，
  **1946适用全部通过；46个clipped center域外单列，不能计为通过。**
  124源/archive、50产物、3PLY与全部分类核验；全部1946原PLY输入独立80位因子空间重算，
  最大scaled errors(p,width,alpha)=(4.99e−15,1.66e−15,2.63e−12)，未改1e−8/floor1e−12门槛。
  不据参数差宣称图像改善；HTGS/3DGUT已有一般透视正确射线评估。
- 下一步：先一般化到46个clip域外输入（packet可能仍编码完整P的Schur分解，需另证），
  再核验FP32/极端数值、CUDA forward/VJP和实际开销；通过后固定模型质量对照，再决定训练。
  当前主生产/旧CPU/旧负结果保持，不把局部等价当整体方法或新颖性已成立。
- 当前LaTeX已原位补入完整证明、匹配因素和全部域限制，内置编译v18成功，
  source hash见 `results/latex_compile_status_v18.json`。没有打开替代文档或新标签。

## 上一阶段：单法向兼容性补充与24真实参数证书完成

- **`directional_field_world_v1` complete；独立重放exit0，不要重跑。**
  `DIRECTIONAL_FIELD_RESULT.md`/`THEORY.md`：固定点绝对衰减和能写成单向量投影
  当且仅当所有梯度共线；指定product-v梯度还需同向。最优一致方向误差>=sqrt(lambda2)，
  合法Gaussian构造达到下界。双基元cross²/trace证书在增加其他基元后仍成立。
  精确符号与64预定数值见证通过；随机样本不是sup/inf的解析证明。
- 原scan37/55/63各8预定回投影点，三个完整模型全部基元进入名义计算。
  **24/24严格正证书；144独立空间导数分量落入80/100位区间。**
  原13源/archive、26产物、3PLY与48raw基元、24精确点、24有理数不等式复核通过。
  名义谱下界/sum||b_i|| median=.06790113，非区间相对界、非像素错误率；
  最小绝对证书3.4193e−10，非零不代表可见效应。所有指数underflow计数保留。
  点不是GT表面或真正世界模型精确median；世界场不等于生产per-view affine图。
  mpmath.iv仍实验性；有理数与高精度交叉核验不等于证明助手认证。
- 已发现明确凸几何重叠：K是zonotope支持函数，单法向对应线段；
  Siegel2023、Torres2024已有一般逼近理论。GGGS近侧条件与Wrapping方向问题均已存在。
  只声称补充适用条件、下界和证据，不声称原创问题或已获学术成果。
  不直接对全模型加单法向惩罚，也不因24点正证书就训练。
- **同24条射线的world/affine检查已complete、exit0。** `WORLD_AFFINE_RAY_RESULT.md`：
  全部世界相机中位根有限，保留相机起点T_minus(0)与所有指数underflow。
  点态/射线logV最大差1.97509e−13，attenuation最大比例差8.46938e−13；
  根最大log residual3.03029e−11。相对旧深度的绝对位移中位 .01262537%，
  最大2.69459373%（scan37/cam0041，旧6.089402428→world6.253487084）。
  40源/archive、26产物、3PLY与24行/汇总算术重放通过；非独立全根重算或区间根认证。
  大方向衰减差不自动导致大中位深度位移；当前仍混合affine、裁剪/选择/clamp等因素。
  下一步做匹配基元/规则的因素分离及GT/跨视图误差联系；不把此差当质量收益。
- 当前LaTeX原位v17成功，哈希见 `results/latex_compile_status_v17.json`。
  holdout健康v4、directional健康v1均terminal+空匹配进程。
  新 `results/thread_stream_diagnosis_v2.json`确认两轮真实错误为
  `stream disconnected before completion`，不能再仅说应用未暴露错误。
  旧父队列暂停是另一项已确认原因；无证据证明训练完成导致断流或CUDA死锁。

## 上一阶段：薄层与区间定理完成；真实射线否定统一区间替代

- **`quantile_band_frozen_v1` complete、exit0；不要重复运行。** 原3072条均匀射线全部保留，
  2621单根有限，2614区间有限。全部主数值检查及24条预定独立前向/差分通过。
  2051/2614（78.4621%）触发假想薄层混合代理守护，统一替代开发门槛失败；不训练。
  深度偏移>.1 median(width)有204/2614（7.80413%），scan37自身11.4253%。
  band/median opacity梯度范数比最大267.366，不能从理想反例声称普遍缩小梯度。
  `QUANTILE_BAND_FROZEN_RESULT.md`与全部NPZ/JSON/sourcezip保留。
  额外q空间全VJP复核未收敛exit1，actual源码/方案/日志archive保留。
  v2独立product-T全分量前向差分通过（位置最大6.50e−8、opacity最大1.593e−10），
  13源/archive、7产物、3PLY和全部NPZ行对应/汇总独立核验。
  不把v2成功当v1积分已修复。两个复核进程均已退出，见frozen_report_v2.json。

- **薄层极限已形成定理与两类反例。** `THIN_QUANTILE_LIMIT.md`、
  `results/thin_quantile_limit_v2.json`：严格平台跨越保证深度极限，额外内部峰值裕量
  保证位置Jacobian one-hot极限。三层例每个有限厚度根均正则，但位置权重恒为
  (.5,0,.5)、中层opacity导数指数发散；平台等号例深度恒在两层之间。
  精确符号、7个200位临界例、6个平台例、20个非临界例通过；9个解析域排除输入保留。
- **分位区间均值定理与CPU原型已完成。** `QUANTILE_BAND_RESULT.md`：
  空间clipped-T积分使泛函C1，梯度不除根斜率；位置偏导非负且和1，opacity有条件界。
  薄层深度/位置权重可混合表面，并给出避开区间端点平台时的opacity梯度极限。
  `results/quantile_band_v2.json`：7临界族、1单层奇点、12平台/区间例，
  60位差分按原门槛通过。v1小分量double FD失败源码/日志封存，不放宽门槛。
  非对称区间手算−1/3符号错误已勘误为+1/3；原方案字节保持，必须连同勘误阅读。
  区间分位均值统计量已有先例，不声称发明；模型应用新颖性仍待核验。
  `figures/thin_quantile_evidence_v1.png/.svg`仅为合成理想模型科研证据图。
  当前LaTeX原位补入完整证明、真实冻结负结果与额外积分失败，最新编译v15，哈希见对应JSON。
  健康快照v3再次确认holdout complete、匹配进程为空；不要重训。

- **`factor_holdout55_v2` complete：全部19阶段exit0，真实进程已退出。**
  `results/factor_holdout55_quality_report_v1.json` 独立核验484源码、32生成文件、数据/标定、
  新共同状态、实际模型、GT、每图指标和按ID配准。不要重启该队列。
  TEST PSNR +0.090075357 dB（未到+.30），TEST SSIM −0.000304571（未到0），两项门槛失败。
  mean overall 0.332269708→0.331238519 mm（改善.310347%），基元929179→957753（+3.07519%）。
  TRAIN PSNR +0.421707653 dB，不可代替TEST。新RGB非劣/容量守护通过不改变原几何门槛失败。
  见 `FACTOR_HOLDOUT_RESULT.md`、`FACTOR_HOLDOUT_PROTOCOL.md`；不自动扩大此RGB训练。
  新warmup完全从零，42训练/7测试，初态68b6c35...、首相机0022；smoke和完整恢复一致。
  实际训练/NCC/filter池排除测试，最终每臂仅7次no-grad测试报告。
  SfM初始化可能用全49视角，scan55已用于选题；不是完全未接触的确认性数据集。
  新配准按图像ID映射：中心RMS .0000765214 mm；旧firstN在该分割会错到380.593721 mm。
  旧49全视角结果未改。健康核验 `factor_holdout55_health_v2.json` 为终态静态快照。
- **新增尺度VJP定理及反例已完成并核验。** `PRECISION_SCALE_VJP_AUDIT.md`：
  相同对称精度上游H下，误差分量−2⟨H,Bᵢ⟩，紧算子界2√λmax(M)，可构造上游达到。
  固定非单位四元数族的分量相对误差按δ²a⁴增长，归一化极限1/4；不是完整梯度定理。
  `results/precision_scale_vjp_defect_v4.json`：精确SymPy、64独立AD通过，全部164输入保留。
  H=I保存样本整体向量最大相对误差仅7.045e−7；单分量比785.079，不混用两者。
  v1展开错误和v2/v3未化简极限比较失败均以实际源码/日志zip保留，未放宽门槛。
  已原位补入当前LaTeX，连同留出负结果内置编译v12成功；源哈希见对应JSON。
- **下一步：界定可成立的理论贡献与新的可证伪问题。**
  固定宽度区间替代门槛已失败，不能未经新方案事后收缩区间或删射线。
  学术新颖性仍需核验；任何自适应区间均需包含端点全导数，不自动启动训练。
  当前薄层/区间证明和负开发诊断是可审阅成果，不等于投稿新颖性或显著质量收益。

- **新增独立前向诊断已完成，不改已完成的 factor64 对照。**
  `gram_forward_v1`：全部164保存输入，六输出块984检查通过原1e-4 / floor1e-5门槛；
  四个原conic失败变为0，最大误差.00290671→.0000276635，射线/法向逐位不变。
  标准Gram/Cauchy-Binet行列式避免最后一次大数乘积相减；不声称新代数或普遍稳定性。
  `gram_smoke55_v1` complete：新完整后端forward函数等于新探针、backward逐字节等于factor64；
  20次Adam、共同初态、有限梯度通过，首损失最大归一化漂移1.2822e-5≤1e-4。
  该变体未启动完整质量训练；已完成的factor64仍用原forward，不能把Gram结果归给它。
  见 `GRAM_FORWARD_AUDIT.md`、`GRAM_FORWARD_SMOKE_PROTOCOL.md`、`results/gram_forward_report_v1.json`。
  原位LaTeX已补入Gram恒等式、限制并更新标题摘要，当前编译哈希为 `latex_compile_status_v13.json`。
  原四个前向失败作为控制记录仍保留；仅新Gram变体在该保存集合上通过。
- **`factor_quality55_v1` complete：全部14阶段 exit 0，训练、提取、对齐和评估已完成。**
  `results/factor_quality55_report_v1.json` 核验475源码、共同初态、固定控制、模型、GT和指标字节。
  overall 0.315025878→0.313129859 mm，改善0.601861%，原≥1%门槛失败；s2d改善0.002958633 mm。
  49 TRAIN图像 PSNR +0.532053 dB，SSIM +0.00312850；35视角提高、14下降。
  基元943641→967292（+2.50636%）；没有独立训练重复或留出图像，不能宣称显著泛化提升。
  见 `FACTOR_PILOT_RESULT.md`。不同后端旧控制重复不能当本次FP64误差条。
  exec86385已exit0；截至本次核验，无匹配研究训练/评估进程。进程事实仍需每轮刷新。
  **后续留出实验已完成且失败，见本文件最上方。**全新42/7共同warmup与按ID配准均已核验；不改原失败门槛。
- **旧 `corrected_quality55_v1` 已 superseded_before_candidate；不恢复旧父队列。**
  旧父 PID405 在极端检查失败后被 SIGSTOP，控制子 PID425 继续并完成，随后成为 zombie；
  因父进程暂停，旧 manifest 仍称 running。控制 30k 模型及 integration 核验后才 SIGTERM→SIGCONT
  结束父队列，未启动 corrected64。原 manifest、暂停记录、偏离说明与控制均保留。
  见 `protocol_deviation_factor_v1.json`、`results/training_stall_diagnosis_v1.json`。
  app 摘要有一轮 interrupted，但无具体原因；不能宣称 GPU 死锁或平台故障已定位。
- 新只读检查器 `inspect_research_run.py --run research/experiments/factor_quality55_v1 --out <新json>`：
  在 WSL 中检查 argv、/proc 状态、日志写入年龄、集成产物；不自动重启或杀进程。
  `results/factor_queue_health_v1.json` 为静态存活快照，不是实时监控。
- **原组合 FP64 并非普遍通过。** `world_preprocess_stress_v1` 的全部 44 极端基元（最大轴比34819），
  normal/plane/ray_precision 有12/3/28组超过原 .001 门槛，最坏47.294；4个前向分量也失败。
  全部失败输入与归档保留。旧72真实+48合成通过不能外推到尾部。
- 直接精度因子路径避免依赖精确正交的 inverse→covariance 往返。新证明：
  δ=‖RᵀR−I‖₂，a=smax/smin 时，‖PΣ−I‖₂≤δ[1+(1+δ)a²]；
  标准四元数多项式 δ=4|‖q‖²−1|‖v‖²。保存样本最大残差28.0014。
  `results/precision_roundtrip_defect_v1.json`：9个精确符号分量、164保存样本上界、64非正交AD例通过。
  这是充分上界和标准矩阵微分，不是完整梯度误差界或原创代数声明。
- `factor_preprocess_stress_v1`：完整原44基元、880参数块通过，最大1.68e-5。
  `factor_regular_inputs_v1` 仅输入适配；`factor_preprocess_regular_v1`：全部72真实+48合成，
  2784参数块通过，最大1.20e-4。合计3664，含结构零，不当作独立统计样本。
  两套参照均保存；仍以物理float64图作入选门槛，device值图仅定位舍入。
  新路径不改forward，4个原极端前向失败仍未修复，也未独立核验完整渲染器。
- `factor_preprocess_backend.py` 的3个局部CUDA函数逐字节等于新探针；forward/config原样。
  新20步smoke梯度有限，初态c97bfd...、首相机0048、首损失与完成的legacy64一致。
  控制已在候选和mesh/image质量观测前预选并完成；固定复用，无论后续结果如何都不换基线。
  新候选是公式+因子+精度的组合干预，单项因果归因仍需消融。当前kernel_size=0系数修正理想无效。
- 质量门槛不变：mean overall至少改善1%，mean s2d不增加，TRAIN PSNR不下降超过.2dB。
  一对pilot、evaluator重复、训练视角不等于独立训练重复、统计显著性或新视角结论。
  顺序墙钟仅资源记账。未过门槛则保留负结果，不自动扩展训练。
- `PRECISION_FACTOR_AUDIT.md` / `results/factor_audit_report_v1.json` 保存机制与归档字节复核。
  当前LaTeX原位补入完整证明与限制、平方阶数不可去掉的反例族和相关工作，
  此部分的v9内置编译成功；当前含尺度VJP和留出结论的最新源哈希见 `results/latex_compile_status_v12.json`。
  `roundtrip_sharpness_v1.json`：固定非单位四元数族的残差/δa²极限为1/2+ε+ε²>0。
  这不是完整梯度下界或每个族成员都来自float32规范化的声明。相关定向文献搜索见
  `literature/factor_stability_search_2026-10-12.json`，Faster-GS已有非解析opacity梯度讨论；新颖性未证实。
  `factor_audit_report_v1.json` 为此前v8编译与训练中快照；当前文档以v12编译状态为准。
  现有研究视图将更新为静态结果快照，不打开新标签，不自动读取进程。

## 已完成且保留的公式诊断

三个局部链式修正：法向原向量长度、射线精度分子、opacity系数平方因子。
`PREPROCESS_CHAIN_RULE_AUDIT.md`、`preprocess_chain_rules_v1.json`记录推导；
`ray_precision_integrability_v1.json`证明固定κ上游的旧viewport场一般旋度非零，
正定反例√10/45；各向同性旧式产生人工κ log cosθ项。仅针对该分量，不声称完整训练无替代目标。
原 `world_preprocess_synthetic_v2` 与 `world_preprocess_real_v1` 常规检查通过，边界见上述压力失败。
旧smoke v1调用帧失败、v2修复通过均保留。所有原始生产CUDA、冻结恢复器和此前负结果保留。

## 实际终态与入口

- `sampling_batch_v2`：完整结束，单查询分组在六个冻结视角有约 4%–8.5% 完整损失反传收益。
- `sampling_full_loss_v1`：完整结束，追加 18 视角 12/162 梯度组失败；原门槛不变。
- `sampling_gradient_diagnostic_v1`：完整结束，同一图重放揭示世界尺度/旋转梯度不稳定。
- `inverse_gradient_diagnostic_v1`：完整结束，直接逆因子局部代数转换不足以稳定完整反传。
- `precise_gradient_diagnostic_v1`：保留混合类型编译失败；v2 完整结束。
- `precise_development_v1`、`mixed_development_v1`：完整结束，162 组均通过 FP64 参考门槛，
  但冻结完整损失路径没有加速。参照同源、包含 float32 基元累加，不是真导数证明。
- `mixed_training_timing55_v1`：三对交替真实 500 步均结束，预热 50 步，计时 7050–7499。
  同一完整初始状态已核验；墙钟比几何均值 0.990889，未过原加速门槛。
- `adjquad55_v1` 及 `adjquad55_control_repeat_v1`：训练、网格、三 evaluator seed、49 训练图像均结束。
  原候选几何门槛失败；控制重复差更大，不换基线翻转判定。
- `peak_development_v1`：完整结束，梯度通过但无速度收益，未启动完整质量训练。

权威汇总：`sampling_development_report_v1.json`、`world_precision_report_v1.json`、
`mixed_training_timing_report_v1.json`、`control_variation_scan55_v1.json`（都在 results/）。
这些文件逐项核验实际源码归档、输入和汇总；新结果用新版本文件，不覆盖失败记录。
当前不存在上述工作仍在训练中的证据；如后续新启动进程，应检查其实际句柄和 manifest。

## 上轮计划的对应状态（前三项已推进，勿重做）

1. 已建立**独立的单基元 CUDA 世界预处理 VJP 参照**，固定投影/法向上游，
   让其输出三维中心、过滤后尺度和规范化四元数的导数，与 PyTorch float64 自动微分及
   三步长方向有限差分比对。当前仅有真实 CPU/CUDA 前向及有效射线 IFT 参照，
   现已完成上述局部世界解析反传核验；完整渲染仍未独立核验。FP64 重复性通过不能代替这一步。
2. `NORMAL_PREPROCESS_VJP.md` 已确认相对 forward 的局部归一化长度因子遗漏，
   有非奇异正定反例，官方 main 也有同一模式。该问题不解释 geometry-only 的数值波动。
   `affine_normal_identity_v1.json` 给出 n=−Pd/‖Pd‖、‖c‖=1/|nᵀd| 的精确恒等式和
   64 个独立世界 VJP 图核验，可作为简化、修正法向反传的参照。
   先验证正确性，再将修正作为单独 opt-in 变体，避免与精度、分组、NCC 导数修正混淆。
3. 修正已通过所述独立导数核验，两项完整状态分叉的质量诊断均已结束并保留失败；以相同提取、配准、
   evaluator seeds 和明确图像 split 比较。先定门槛，保留原控制和所有负结果。
   方向性效果需要跨场景和独立训练重复；不把训练视角 PSNR 或 evaluator 重复当作新视角/训练重复。

不继续为纯效率扩大当前 mixed1：冻结和真实短程加速门槛均未通过。
仍可作为数值诊断参照，不能宣布质量改善。

## 环境与恢复

- 原始生产训练及 CUDA 源码保持未改。没有 Git 仓库，不声称 commit；实际源文件 zip 快照用于恢复。
- Windows PowerShell；训练在 WSL Ubuntu，Python `/home/someone/miniconda3/envs/uggs/bin/python`，
  CUDA `/usr/local/cuda-12.8`，gcc/g++ 11，`MAX_JOBS=2`，`TORCH_CUDA_ARCH_LIST=12.0`。
- `train_solver_branch.py` / `checkpoint_state.py` 保持冻结，恢复外观对象后再绑定优化器，
  补齐 filter3D、绝对密化统计、RNG、相机顺序及队列；不要直接用原 restore 获得所谓公平分叉。
- 保持当前 `median_depth_note.tex` 编辑器打开，原位编辑后用内置编译器核验。
- 更新现有研究视图，不创建新的编辑器标签；视图不是自动读取进程状态。

工作流溯源：Kassis et al. (2026), Scientific Agent Skills,
https://doi.org/10.48550/arXiv.2609.00065 。
