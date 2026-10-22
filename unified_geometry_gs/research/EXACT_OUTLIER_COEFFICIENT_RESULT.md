# 有效alpha差、跨层平台与世界参照

2026-10-14。本轮全部纯CPU，不训练、不重编译CUDA。目标仍未完成；这是事后异常机制证据，不是GT改善或新颖性确认。

## 72个固定混合模型

`exact_outlier_mix_v1` complete/exit0，全部9点×8组合有限根；120位根、180位端点复查72/72通过。
factor=mask0、packet=mask7，bit0/1/2分别替换alpha/peak/rho；端点18/18吻合旧oracle。

最坏case2：全替换根差−0.182098923724746455 ray，单换alpha根差与它仅约1.56e−19 ray之差。
全部9点，alpha-only与全替换根差的差最多9.582497638e−7；只换peak+rho的变化最多9.580981213e−7。
因此本批保存模型的大根差由**沿射线有效alpha**表示差主导。alpha包含off-ray能量衰减，不等于单独的训练opacity参数；几何能量计算仍是候选来源。
alpha与几何项的交互最大9.545268943e−7，六顺序Shapley只作有限代数分配，非因果贡献率；混合系数也未必对应共同合法3D Gaussian。

`results/exact_outlier_mix_report_check_v1.json`独立重放27源/archive、21产物、216个F32系数字段向量、72精确质量、54顺序/162边际/27分配/63交互及9抵消指标。旧输入与门槛未改。

## 根间平台的充分界

对没有peak的[a,b]，固定L={p_i≤a}、R={p_i≥b}，记M_L=Σ_L−log(1−alpha)，u_i(t)=−.5log(1−alpha_i exp[−rho_i²(t−p_i)²/2])。
则H(t)=M_L−Σ_L u_i(t)+Σ_R u_i(t)，且

`M_L−Σ_L u_i(a) ≤ H(t) ≤ M_L+Σ_R u_i(b)`。

若左界>log2，整段在根右；若右界<log2，整段在根左。该普通单调性/tail充分界不依赖根处导数下界，不宣称原创。
区间固定为两根之间中央60%，全18臂无内部peak。case1/2/5的两臂均获相反侧充分判定；其余6点有一臂界不充分，完整保留。
case2中点H_t约5.584e−13，case5约6.4e−161；两臂残差异号。根处约1e−3的局部斜率无法控制这样的跨层区间。
`exact_outlier_gap_v1`与独立MP180 direct-H审计通过18模型/126点，6项inconclusive保留；不是有向区间认证。

科研图及原曲线数据见 [gap_evidence.png](experiments/exact_outlier_gap_v1/gap_evidence.png)、同目录SVG/curve_data.npz/figure_manifest.json。
图先展示全部9点，再展示数学根差最大的2点；曲线为保存系数模型，非GT，symlog残差线性阈值1e−5，未平滑。PNG已人工检查及完整性检查。

## 原PLY高精度参照

`exact_outlier_world_v1` complete/exit0：原3个PLY按ID核验772行原始F32字段，MP80显式covariance逆与MP140真正inverse factor/垂直残差独立构造相符。
两参考明确区分：saved-activated保留F32激活尺度/opacity/quaternion，真实逆非正交quaternion polynomial covariance；raw-physical从原PLY重新高精度exp/sigmoid/filter/unit quaternion激活。
相机取保存F32 view的精确仿射逆，query_xy固定，t严格为单位camera-ray参数，不假定单位world弧长。alpha保持MP精度，应用同一精确F32 .99 cap后不round F32再求根。

全部36个MP80根在MP140复查端点符号通过；系数两精度scaled差最多2.407e−73，根差最多3.752e−31。
两套参照下均为factor更近8/9、packet更近1/9（case0）。最坏case2相对raw-physical，packet根差−.1816941078、factor+.0004048159 ray。
factor仍不是精确真值：case0相对raw-physical误差约+.0012097072 ray，超过1e−3量级；不能从8/9宣布完整采用或泛化排名。

还发现conic_opacity.w与送入预处理的filtered_opacity末位不同：17581候选中2307项、772 accepted中96项，accepted最大绝对差5.96046e−8。
两exact臂始终共享同一个co.w，故这不是两臂差异的独立opacity缓存解释。
新增第三参考固定saved-activated几何而取实际共同co.w，排除此共同amplitude因素后仍为factor8/9、packet1/9；参考根相对filtered-opacity版本位移最多1.78528e−6 ray。
这些参照是明确的数学三维对象，不能等同GT表面或证明真实训练质量。

## 下一项

已把问题从求根算术收敛到有效alpha/能量表示。候选修复应从同一precision factor构造一致的b、c和Schur/Gram能量，避免混合近似互逆链，并明确处理camera-mean投影与舍入。
`EXACT_OUTLIER_ALPHA_AUDIT.md`仅给待验证构造/缓存预算；9个F32=36B并非稳定性或加速证据。
先核验一致构造对既有1992输入及本772项的数值/条件误差，再锁定新构造确认；另推进薄层质量裕量与层切换的严格理论和相关工作核验。
首版packet全24视角及2.60倍耗时失败保持，当前不扩大其训练。

来源：mix_report/check、world_v1/world_audit_v1、gap_v1/gap_audit_v1、各协议与源码archive。原LaTeX同文件更新、内置编译记录见最新compile状态。
