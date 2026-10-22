# 根间质量平台：补充事后诊断

2026-10-14。已观察最大的两例根间中点斜率极小和残差异号，以下是事后补充，不称预先注册。
覆盖全部9点的packet/factor，输入和根固定，不选择新像素。只CPU高精度，不训练。

根间区间定义为两MP根的中央60%：[r_low+.2*gap,r_low+.8*gap]，中点为两根平均。
报告内部peak数、精确累计terminal-prefix质量、两端tail界、真实中点残差和H_t。
若某臂区间无peak，其左侧集合L固定，右侧集合R固定，记
M_L=Σ_L−log(1−alpha)、B_L(t)=Σ_L−.5log(1−alpha exp[-rho²(t−p)²/2])，B_R同理。
于是 H(t)−log2 = M_L−log2−B_L(t)+B_R(t)。
对整个[a,b]，−B_L(a)≤H(t)−M_L≤B_R(b)。
若M_L−log2>B_L(a)，整段H>log2，中位根在a左；
若log2−M_L>B_R(b)，整段H<log2，中位根在b右。
这是普通单调性与非负tail的充分界，不能宣称一般新颖性。
质量平台常数用Fraction的beta_L=prod_L(1−alpha)构造，再MP log1p；不先减两个近等log常数。

输出全部状态。120位求值另以180位复算符号；MP数值检查不是向外舍入区间证书。
无peak条件失败则不使用该区间的tail充分界，不删样本；仍记录中点值和peak数。
图使用已保存数值曲线，严格区分模型曲线、选择后的异常样本与GT/统计总体。
