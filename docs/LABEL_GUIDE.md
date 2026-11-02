# 猜测标签版本与局限

所有17号标签文件都是推断/拟合产物，不是观测真值。原final_test.csv没有广告点击label。

| 版本 | 位置 | 约束范围与状态 |
| --- | --- | --- |
| l04 | labels/l04_four_score_hypotheses.csv | 首4份评分；A/B/C三套候选及三种点击数概率情景 |
| l05 | labels/l05_five_score_hypotheses.csv | 首5份评分；A/B/C三套候选及反标签可行性检查 |
| l06 | labels/l06_selected_six_score_hypothesis.csv | 首6份评分；选定一套，NPZ保存11套候选；未证明全局最优 |
| l08_soft | labels/l08_eight_feedback_soft_estimates.npz | 首8份评分的最小KL软概率，尚未吸收0.75832 |

目前整理出37套去重的17号完整二进制假设。
`scores/hypothesis_inventory.csv`保留CSV列和NPZ数组列的对应关系，不用猜测数组下标。
五评分NPZ还包含最高概率20行的相反标签完整见证，证明这些行当时并未被总分强制锁定。

六评分主CSV中的`label`是选定的0/1假设，`estimated_click_probability`是模型条件估计。
两者目标不同，不能通过概率>0.5重建该套硬标签。3459个1是点击率假设，不是真实正例数。
四/五评分CSV中的`log_id`可对应原final_test行；不要按log_id去重，原始有858个重复log_id行。

较新的平台反馈已经否定“六评分选定候选就是真实标签”的解释。继续把同一猜测标签用于调分
再用于评分，只能调出本地目标数字，不能证明真实平台提升。本包保留这些文件用于追溯与后续研究。

`labels/support/day16_pseudo_training_simulation.npy`属于16日训练模拟，与train.csv中pt_d=20191116的原始行顺序一致，
不是17号标签。16日真实广告label仍可在原train.csv中找到，用于对照。
