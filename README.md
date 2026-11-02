# 广告点击预测数据交接包

整理日期：2026-11-01。包版本：ad_click_prediction_handoff_20261101_v1。
本包包含完整原始数据、全部已明确绑定文件的9条平台反馈、11份提交、各版猜测标签及复核资料。
平台成绩来自用户反馈，未获取平台真实标签，也未独立重算平台分数。

## 首先阅读

1. `scores/platform_feedback.csv` 是平台反馈总表，精确绑定具体CSV与SHA256。
2. `scores/submission_catalog.csv` 包含9份有反馈和2份无反馈的提交，未知成绩留空。
3. `labels/l06_selected_six_score_hypothesis.csv` 是此前用户选用的拟合标签，**不是真实标签**。
4. `docs/LABEL_GUIDE.md` 说明标签版本、备用候选、后续评分否定原假设的情况。
5. `docs/SCORE_GUIDE.md` 区分平台反馈、真实开发集AUC、本地伪标签AUC及软排名矩指标。

## 平台反馈

| ID | 版本 | 用户反馈平台AUC |
| --- | --- | --- |
| s01 | v6 | 0.80817 |
| s02 | v7 | 0.80581 |
| s03 | v8 | 0.80745 |
| s04 | v9 | 0.80856 |
| s05 | legacy_best | 0.74404 |
| s06 | legacy_submission_3 | 0.70590 |
| s07 | fitted_0810 | 0.80855 |
| s08 | fitted_0812 | 0.80855 |
| s09 | feedback_blend_0780 | 0.75832 |

当前已报告的平台最佳仍是s04/v9的0.80856。s09是最新已反馈文件，成绩0.75832，并非最佳。
s10的本地目标0.815和s11的伪标签训练结果尚无平台成绩，不能填成真实平台AUC。
早期口头版本名存在歧义，本包优先采用已核对的CSV哈希绑定；v7=0.80581，v8=0.80745。

## 原始数据

| 文件 | 行数 | 列数 |
| --- | --- | --- |
| train.csv | 2,526,320 | 35 |
| train_feeds.csv | 827,820 | 28 |
| final_test.csv | 404,182 | 34 |
| test_feeds.csv | 142,962 | 28 |

`data/train.csv` 的label是已观测广告点击；共21,354个点击。
`data/final_test.csv` 全部为20191117，共404,182行，**没有广告点击label列**。
`train_feeds.csv`和`test_feeds.csv`中的label/cillabel属于资讯行为数据，不能直接当成final_test的广告点击真值。
所有原始CSV均保留原始字节内容，未删行、去重或改标签。

## 目录

```text
data/          四份完整原始CSV
submissions/   规范命名的11份提交CSV，内容未改
labels/        四/五/六评分猜测标签、全部备用二进制候选、八反馈软概率
scores/        平台反馈、数据与标签目录、各套假设本地AUC、历史AUC字段索引
reports/       原协议、报告、复核记录、反馈与历史JSON快照
docs/          指标/标签说明、原数据说明PDF与原项目文档
code/          与本次评分、标签推断和伪标签训练有关的源码快照
baselines/     原始CatBoost和Logistic基线Notebook
tools/         完整性检查与本地AUC复算脚本
manifest.json  每个文件的路径、字节数、SHA256及原路径映射
```

所有新增归档路径使用ASCII英文命名，MD为UTF-8，索引CSV为UTF-8 BOM。
提交文件名中的platform分数来自用户反馈；unscored表示未知，不能由文件名猜测平台分数。

## 当前分析状态

- 四/五/六评分版本是假设集合；匹配历史总分不代表识别了逐条真实点击。
- 六评分选定标签基于假设3459次点击；低/高点击数情景为2940/3978，均不是观测到的17号真实总数。
- 原候选标签把s07/s08算成0.81000/0.81200，但平台都为0.80855，已与该候选就是真实标签不相容。
- s09在原候选标签上为0.78000；此前八反馈软排名矩指标约0.7596947；用户后来报告实际0.75832。
- 八反馈概率数组仅使用s01至s08，**尚未纳入第9条0.75832**。未完成九反馈新标签推断或下一版0.78平台目标提交。
- 先前优化任务被用户打断，当前工作仅为打包；没有继续自动训练。

## 如何核对

解压后，在包根目录运行（只需Python标准库）：

```powershell
python tools/verify_bundle.py
```

重新计算11份提交对六评分选定猜测标签的本地AUC（需要numpy/pandas/scikit-learn）：

```powershell
python tools/recompute_local_auc.py
```

`scores/local_auc_by_hypothesis.csv` 已列出所有37套去重的17号二进制假设与11份提交的本地AUC。
`scores/hypothesis_inventory.csv`列出每一CSV列或NPZ数组列及规范假设ID，重复向量通过同一ID关联。
`scores/historical_auc_field_index.csv`收录978份JSON中的24470个数值AUC字段，包含开发指标、阈值、目标和预期矩，**不是平台排行榜**。

## 复现范围与来源

`manifest.json`记录归档文件名与原始路径的映射；历史报告内的绝对路径原样保留，按映射追溯。
源码快照用于审计历史流程，不保证所有历史脚本开箱即跑；原深度模型权重、巨量特征缓存及训练环境不包含在本包。
原始提交概率、标签CSV和NPZ都已保存，因此本地指标和输入完整性可以独立复核。
manifest自身不列入自己的哈希清单，以避免循环依赖；其余所有归档文件都有SHA256。
