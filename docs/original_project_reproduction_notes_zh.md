# 项目清理后的保留内容

本次按整个项目清理了大型训练矩阵、重复实验缓存和未采用的中间数组。原始数据、源码、模型权重、指标协议及当前提交均保留。

- `2026ml/`：原始训练集、final及新闻源域CSV，共约1.09GB。
- `output/best_nextday_arch_v7/submission.csv`：当前推荐提交，404182行，内容与清理前完全一致。
- `compact_final/`：约3.35GB的独立最终集复算包，含10个模型、去重的final特征切片、词表、代码、原预测参考及哈希；无需原来的全量训练缓存。
- `analysis/`：训练/特征脚本、报告、指标、实验协议、选中开发模型和验证预测等证据。

## 清理后重新计算final

使用原机器学习环境运行：

```bash
"E:/Users/env/Machine_Learning/python.exe" compact_final/replay_compact.py --device cuda
```

该命令依次加载10个模型，预测全部404182行，重算完整v7融合；结果写入 `compact_final/replayed/`，核验记录为 `compact_final/replay_verification.json`。使用CPU时将`cuda`改为`cpu`。

需要提交现有成绩对应文件时，仍使用 `output/best_nextday_arch_v7/submission.csv`。CUDA集合归约存在极小浮点差异，新复算CSV末位可能变化；原提交完整保留。

## 重新做训练实验

全量训练特征缓存已按要求永久删除。原始CSV和特征生成代码仍在；重新训练前需按相应准备脚本重建缓存。旧实验的指标、预测和协议保留，但其大型训练输入不再占用磁盘。

2026-10-31新增的`analysis/strong_interest_control_20261031/`也支持在内存重建原166维强主干输入，无需恢复大型磁盘缓存。两天旧权重预测重放误差均为0；新增兴趣注意力未确认稳定收益，推荐提交仍为原v7。实验结果与边界见该目录的`实验结论.md`。

本次永久删除对象、文件大小及清理结果见 `analysis/project_cleanup_20261031/`。此前移入回收站的少量历史权重未混同为本次170GB以上的实际释放量，也未清空其他回收站内容。
