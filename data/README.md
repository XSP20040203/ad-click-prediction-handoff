# data/ —— 原始数据不放在 Git 里

四份原始 CSV 合计约 1.1 GB,单文件最大 519 MB,超过 GitHub 单文件 100 MB 的硬限制,
因此不进版本库,改为挂在 Release 上,内容和哈希与 `manifest.json` 完全一致。

| 文件 | 字节数 | 行数 | 列数 |
| --- | --- | --- | --- |
| `train.csv` | 519,864,351 | 2,526,320 | 35 |
| `train_feeds.csv` | 412,305,071 | 827,820 | 28 |
| `final_test.csv` | 82,684,329 | 404,182 | 34 |
| `test_feeds.csv` | 71,755,288 | 142,962 | 28 |

## 获取方式

**方式一:整包 Release(推荐)**

在本仓库的 Releases 页下载 `ad_click_prediction_handoff_20261101_v1.zip`,
解压后即为完整交接包(含上述四份 CSV)。

**方式二:只补数据文件**

把同名的四份 CSV 直接放到本目录下即可,`tools/verify_bundle.py`
会用 `manifest.json` 里的 SHA256 逐字节核对。

## 核对

```powershell
# 放到 data/ 之后,在包根目录运行
python tools/verify_bundle.py
```

`manifest.json` 不把自己的哈希列进清单(避免自我依赖),其余文件都有 SHA256。
