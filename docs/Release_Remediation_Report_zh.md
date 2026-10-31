# RAC-956 最后发布信息修复与只读复现验收

日期：2026-10-31。结论：**CONDITIONAL GO**。

## 在线与分支状态

仓库 https://github.com/HU18164053837-ai/RAC-Feature-Representation-Explanation-Stability 已公开，默认 main。开始时工作区干净，本地 main 与 fetch 后的 origin/main 均为 bbc95a0cbaaef4440c7eaa291ecc1f786e8513d9。修复分支为 chore/rac956-release-metadata-fix；分支提交和 PR 不等于已经合并 main，实际链接及提交以最终交付记录为准。

## 修复范围

README 更新公开状态、真实地址、作者单位/联系信息和已确认声明；保留原研究设计、所有指标、阴性结论、数据引用与依赖/复现说明。CITATION.cff 采用 Sun/Shiqi 与 Hu/Yi，填写真实 repository-code，删除未经发布确认的候选版本号，不填写论文或归档 DOI。GitHub_Release_QA、GitHub_Upload_Guide、RELEASE_CHECKLIST 与第三方说明清理旧状态；将 24 October 2026 本地验收明确标为历史。历史科研审计 JSON 及作者未提供新证据的原记录不重写。

## 方法与数据完整性

只读核对原始工作簿、722 冻结样本、563 开发/159 已使用来源隔离测试样本、980 分类账、锁定参数、保存预测与指标、30 个开发外折模型、SHAP 和共同材料块结果、26 张补充表、5+5 图像资产、主要代码/依赖及路径映射。所有原有非发布文档文件逐字节对比远端基线；不存在科学文件变化。脚本 verify_saved_results.py 与全部断言不变。

## CFF 与 SHA256

以 PyYAML 解析 CFF，再使用官方 CFF 1.2.0 JSON Schema（含 format checker）验证；姓名/URL/无虚构版本 DOI 另作显式断言。官方 schema 来源：https://raw.githubusercontent.com/citation-file-format/citation-file-format/1.2.0/schema.json 。CFF 结构验证通过不等于已经通过 GitHub 网页引用按钮进行端到端渲染验收。

保留初始 SHA256 所有条目；更新已修改文档的 bytes/hash，并将新增本报告及原清单未覆盖的 .gitattributes 加入。清单仅排除自身，未跳过任何原有文件。干净导出/检出中运行标准库入口 python scripts/verify_saved_results.py；实际输出保存于独立验收记录，不把历史包装验收冒充新在线模型评价。

## 版权授权结论

没有正式 LICENSE；LICENSE_CODE_PENDING.md 不是许可证。建议权利人考虑 BSD-3-Clause，因为相较 MIT 明确限制使用权利人/贡献者姓名背书；MIT 更简短，两者均为宽松、含免责条款的选择。官方文本：https://opensource.org/license/mit 与 https://opensource.org/license/bsd-3-clause 。未代作者采用任何许可，也未根据作者排序推定版权归属。数据 CC BY 4.0 继续单独署名；数据许可不自动覆盖代码、缓存模型和其他文件。

## 投稿剩余事项

版权权利人、机构/第三方贡献权利、正式代码/模型许可范围仍待批准；永久归档 DOI 未提供；作者需将主稿可用性声明与真实发布状态同步并核对期刊要求。上述事项使本次结论为 CONDITIONAL GO，而非无条件 GO。没有数据不一致或重训需求，不能据此宣布完整论文已获得期刊投稿通过。

## 本轮实际执行结果

CFF 官方 schema：PASS。干净 Git tree 导出中，独立 no-pip 环境执行只读入口：

```json
{
  "status": "PASS",
  "manifest_files": 800,
  "matrix": 722,
  "development": 563,
  "consumed_holdout": 159,
  "saved_metrics_checked": 12,
  "model_fits": 0,
  "new_predictions": 0,
  "stdlib_only": true
}
```

原有非修改范围 792 个文件逐字节不变；26 张补充表、5+5 PNG 图像均在。上述是保存结果核验，不是新的测试集预测。初次附加清点把整个 Stage 3 的 33 个模型误当为 30 个外折模型，随后按目录确认 30 个 outer_models 加 3 个 locked_development_models；未改科研文件或原验证断言。最终提交仍需在干净检出中复核。
