# AgriOcc 中文论文草稿

本目录是面向 *Expert Systems with Applications*（ESWA）、*Computers and Electronics in Agriculture*（CEA）等期刊的中文初稿。主文件采用 Elsevier 官方 `elsarticle` 的 `preprint,12pt` 单栏选项；提交时可按目标期刊的 author guide 改为对应的双栏或终稿选项。

## 编译

本项目已配置为使用服务器的 `/data/HL/texlive/2026`（含 `elsarticle`、`ctex`、`tikz` 与 `latexmk`）。在远程 VS Code 中安装 LaTeX Workshop 后，工作区的 `.vscode/settings.json` 会自动选用同一工具链；保存 `paper_cn/*.tex` 将触发 XeLaTeX 编译。

```bash
cd paper_cn
/data/HL/texlive/2026/bin/x86_64-linux/latexmk \\
  -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

若 VS Code 已经打开本仓库，请执行“Developer: Reload Window”或重连远程服务器，使 LaTeX Workshop 读取新的工作区设置。PDF 默认在 VS Code 编辑器标签页中打开。

## 数据来源与写作约定

- FarmSim 数字来自对应 `work_dirs/*/*.log.json` 的最后一个验证记录；参数量来自训练日志中的 `Model built` 行。
- ORAD-3D 数字来自 `work_dirs/orad3d_evaluations/orad3d_eval_summary.csv`，均为 checkpoint 的独立官方测试集或 `farm_all` 子集评估。
- 表中空白单元格表示该组实验尚未完成，不应在正文或摘要中据此做比较性结论。
- FarmSim 消融是单次固定种子（seed 0）运行，未报告方差或显著性检验；正文据此使用“在该次运行中”“结果表明”等谨慎措辞。
