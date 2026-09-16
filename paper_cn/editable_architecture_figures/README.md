# AgriOcc 可编辑架构图 PPTX

`AgriOcc_editable_architecture_figures.pptx` 含五页 16:9 幻灯片，对应论文中的：

1. AgriOcc 总体架构；
2. GeoVis-BEV 编码器；
3. GVAD；
4. NearFar BEV 查询布局；
5. Agri-AMoE 三维占用解码器。

每页最上层都有一个名为 `Reference Raster — delete to reveal editable reconstruction` 的图片对象，用于保证幻灯片显示结果与论文 PNG 完全一致。
如需编辑结构，请在 PowerPoint 的“选择窗格”中删除该对象；下方的文本、模块、箭头和网格均为独立可编辑对象，名称以 `Edit:` 开头。

可使用 `generate_editable_architecture_pptx.py` 重新生成 PPTX；脚本依赖仅安装在临时目录 `/tmp/agriocc_ppt_deps`，不会写入项目环境。

## 推荐的 HTML/SVG 编辑器

打开 `architecture_editor.html` 即可使用。它默认显示论文 PNG，因此外观与论文图保持一致；点击左侧“编辑 SVG 重建层”后，可点击或拖动模块，并在属性面板中修改文字、位置、尺寸与颜色。页面还支持下载当前 SVG 和编辑数据 JSON。

若浏览器限制直接打开本地图片，可在仓库根目录运行：

```bash
python3 -m http.server 8000
```

然后访问 `http://localhost:8000/paper_cn/editable_architecture_figures/architecture_editor.html`。
