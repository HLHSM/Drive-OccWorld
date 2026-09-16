#!/usr/bin/env python3
"""Build a five-slide editable companion deck for AgriOcc architecture figures.

The source artwork was generated as PNG.  Every slide therefore contains two
layers: (1) editable PowerPoint shapes/text/connectors; (2) a removable,
top-level 16:9 reference image that guarantees the exported slide is
pixel-identical to the figure used in the paper. Delete the object whose name
begins with "Reference Raster" in PowerPoint to expose the vector reconstruction.
"""

from pathlib import Path
import sys

DEPS = Path("/tmp/agriocc_ppt_deps")
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUT = Path(__file__).with_name("AgriOcc_editable_architecture_figures.pptx")

W, H = 13.333333, 7.5

NAVY = "062A62"
BLUE = "A9D6FF"
CYAN = "D6F4FC"
PURPLE = "CDB7F6"
GREEN = "BFF1B8"
ORANGE = "FFC48B"
PALE = "F7FBFF"
LAVENDER = "DCD7FF"
GREY = "EEF2F6"
WHITE = "FFFFFF"
BLACK = "111827"


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


def set_text(tf, text, size=14, bold=False, color=NAVY, align=PP_ALIGN.CENTER):
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.margin_top = Pt(1)
    tf.margin_bottom = Pt(1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)


def shape(slide, x, y, w, h, text="", fill=PALE, line=NAVY, size=13,
          bold=False, radius=True, name="Edit: module", transparency=0):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    item = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    item.name = name
    item.fill.solid()
    item.fill.fore_color.rgb = rgb(fill)
    item.fill.transparency = transparency
    item.line.color.rgb = rgb(line)
    item.line.width = Pt(1.35)
    if text:
        set_text(item.text_frame, text, size=size, bold=bold)
    return item


def label(slide, x, y, w, h, text, size=13, bold=False, color=NAVY,
          name="Edit: label", align=PP_ALIGN.CENTER):
    item = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    item.name = name
    set_text(item.text_frame, text, size=size, bold=bold, color=color, align=align)
    return item


def arrow(slide, x1, y1, x2, y2, color=NAVY, dashed=False, name="Edit: arrow"):
    item = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    item.name = name
    item.line.color.rgb = rgb(color)
    item.line.width = Pt(1.5)
    if dashed:
        item.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    item.line.end_arrowhead = True
    return item


def container(slide, x, y, w, h, title, fill=WHITE, name="Edit: container"):
    item = shape(slide, x, y, w, h, "", fill=fill, line=NAVY, radius=True, name=name)
    label(slide, x + 0.12, y + 0.05, w - 0.24, 0.34, title, size=16, bold=True,
          name="Edit: container title")
    return item


def grid(slide, x, y, cols, rows, cell, fill=CYAN, name="Edit: BEV grid"):
    for r in range(rows):
        for c in range(cols):
            shape(slide, x + c * cell, y + r * cell, cell, cell, "", fill=fill,
                  line="3273B9", radius=False, name=name)


def slide_base(prs):
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(WHITE)
    return slide


def add_reference(slide, image):
    from PIL import Image
    # Bake a white 16:9 canvas around the source image. Unlike a separate
    # white rectangle, this gives LibreOffice and PowerPoint one top-level
    # raster object and keeps the reconstruction below completely hidden.
    canvas_dir = OUT.parent / "_reference_canvases"
    canvas_dir.mkdir(exist_ok=True)
    canvas_path = canvas_dir / f"{image.stem}_16x9.png"
    target_w, target_h = 1920, 1080
    src = Image.open(image).convert("RGB")
    scale = min(target_w / src.width, target_h / src.height)
    rendered = src.resize((round(src.width * scale), round(src.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), "white")
    canvas.paste(rendered, ((target_w - rendered.width) // 2, (target_h - rendered.height) // 2))
    canvas.save(canvas_path, optimize=True)
    item = slide.shapes.add_picture(str(canvas_path), Inches(0), Inches(0),
                                    width=Inches(W), height=Inches(H))
    item.name = "Reference Raster — delete to reveal editable reconstruction"


def overall(prs):
    slide = slide_base(prs)
    label(slide, 0.2, 0.1, 12.93, 0.45, "AgriOcc Architecture", size=25, bold=True,
          name="Edit: title")
    container(slide, 0.15, 0.85, 1.55, 5.95, "3 Front-view Images")
    for i, text in enumerate(("Front-Left", "Front", "Front-Right")):
        shape(slide, 0.38, 1.45 + 1.55 * i, 1.1, 0.88, text, fill=BLUE, size=11,
              name="Edit: camera view")
    container(slide, 1.78, 0.85, 2.35, 5.95, "ResNet-101 + FPN")
    for i in range(4):
        shape(slide, 2.14, 1.65 + i * 0.8, 0.55, 0.38, "", fill=BLUE, size=9,
              radius=False, name="Edit: ResNet feature")
        shape(slide, 3.1, 1.65 + i * 0.8, 0.62, 0.32, f"P{i+2}", fill=PURPLE, size=10,
              radius=False, name="Edit: FPN feature")
        arrow(slide, 2.7, 1.84 + i * 0.8, 3.1, 1.82 + i * 0.8)
    label(slide, 2.0, 5.3, 1.95, 0.5, "4-scale image features", size=12, bold=True)
    container(slide, 4.25, 0.85, 3.75, 5.95, "GeoVis-BEV Encoder (6 layers)", fill="F3FFF0")
    shape(slide, 4.55, 1.35, 3.15, 0.42, "NearFar Query Layout, Layers 1–5", fill=GREEN,
          size=13, bold=True, name="Edit: NearFar band")
    shape(slide, 4.72, 2.05, 0.68, 0.65, "GVAD", fill=ORANGE, size=13, bold=True)
    shape(slide, 5.7, 2.05, 1.15, 0.65, "Image\nCross-Attention", fill=BLUE, size=11, bold=True)
    shape(slide, 7.15, 2.05, 0.55, 0.65, "FFN", fill=ORANGE, size=12, bold=True)
    arrow(slide, 5.4, 2.37, 5.7, 2.37); arrow(slide, 6.85, 2.37, 7.15, 2.37)
    label(slide, 7.5, 2.2, 0.32, 0.35, "× 6", size=16, bold=True)
    shape(slide, 4.75, 3.1, 2.9, 0.4, "Restore Dense BEV before Layer 6", fill=GREEN, size=12,
          bold=True, name="Edit: dense restore")
    grid(slide, 4.75, 4.2, 7, 4, 0.22); grid(slide, 6.55, 4.2, 7, 4, 0.22)
    arrow(slide, 6.3, 4.65, 6.55, 4.65)
    label(slide, 4.55, 5.15, 1.75, 0.45, "BEV Tokens\n100 × 100 × 256", size=11, bold=True)
    label(slide, 6.45, 5.15, 1.6, 0.45, "Dense BEV Tokens\n100 × 100 × 256", size=11, bold=True)
    container(slide, 8.1, 0.85, 5.05, 5.95, "3-stage Occupancy Decoder")
    xs = [8.38, 9.28, 10.18]
    for x, text, fill in zip(xs, ("Stage 1", "Stage 2", "Final Stage:\nAgri-AMoE"), (BLUE, BLUE, ORANGE)):
        shape(slide, x, 2.0, 0.72 if x < 10 else 1.05, 0.65, text, fill=fill, size=11, bold=True)
    arrow(slide, 9.1, 2.32, 9.28, 2.32); arrow(slide, 10.0, 2.32, 10.18, 2.32)
    shape(slide, 11.4, 2.0, 0.68, 0.65, "BEV-to-3D\nLift", fill=GREY, size=10, bold=True)
    arrow(slide, 11.23, 2.32, 11.4, 2.32)
    grid(slide, 12.15, 2.55, 5, 5, 0.14, fill="BAE3A6")
    label(slide, 11.98, 3.35, 0.95, 0.55, "6-class\nSemantic Logits\n100 × 100 × 25 × 6", size=9, bold=True)
    shape(slide, 8.72, 3.2, 2.45, 0.38, "Multi-level Occupancy Supervision", fill=LAVENDER, size=11, bold=True)
    for x in (8.75, 9.65, 10.7): arrow(slide, x, 2.65, x, 3.2, dashed=True)
    shape(slide, 9.25, 4.15, 2.25, 1.65, "Agri-AMoE", fill=ORANGE, size=15, bold=True,
          name="Edit: Agri-AMoE block")
    for x, t in zip((9.42, 10.05, 10.68), ("Planar\nExpert", "Vertical\nExpert", "Context\nExpert")):
        shape(slide, x, 4.78, 0.5, 0.52, t, fill="FFF2D7", size=8, bold=True)
    add_reference(slide, FIGURES / "agriocc_overview_geovis_v2.png")


def geovis(prs):
    slide = slide_base(prs)
    label(slide, 0.2, 0.1, 12.93, 0.4, "GeoVis-BEV Encoder", size=25, bold=True)
    container(slide, 0.25, 0.75, 2.05, 5.95, "Multi-view Image Features")
    for i, txt in enumerate(("Front-Left", "Front", "Front-Right")):
        shape(slide, 0.55, 1.3 + i * 1.35, 1.35, 0.75, txt, fill=BLUE, size=11, bold=True)
    container(slide, 2.55, 0.75, 1.8, 5.95, "ResNet-101 + FPN")
    for i in range(4):
        shape(slide, 2.85, 1.35 + i * 0.65, 0.55, 0.3, "", fill=BLUE, radius=False)
        shape(slide, 3.57, 1.35 + i * 0.65, 0.5, 0.3, f"P{i+2}", fill=PURPLE, size=10, radius=False)
    container(slide, 4.65, 0.75, 5.2, 5.95, "GeoVis-BEV Encoder (6 layers)", fill="F3FFF0")
    shape(slide, 4.95, 1.22, 4.6, 0.42, "NearFar Query Layout: Layers 1–5", fill=GREEN, size=14, bold=True)
    for row, layer in enumerate(("Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5")):
        y = 1.95 + row * 0.56
        label(slide, 4.84, y, 0.62, 0.34, layer, size=10, bold=True)
        for x, txt, fill in ((5.5, "GVAD", ORANGE), (6.6, "Image Cross-Attention", BLUE), (8.3, "FFN", ORANGE)):
            shape(slide, x, y, 0.85 if txt != "Image Cross-Attention" else 1.45, 0.34, txt, fill=fill,
                  size=8, bold=True)
    shape(slide, 5.2, 4.9, 4.0, 0.38, "Bilinear Restore to Dense BEV Grid", fill=GREEN, size=12, bold=True)
    label(slide, 4.84, 5.52, 0.62, 0.34, "Layer 6", size=10, bold=True)
    for x, txt, fill in ((5.5, "GVAD", ORANGE), (6.6, "Image Cross-Attention", BLUE), (8.3, "FFN", ORANGE)):
        shape(slide, x, 5.52, 0.85 if txt != "Image Cross-Attention" else 1.45, 0.34, txt, fill=fill,
              size=8, bold=True)
    container(slide, 10.1, 0.75, 2.95, 5.95, "Dense BEV Output")
    grid(slide, 10.82, 2.1, 8, 6, 0.19)
    label(slide, 10.34, 3.65, 2.5, 0.55, "Dense BEV Tokens\n100 × 100 × 256", size=13, bold=True)
    arrow(slide, 2.3, 3.6, 2.55, 3.6); arrow(slide, 4.35, 3.6, 4.65, 3.6); arrow(slide, 9.85, 3.6, 10.1, 3.6)
    add_reference(slide, FIGURES / "geovis_bev_encoder_overview.png")


def gvad(prs):
    slide = slide_base(prs)
    label(slide, 0.2, 0.1, 12.93, 0.4, "Geometry-visible Anchor Deformable Attention (GVAD)", size=22, bold=True)
    shape(slide, 0.35, 2.85, 1.2, 0.62, "BEV Query\nq + position", fill=CYAN, size=12, bold=True)
    shape(slide, 0.35, 1.45, 1.2, 0.62, "Camera Projection\nMask", fill=GREY, size=11, bold=True)
    arrow(slide, 0.95, 2.85, 0.95, 2.07)
    shape(slide, 1.88, 1.45, 1.25, 0.62, "Visibility\nm", fill=GREEN, size=12, bold=True)
    arrow(slide, 1.55, 1.76, 1.88, 1.76)
    container(slide, 1.9, 2.45, 4.65, 3.65, "Local Deformable Path", fill="F6FBFF")
    shape(slide, 2.25, 3.1, 1.1, 0.62, "Offset & Weight\nPrediction", fill=BLUE, size=11, bold=True)
    shape(slide, 3.75, 3.1, 1.05, 0.62, "4 sampling\npoints × 8 heads", fill=CYAN, size=10, bold=True)
    shape(slide, 5.15, 3.1, 1.0, 0.62, "Local\nUpdate d", fill=ORANGE, size=11, bold=True)
    arrow(slide, 1.55, 3.15, 2.25, 3.4); arrow(slide, 3.35, 3.4, 3.75, 3.4); arrow(slide, 4.8, 3.4, 5.15, 3.4)
    container(slide, 6.8, 1.0, 3.35, 4.05, "Geometry-visible Anchor Path", fill="FFF9F0")
    shape(slide, 7.12, 1.65, 1.1, 0.6, "4 × 8 BEV\nAnchor Grid", fill=CYAN, size=11, bold=True)
    shape(slide, 8.52, 1.65, 1.18, 0.6, "Confidence +\nlog Visibility", fill=GREEN, size=10, bold=True)
    shape(slide, 7.12, 2.75, 1.1, 0.6, "32 Visible\nAnchors", fill=PURPLE, size=11, bold=True)
    shape(slide, 8.52, 2.75, 1.18, 0.6, "Distance-biased\nAttention", fill=ORANGE, size=10, bold=True)
    shape(slide, 7.82, 3.85, 1.22, 0.6, "Anchor\nContext g", fill=ORANGE, size=11, bold=True)
    arrow(slide, 8.22, 1.95, 8.52, 1.95); arrow(slide, 7.67, 2.25, 7.67, 2.75); arrow(slide, 9.1, 2.25, 9.1, 2.75); arrow(slide, 8.22, 3.35, 8.4, 3.85); arrow(slide, 9.1, 3.35, 8.4, 3.85)
    shape(slide, 10.55, 2.75, 1.15, 0.72, "Visibility Gate\nσ(fg[q,m])", fill=GREEN, size=11, bold=True)
    shape(slide, 11.98, 2.75, 1.05, 0.72, "Residual\nFusion", fill=ORANGE, size=11, bold=True)
    arrow(slide, 6.15, 3.4, 10.55, 3.1); arrow(slide, 9.0, 4.45, 10.55, 3.35); arrow(slide, 11.7, 3.1, 11.98, 3.1)
    shape(slide, 11.77, 4.25, 1.35, 0.6, "Enhanced\nBEV Token q′", fill=CYAN, size=11, bold=True)
    arrow(slide, 12.5, 3.47, 12.5, 4.25)
    add_reference(slide, FIGURES / "gvad_detailed_architecture_v2.png")


def nearfar(prs):
    slide = slide_base(prs)
    label(slide, 0.2, 0.1, 12.93, 0.4, "NearFar BEV Query Layout", size=25, bold=True)
    container(slide, 0.35, 1.0, 2.5, 5.55, "1. Query Selection")
    grid(slide, 0.72, 1.75, 7, 7, 0.2)
    shape(slide, 0.65, 3.7, 1.9, 0.42, "Near 60%: retain all tokens", fill=GREEN, size=11, bold=True)
    shape(slide, 0.65, 4.3, 1.9, 0.42, "Far 40%: stride-2 sampling", fill=PURPLE, size=11, bold=True)
    label(slide, 0.55, 5.1, 2.1, 0.5, "Keep original dense\nBEV coordinates", size=12, bold=True)
    container(slide, 3.15, 1.0, 3.2, 5.55, "2. Sparse Encoding")
    for i in range(5):
        y = 1.7 + i * 0.72
        shape(slide, 3.58, y, 2.34, 0.45, f"Layer {i+1}:  GVAD  →  Cross-Attn  →  FFN", fill=BLUE, size=9, bold=True)
    label(slide, 3.55, 5.47, 2.45, 0.4, "Active Tokens Only", size=13, bold=True)
    container(slide, 6.65, 1.0, 2.55, 5.55, "3. Restore Dense BEV")
    shape(slide, 7.08, 1.78, 1.7, 0.62, "4 Neighboring\nActive Tokens", fill=CYAN, size=12, bold=True)
    shape(slide, 7.08, 2.86, 1.7, 0.62, "Bilinear\nInterpolation", fill=GREEN, size=12, bold=True)
    grid(slide, 7.28, 4.0, 7, 5, 0.19)
    arrow(slide, 7.93, 2.4, 7.93, 2.86); arrow(slide, 7.93, 3.48, 7.93, 4.0)
    container(slide, 9.5, 1.0, 3.25, 5.55, "Final Dense Refinement")
    shape(slide, 9.9, 2.0, 2.45, 0.6, "Layer 6: GVAD → Image Cross-Attn → FFN", fill=ORANGE, size=11, bold=True)
    grid(slide, 10.25, 3.25, 7, 5, 0.2)
    label(slide, 9.8, 4.65, 2.6, 0.52, "Dense BEV Tokens\n100 × 100 × 256", size=13, bold=True)
    arrow(slide, 2.85, 3.75, 3.15, 3.75); arrow(slide, 6.35, 3.75, 6.65, 3.75); arrow(slide, 9.2, 3.75, 9.5, 3.75)
    add_reference(slide, FIGURES / "nearfar_detailed_architecture_v2.png")


def amoe(prs):
    slide = slide_base(prs)
    label(slide, 0.2, 0.1, 12.93, 0.4, "Agri-AMoE 3D Occupancy Decoder", size=24, bold=True)
    shape(slide, 0.35, 2.8, 1.3, 0.7, "Final BEV\nTokens", fill=CYAN, size=13, bold=True)
    shape(slide, 1.95, 2.8, 1.28, 0.7, "BEV-to-3D\nLift", fill=ORANGE, size=13, bold=True)
    arrow(slide, 1.65, 3.15, 1.95, 3.15)
    container(slide, 3.55, 1.15, 2.25, 4.6, "3D Feature Volume")
    grid(slide, 4.0, 2.25, 7, 7, 0.2, fill="BAE3A6")
    label(slide, 3.83, 4.1, 1.72, 0.5, "Cₐ = 96, Z = 25", size=12, bold=True)
    arrow(slide, 3.23, 3.15, 3.55, 3.15)
    container(slide, 6.15, 1.15, 2.18, 4.6, "Feature-guided Router", fill="F6FBFF")
    shape(slide, 6.48, 1.9, 1.52, 0.52, "Channel Saliency", fill=BLUE, size=11, bold=True)
    shape(slide, 6.48, 2.72, 1.52, 0.52, "Spatial Saliency", fill=GREEN, size=11, bold=True)
    shape(slide, 6.48, 3.54, 1.52, 0.52, "x/y/z Gradient\nEnergy", fill=PURPLE, size=11, bold=True)
    shape(slide, 6.48, 4.46, 1.52, 0.52, "Voxel-wise\nSoftmax Router", fill=ORANGE, size=11, bold=True)
    arrow(slide, 5.8, 3.15, 6.15, 3.15)
    container(slide, 8.68, 1.15, 2.45, 4.6, "Three Anisotropic Experts", fill="FFF9F0")
    for y, t in zip((1.85, 2.9, 3.95), ("Planar Detail\n1 × 3 × 3", "Vertical Structure\n3 × 1 × 1", "Planar Context\nDilation 2")):
        shape(slide, 9.07, y, 1.68, 0.62, t, fill=ORANGE, size=11, bold=True)
    arrow(slide, 8.33, 4.72, 8.68, 4.72)
    shape(slide, 11.48, 2.8, 1.38, 0.7, "Residual Mix\n+ 1×1×1 Classifier", fill=ORANGE, size=11, bold=True)
    arrow(slide, 11.13, 3.15, 11.48, 3.15)
    shape(slide, 11.48, 4.2, 1.38, 0.7, "6-class\nSemantic Logits", fill=CYAN, size=12, bold=True)
    arrow(slide, 12.17, 3.5, 12.17, 4.2)
    add_reference(slide, FIGURES / "agri_amoe_module_imagegen.png")


def main():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "AgriOcc Editable Architecture Figures"
    prs.core_properties.subject = "Five editable architecture reconstructions with reference rasters"
    overall(prs)
    geovis(prs)
    gvad(prs)
    nearfar(prs)
    amoe(prs)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
