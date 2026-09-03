"""检测可视化纯函数: 缩放/画框/标注头/网格拼装(la_eval 与 review_pack 共用)."""

from PIL import Image, ImageDraw, ImageFont

from jxl.det.hardmine import Box

RGB = tuple[int, int, int]
_HEADER_H = 26


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def scale_to_width(im: Image.Image, width: int) -> Image.Image:
    """等比缩放到指定宽."""
    ratio = width / im.width
    return im.resize((width, max(1, int(im.height * ratio))))


def draw_boxes(im: Image.Image, boxes: list[Box], color: RGB, width_px: int = 3) -> Image.Image:
    """归一化 xyxy 框画到图上(返回新图, 不改原图)."""
    out = im.copy()
    dr = ImageDraw.Draw(out)
    for b in boxes:
        dr.rectangle(
            (b[0] * im.width, b[1] * im.height, b[2] * im.width, b[3] * im.height),
            outline=color,
            width=width_px,
        )
    return out


def label_header(im: Image.Image, text: str) -> Image.Image:
    """顶部加黑条白字标题行."""
    out = Image.new("RGB", (im.width, im.height + _HEADER_H), (0, 0, 0))
    out.paste(im, (0, _HEADER_H))
    ImageDraw.Draw(out).text((4, 4), text, font=_font(18), fill=(255, 255, 255))
    return out


def grid(images: list[Image.Image], cols: int, tile_w: int) -> Image.Image:
    """等宽拼网格(cols 列, 行数自适应; tile 高含 header 取最大)."""
    if not images:
        raise ValueError("grid: empty images")
    tiles = [scale_to_width(im, tile_w) for im in images]
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (tile_w * cols, th * rows), (30, 30, 30))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % cols) * tile_w, (i // cols) * th))
    return canvas
