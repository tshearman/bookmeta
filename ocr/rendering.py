import base64
import io
import fitz
from PIL import Image, ImageOps


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Convert to grayscale and enhance contrast for OCR."""
    gray = image.convert("L")
    return ImageOps.autocontrast(gray)


def _pixmap_to_image(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return image.convert("RGB") if mode == "RGBA" else image


def resize_image_to_long_edge(
    image: Image.Image, max_long_edge: int = 1200
) -> Image.Image:
    """Return a resized copy of the image constrained by its longest edge."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_long_edge:
        return image
    scale = max_long_edge / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def page_to_image(
    page: fitz.Page, grayscale: bool = True, max_long_edge: int | None = None
) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    image = _pixmap_to_image(pix)
    if grayscale:
        image = preprocess_for_ocr(image)
    if max_long_edge:
        image = resize_image_to_long_edge(image, max_long_edge=max_long_edge)
    return image


def img_to_b64(
    img: Image.Image,
    format: str = "PNG",
    decode: str = "ascii",
) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode(decode)


def img_to_url(img: Image.Image):
    mime = "image/png"
    b64 = img_to_b64(img)
    return f"data:{mime};base64,{b64}"


def render_page_image_base64(
    doc: fitz.Document,
    page_index: int,
    max_long_edge: int = 1200,
    format: str = "PNG",
    decode: str = "ascii",
) -> str:
    """Render a page to an in-memory PNG and return a base64 string."""
    page = doc.load_page(page_index)
    img = page_to_image(page, max_long_edge=max_long_edge)
    return img_to_b64(img, format, decode)
