import io
import threading

PILLOW_MAX_IMAGE_PIXELS = 16_000_000  # 16 MP decompression-bomb and memory ceiling
PILLOW_ALLOWED_FORMATS = ("PNG", "JPEG", "WEBP")  # blocks EPS/PostScript (Ghostscript) delegation
PILLOW_MAX_BYTES = 10 * 1024 * 1024  # 10 MB cap per embedded OCR image
FORBIDDEN_IMAGE_SIGNATURES = (b"%!PS", b"\xc5\xd0\xd3\xc6", b"%PDF")  # PostScript/EPS (ASCII & binary) and PDF
_LEADING_SIGNATURE_STRIP = b" \t\r\n\x0b\x0c\x00\x04"

_cache = None
_cache_lock = threading.Lock()
_pillow_lock = threading.Lock()


class OCRUnavailableError(RuntimeError):
    """Raised when OCR is needed but the easyocr package is not installed."""


class UnsafeImageError(ValueError):
    """Raised when an embedded OCR image violates safety limits or forbidden signatures."""


def _get_cache():
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                from .runtime import SingleFlightCache

                _cache = SingleFlightCache(maxsize=2)
    return _cache


def validate_image_bytes(data, *, max_bytes=PILLOW_MAX_BYTES):
    """Validate raw embedded image bytes with pure stdlib before Pillow decodes them."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not isinstance(data, (bytes, bytearray)):
        raise UnsafeImageError("Embedded image payload must be bytes")
    if not data:
        raise UnsafeImageError("Embedded image payload is empty")
    if len(data) > max_bytes:
        raise UnsafeImageError(f"Embedded image payload exceeds {max_bytes} bytes")

    raw = bytes(data)
    normalized = raw.lstrip(_LEADING_SIGNATURE_STRIP)
    if normalized.startswith(b"\xef\xbb\xbf"):
        normalized = normalized[3:].lstrip(_LEADING_SIGNATURE_STRIP)
    if normalized.startswith(FORBIDDEN_IMAGE_SIGNATURES):
        raise UnsafeImageError("Forbidden PostScript/EPS/PDF signature detected in embedded image payload")
    return raw


def harden_pillow(image_module=None):
    """Enforce the 16 MP pixel cap and strip Pillow's EPS (Ghostscript) decoder."""
    if image_module is None:
        try:
            from PIL import Image as image_module
        except ImportError as exc:
            raise OCRUnavailableError("Reading scanned pages needs the Pillow package") from exc

    with _pillow_lock:
        image_module.MAX_IMAGE_PIXELS = PILLOW_MAX_IMAGE_PIXELS
        image_module.init()
        image_module.OPEN.pop("EPS", None)
        if isinstance(image_module.ID, list):
            image_module.ID[:] = [fmt for fmt in image_module.ID if fmt != "EPS"]
        elif isinstance(image_module.ID, set):
            image_module.ID.discard("EPS")
        else:
            while "EPS" in image_module.ID:
                image_module.ID.remove("EPS")
    return image_module


def open_image_safely(data, *, max_pixels=None, max_bytes=PILLOW_MAX_BYTES):
    """Open embedded image bytes with strict PNG/JPEG/WEBP allowlist, byte cap, and 16 MP ceiling."""
    if max_pixels is None:
        limit = PILLOW_MAX_IMAGE_PIXELS
    else:
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
            raise ValueError("max_pixels must be a positive integer")
        limit = min(max_pixels, PILLOW_MAX_IMAGE_PIXELS)

    raw = validate_image_bytes(data, max_bytes=max_bytes)

    try:
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailableError("Reading scanned pages needs the Pillow package") from exc

    harden_pillow(Image)
    img = Image.open(io.BytesIO(raw), formats=PILLOW_ALLOWED_FORMATS)
    try:
        if img.width * img.height > limit:
            raise Image.DecompressionBombError(
                f"Image pixel count ({img.width}x{img.height}) exceeds {limit} pixels"
            )
        img.load()
    except Exception:
        img.close()
        raise
    return img


def get_reader(languages):
    try:
        # Imported lazily: easyocr loads torch and its models take seconds to start.
        import easyocr
        import torch
    except ImportError as exc:
        raise OCRUnavailableError("Reading scanned pages needs the easyocr package") from exc

    # The first call downloads the text detection and recognition models (about 100 MB).
    return _get_cache().get(
        tuple(languages),
        lambda: easyocr.Reader(list(languages), gpu=torch.cuda.is_available(), verbose=False),
    )


def read_page_images(page, languages=("tr", "en")):
    """Return the text found by OCR in the images of a pypdf page.

    A scanned PDF page is one picture of the paper, so its text can only be recovered
    by reading the picture.
    """
    page_images = list(page.images)
    if not page_images:
        return ""

    # Pure-stdlib pre-pass: reject forbidden signatures and oversized payloads even when
    # Pillow, numpy, or easyocr are unavailable.
    validated_payloads = [
        validate_image_bytes(image.data, max_bytes=PILLOW_MAX_BYTES) for image in page_images
    ]

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise OCRUnavailableError("Reading scanned pages needs the Pillow package") from exc

    harden_pillow(Image)

    decoded_arrays = []
    np_module = None
    for raw in validated_payloads:
        try:
            with open_image_safely(raw, max_bytes=PILLOW_MAX_BYTES) as opened:
                rgb_image = opened.convert("RGB")
                try:
                    if np_module is None:
                        try:
                            import numpy as np_module
                        except ImportError as exc:
                            raise OCRUnavailableError(
                                "Reading scanned pages needs the numpy package"
                            ) from exc
                    decoded_arrays.append(np_module.array(rgb_image))
                finally:
                    rgb_image.close()
        except UnsafeImageError:
            raise
        except Image.DecompressionBombError as exc:
            raise UnsafeImageError(str(exc)) from exc
        except (UnidentifiedImageError, OSError):
            # Some embedded formats (such as JBIG2) can't be decoded by Pillow; skip them
            # rather than fail the whole document.
            continue

    if not decoded_arrays:
        return ""

    reader = get_reader(tuple(languages))
    paragraphs = []
    for array in decoded_arrays:
        paragraphs.extend(reader.readtext(array, detail=0, paragraph=True))
    return "\n".join(paragraphs)
