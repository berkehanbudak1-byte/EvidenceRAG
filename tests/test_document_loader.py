from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.rag.config import InputLimitError
from app.rag.document_loader import DocumentError, EmptyDocumentError, load_document


def write_pdf(path, texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=100, height=100)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 10 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


@pytest.mark.parametrize("extension", [".txt", ".md"])
@pytest.mark.parametrize("size", [4, 5, 6])
def test_text_char_boundary_unicode(tmp_path, extension, size):
    path = tmp_path / ("document" + extension)
    path.write_text("ş" * size, encoding="utf-8")
    if size > 5:
        with pytest.raises(InputLimitError, match="characters"):
            load_document(path, max_text_chars=5)
    else:
        assert load_document(path, max_text_chars=5)[0]["text"] == "ş" * size


def test_text_read_requests_only_limit_plus_one(monkeypatch, tmp_path):
    from io import StringIO
    path = tmp_path / "document.txt"
    path.write_text("exists", encoding="utf-8")
    counts = []
    class MeasuredText(StringIO):
        def read(self, count=-1):
            counts.append(count)
            return super().read(count)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: MeasuredText("x" * 100))
    with pytest.raises(InputLimitError):
        load_document(path, max_text_chars=5)
    assert counts == [6]


def test_invalid_utf8_is_document_error(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(DocumentError, match="UTF-8"):
        load_document(path)


@pytest.mark.parametrize("pages", [1, 2, 3])
def test_native_pdf_page_boundary(tmp_path, pages):
    path = tmp_path / "small.pdf"
    write_pdf(path, ["a"] * pages)
    if pages > 2:
        with pytest.raises(InputLimitError, match="pages"):
            load_document(path, max_pdf_pages=2)
    else:
        result = load_document(path, max_pdf_pages=2)
        assert [p["metadata"]["page"] for p in result] == list(range(1, pages + 1))


@pytest.mark.parametrize("size", [4, 5, 6])
def test_native_pdf_text_boundary(tmp_path, size):
    path = tmp_path / "small.pdf"
    write_pdf(path, ["x" * size])
    if size > 5:
        with pytest.raises(InputLimitError, match="characters"):
            load_document(path, max_text_chars=5)
    else:
        assert load_document(path, max_text_chars=5)[0]["text"] == "x" * size


def test_pdf_stops_extracting_after_total_text_overflow(monkeypatch, tmp_path):
    from pypdf._page import PageObject
    path = tmp_path / "small.pdf"
    write_pdf(path, ["xxx", "xxx", "xxx"])
    calls = []
    original = PageObject.extract_text
    def extract(page, *args, **kwargs):
        calls.append(page)
        return original(page, *args, **kwargs)
    monkeypatch.setattr(PageObject, "extract_text", extract)
    with pytest.raises(InputLimitError):
        load_document(path, max_text_chars=5)
    assert len(calls) == 2


def test_native_blank_pdf_is_empty_document_error(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=10, height=10)
    writer.write(path)
    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_native_corrupt_pdf_is_document_error(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(DocumentError, match="Could not read PDF"):
        load_document(path)

import io
from types import SimpleNamespace
import warnings
import pytest
from PIL import Image

from app.rag import document_loader
from app.rag.document_loader import EmptyDocumentError, load_document
from app.rag.ocr import OCRUnavailableError, UnsafeImageError


@pytest.fixture
def scanned_pdf(tmp_path):
    path = tmp_path / "taranmis.pdf"
    Image.new("L", (200, 100), color=255).save(path, "PDF")
    return path


def test_scanned_pages_are_read_with_ocr(scanned_pdf, monkeypatch):
    calls = []

    def fake_ocr(page, languages):
        calls.append(tuple(languages))
        return "Sayımda 315 koli bulundu."

    monkeypatch.setattr(document_loader, "read_page_images", fake_ocr)

    pages = load_document(scanned_pdf, ocr_languages=["tr"])

    assert calls == [("tr",)]
    assert pages[0]["text"] == "Sayımda 315 koli bulundu."
    assert pages[0]["metadata"]["page"] == 1
    assert pages[0]["metadata"]["ocr"] is True


def test_scanned_pdf_is_rejected_when_ocr_is_disabled(scanned_pdf):
    with pytest.raises(EmptyDocumentError):
        load_document(scanned_pdf, ocr_languages=[])


def test_scanned_pdf_is_rejected_when_ocr_is_not_installed(scanned_pdf, monkeypatch):
    def missing_ocr(page, languages):
        raise OCRUnavailableError("Reading scanned pages needs the easyocr package")

    monkeypatch.setattr(document_loader, "read_page_images", missing_ocr)

    with pytest.raises(EmptyDocumentError, match="OCR is unavailable"):
        load_document(scanned_pdf, ocr_languages=["tr"])


def test_text_files_are_not_marked_as_ocr(tmp_path):
    path = tmp_path / "notlar.txt"
    path.write_text("düz metin", encoding="utf-8")

    pages = load_document(path, ocr_languages=["tr"])

    assert "ocr" not in pages[0]["metadata"]


def test_legitimate_pdf_still_loads_with_ocr_hardening(tmp_path):
    path = tmp_path / "legitimate.pdf"
    write_pdf(path, ["Kanit belgesi 2026"])

    pages = load_document(path, ocr_languages=["tr"])

    assert len(pages) == 1
    assert pages[0]["text"] == "Kanit belgesi 2026"
    assert "ocr" not in pages[0]["metadata"]


@pytest.mark.parametrize(
    "payload",
    [
        b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\nshowpage\n",
        b"\xc5\xd0\xd3\xc6" + b"\x00" * 28,
    ],
)
def test_eps_disguised_as_pdf_is_rejected_with_document_error(tmp_path, payload):
    path = tmp_path / "disguised.pdf"
    path.write_bytes(payload)

    with pytest.raises(DocumentError, match="Could not read PDF"):
        load_document(path, ocr_languages=["tr"])


@pytest.mark.parametrize(
    "payload",
    [
        b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\nshowpage\n",
        b"\xc5\xd0\xd3\xc6" + b"\x00" * 28,
        b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n",
    ],
)
def test_scanned_pdf_propagates_unsafe_image_error_for_malicious_embedded_streams(
    scanned_pdf, monkeypatch, payload
):
    fake_page = SimpleNamespace(
        extract_text=lambda: "",
        images=[SimpleNamespace(data=payload)],
    )
    monkeypatch.setattr(
        document_loader,
        "PdfReader",
        lambda _path: SimpleNamespace(pages=[fake_page]),
    )

    with pytest.raises(UnsafeImageError):
        load_document(scanned_pdf, ocr_languages=["tr"])


def test_scanned_pdf_propagates_unsafe_image_error_for_decompression_bomb_stream(
    scanned_pdf, monkeypatch
):
    buf = io.BytesIO()
    Image.new("1", (4500, 4500), 0).save(buf, "PNG")
    fake_page = SimpleNamespace(
        extract_text=lambda: "",
        images=[SimpleNamespace(data=buf.getvalue())],
    )
    monkeypatch.setattr(
        document_loader,
        "PdfReader",
        lambda _path: SimpleNamespace(pages=[fake_page]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with pytest.raises(UnsafeImageError):
            load_document(scanned_pdf, ocr_languages=["tr"])
