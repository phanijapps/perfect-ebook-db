from chapter_extractor.extractor import PDFChapterExtractor, read_pdf_pages
from chapter_extractor.models import Chapter


def test_extract_chapters(monkeypatch):
    pages = [
        "Chapter 1\nIntro text",
        "More about chapter 1",
        "Chapter 2\nDetails",
    ]

    def fake_read_pdf(_):
        return pages

    monkeypatch.setattr(
        "chapter_extractor.extractor.read_pdf_pages", fake_read_pdf
    )

    extractor = PDFChapterExtractor()
    chapters = extractor.extract("dummy.pdf")

    assert chapters == [
        Chapter(chapter_name="Chapter 1", start_page=1, end_page=2),
        Chapter(chapter_name="Chapter 2", start_page=3, end_page=3),
    ]
