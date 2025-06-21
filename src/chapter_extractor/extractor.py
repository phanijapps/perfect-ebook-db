from __future__ import annotations

import math
from typing import Iterable, List

from .models import Chapter


def read_pdf_pages(pdf_path: str) -> List[str]:
    """Read pages from a PDF using pypdf."""
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def _features(line: str) -> List[float]:
    stripped = line.strip()
    length = len(stripped)
    words = len(stripped.split())
    upper_ratio = sum(c.isupper() for c in stripped) / length if length else 0
    digit_ratio = sum(c.isdigit() for c in stripped) / length if length else 0
    has_keyword = int("chapter" in stripped.lower())
    return [length, words, upper_ratio, digit_ratio, has_keyword]


class SimpleLogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 100):
        self.lr = lr
        self.epochs = epochs
        self.weights: List[float] | None = None
        self.bias: float = 0.0

    def _sigmoid(self, z: float) -> float:
        return 1 / (1 + math.exp(-z))

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        if not X:
            raise ValueError("Training data is empty")
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        for _ in range(self.epochs):
            for x_vec, y_true in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, x_vec)) + self.bias
                y_pred = self._sigmoid(z)
                error = y_pred - y_true
                for i in range(n_features):
                    self.weights[i] -= self.lr * error * x_vec[i]
                self.bias -= self.lr * error

    def predict_prob(self, X: Iterable[List[float]]) -> List[float]:
        if self.weights is None:
            raise ValueError("Model not trained")
        probs = []
        for x_vec in X:
            z = sum(w * x for w, x in zip(self.weights, x_vec)) + self.bias
            probs.append(self._sigmoid(z))
        return probs


class PDFChapterExtractor:
    """Extract chapters from a PDF."""

    def __init__(self) -> None:
        self.model = SimpleLogisticRegression()

    def _train(self, pages: List[str]) -> None:
        features = []
        labels = []
        for page in pages:
            for line in page.splitlines():
                f = _features(line)
                label = 1 if "chapter" in line.lower() else 0
                features.append(f)
                labels.append(label)
        if len(set(labels)) > 1:
            self.model.fit(features, labels)
        else:
            self.model.weights = [0.0] * len(features[0])

    def extract(self, pdf_path: str) -> List[Chapter]:
        pages = read_pdf_pages(pdf_path)
        self._train(pages)

        chapters: List[Chapter] = []
        chapter_pages = []
        for page_idx, text in enumerate(pages, start=1):
            lines = text.splitlines()
            probs = self.model.predict_prob(
                [_features(line) for line in lines]
            )
            for line, prob in zip(lines, probs):
                if prob >= 0.5 and line.lower().startswith("chapter"):
                    chapter_pages.append((line.strip(), page_idx))

        total_pages = len(pages)
        for i, (name, start) in enumerate(chapter_pages):
            end = (
                chapter_pages[i + 1][1] - 1
                if i + 1 < len(chapter_pages)
                else total_pages
            )
            chapters.append(
                Chapter(chapter_name=name, start_page=start, end_page=end)
            )
        return chapters
