from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Tuple

from .models import Chapter


def read_pdf_pages(pdf_path: str) -> List[str]:
    """Read pages from a PDF using pypdf."""
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def _features(line: str) -> List[float]:
    """Compute simple text features."""
    stripped = line.strip()
    length = len(stripped)
    words = len(stripped.split())
    upper_ratio = sum(c.isupper() for c in stripped) / length if length else 0
    digit_ratio = sum(c.isdigit() for c in stripped) / length if length else 0
    has_keyword = int("chapter" in stripped.lower())
    return [length, words, upper_ratio, digit_ratio, has_keyword]


class SimpleLogisticRegression:
    """Minimal logistic regression supporting multiprocessing."""

    def __init__(self, lr: float = 0.1, epochs: int = 100, n_jobs: int = 1):
        self.lr = lr
        self.epochs = epochs
        self.n_jobs = max(1, n_jobs)
        self.weights: List[float] | None = None
        self.bias: float = 0.0

    def _sigmoid(self, z: float) -> float:
        """Numerically stable sigmoid."""
        if z >= 0:
            z = min(z, 500)
            return 1 / (1 + math.exp(-z))
        z = max(z, -500)
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)

    @staticmethod
    def _partial_grad(
        weights: List[float],
        bias: float,
        X: List[List[float]],
        y: List[int],
    ) -> Tuple[List[float], float]:
        grad_w = [0.0] * len(weights)
        grad_b = 0.0
        for x_vec, y_true in zip(X, y):
            z = sum(w * x for w, x in zip(weights, x_vec)) + bias
            y_pred = 1 / (1 + math.exp(-max(min(z, 500), -500)))
            error = y_pred - y_true
            for i in range(len(weights)):
                grad_w[i] += error * x_vec[i]
            grad_b += error
        return grad_w, grad_b

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        if not X:
            raise ValueError("Training data is empty")
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        for _ in range(self.epochs):
            if self.n_jobs > 1 and len(X) >= self.n_jobs * 2:
                chunk_size = (len(X) + self.n_jobs - 1) // self.n_jobs
                args = [
                    (
                        self.weights,
                        self.bias,
                        X[i : i + chunk_size],
                        y[i : i + chunk_size],
                    )
                    for i in range(0, len(X), chunk_size)
                ]
                with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
                    results = list(ex.map(_call_partial_grad, args))
                grad_w = [0.0] * n_features
                grad_b = 0.0
                for gw, gb in results:
                    for i in range(n_features):
                        grad_w[i] += gw[i]
                    grad_b += gb
            else:
                grad_w, grad_b = self._partial_grad(self.weights, self.bias, X, y)

            for i in range(n_features):
                self.weights[i] -= self.lr * grad_w[i]
            self.bias -= self.lr * grad_b

    def predict_prob(self, X: Iterable[List[float]]) -> List[float]:
        if self.weights is None:
            raise ValueError("Model not trained")
        probs = []
        for x_vec in X:
            z = sum(w * x for w, x in zip(self.weights, x_vec)) + self.bias
            probs.append(self._sigmoid(z))
        return probs


def _call_partial_grad(args: Tuple[List[float], float, List[List[float]], List[int]]) -> Tuple[List[float], float]:
    """Wrapper so ``ProcessPoolExecutor`` can call ``_partial_grad``."""
    return SimpleLogisticRegression._partial_grad(*args)


class PDFChapterExtractor:
    """Extract chapters from a PDF."""

    def __init__(self, n_jobs: int | None = None) -> None:
        jobs = n_jobs or os.cpu_count() or 1
        self.n_jobs = max(1, jobs)
        self.model = SimpleLogisticRegression(n_jobs=self.n_jobs)

    def _process_page(self, page: str) -> Tuple[List[List[float]], List[int]]:
        feats = []
        labs = []
        for line in page.splitlines():
            feats.append(_features(line))
            labs.append(1 if "chapter" in line.lower() else 0)
        return feats, labs

    def _train(self, pages: List[str]) -> None:
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            results = list(ex.map(self._process_page, pages))

        features: List[List[float]] = []
        labels: List[int] = []
        for feats, labs in results:
            features.extend(feats)
            labels.extend(labs)

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
