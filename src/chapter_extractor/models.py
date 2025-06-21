from __future__ import annotations

from dataclasses import dataclass, asdict


class BaseModel:
    """Minimal pydantic-like base model."""

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class Chapter(BaseModel):
    chapter_name: str
    start_page: int
    end_page: int
