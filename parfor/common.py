from __future__ import annotations

import os
from typing import Protocol

cpu_count = int(os.cpu_count())


class Bar(Protocol):
    def update(self, n: int = 1) -> None: ...
