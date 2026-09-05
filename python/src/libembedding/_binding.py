"""Low-level cffi binding layer. Loads the shared library and exposes ffi/lib.

Auteur: David Orel
Version: 1.4.0
"""

import platform
from pathlib import Path

import cffi

ffi = cffi.FFI()

_cdefs_path = Path(__file__).with_name("_cdefs.h")
ffi.cdef(_cdefs_path.read_text(encoding="utf-8"))


def _find_library() -> str:
    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(
        platform.system(), ".so"
    )
    pkg_dir = Path(__file__).parent
    candidates = [
        pkg_dir / f"libembedding{ext}",
        pkg_dir / "lib" / f"libembedding{ext}",
        # Development: look in build directory relative to project root
        pkg_dir.parent.parent.parent.parent / "build" / "python" / f"libembedding{ext}",
        pkg_dir.parent.parent.parent.parent / "build" / f"libembedding{ext}",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return f"libembedding{ext}"


lib = ffi.dlopen(_find_library())

