"""Low-level cffi binding layer. Loads the shared library and exposes ffi/lib."""

import cffi
import platform
from pathlib import Path

ffi = cffi.FFI()

_cdefs_path = Path(__file__).with_name("_cdefs.h")
ffi.cdef(_cdefs_path.read_text())


def _find_library() -> str:
    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(
        platform.system(), ".so"
    )
    pkg_dir = Path(__file__).parent
    candidates = [
        pkg_dir / f"libembedding{ext}",
        pkg_dir / f"lib" / f"libembedding{ext}",
        # Development: look in build directory relative to project root
        pkg_dir.parent.parent.parent.parent / "build" / f"python" / f"libembedding{ext}",
        pkg_dir.parent.parent.parent.parent / "build" / f"libembedding{ext}",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return f"libembedding{ext}"


lib = ffi.dlopen(_find_library())
