"""Minimal backport of Python's removed imghdr module for compatibility.

Streamlit <1.30 expects imghdr (removed in Python 3.13+). This file provides
the same interface used by Streamlit: `what(file, h=None)` returns a short
image type string or None.
"""

from binascii import b2a_uu


def _accept(prefix: bytes, data: bytes) -> bool:
    return data.startswith(prefix)


def what(file, h=None):
    """Identify image type based on header."""
    if h is None:
        if hasattr(file, "read"):
            pos = file.tell()
            h = file.read(32)
            file.seek(pos)
        else:
            with open(file, "rb") as f:
                h = f.read(32)

    if len(h) < 4:
        return None

    # JPEG
    if h[0:3] == b"\xff\xd8\xff":
        return "jpeg"
    # PNG
    if _accept(b"\211PNG\r\n\032\n", h):
        return "png"
    # GIF
    if _accept(b"GIF87a", h) or _accept(b"GIF89a", h):
        return "gif"
    # TIFF
    if _accept(b"MM\x00*", h) or _accept(b"II*\x00", h):
        return "tiff"
    # BMP
    if _accept(b"BM", h):
        return "bmp"
    # WebP
    if _accept(b"RIFF", h[0:4]) and _accept(b"WEBP", h[8:12]):
        return "webp"

    # Old-style XPM/SGI headers fall back to None here.
    return None


__all__ = ["what"]
