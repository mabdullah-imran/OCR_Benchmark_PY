from pathlib import Path


def get_mime_type(url: str) -> str:
    """Return a reasonable mime type for the file at `url` based on extension.

    Defaults to 'image/png' for unknown extensions to match TS implementation.
    """
    if not url:
        return "image/png"

    # Use Path.suffix to be robust to query strings and fragments
    suffix = Path(url.split("?")[0].split("#")[0]).suffix.lower().lstrip(".")

    mapping = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }

    return mapping.get(suffix, "image/png")


__all__ = ["get_mime_type"]
