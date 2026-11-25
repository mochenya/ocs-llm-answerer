"""在 HTTP 边界校验 OCS 图片，并生成适合模型输入的 Data URL。"""

import base64
import hashlib

from ocs_llm_answerer.answer.models import QuestionImage

MAX_IMAGES = 16
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_IMAGE_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)


def parse_images(data_urls: list[str]) -> list[QuestionImage]:
    """保留图片顺序，校验 Base64、图片格式和解码后的大小。"""
    images: list[QuestionImage] = []
    total_bytes = 0
    for index, data_url in enumerate(data_urls):
        prefix, separator, encoded = data_url.partition(",")
        field = f"images[{index}]"
        if not separator or not prefix.startswith("data:image/") or not prefix.endswith(";base64"):
            raise ValueError(f"{field} must be a Base64 image data URL")
        if len(encoded) > _MAX_IMAGE_BASE64_LENGTH:
            raise ValueError(f"{field} exceeds the 5 MiB decoded image limit")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError(f"{field} contains invalid Base64") from exc
        size = len(image_bytes)
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"{field} exceeds the 5 MiB decoded image limit")
        total_bytes += size
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("images exceed the 16 MiB combined decoded image limit")
        media_type = _identify_media_type(image_bytes)
        if media_type is None:
            raise ValueError(f"{field} must contain PNG, JPEG or WebP bytes")
        images.append(
            QuestionImage(
                media_type=media_type,
                base64_data=encoded,
                sha256=hashlib.sha256(image_bytes).hexdigest(),
            )
        )
    return images


def _identify_media_type(data: bytes) -> str | None:
    # OCS 可能只改写前缀为 image/png，原始图片字节仍然是 JPEG 或 WebP。
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None
