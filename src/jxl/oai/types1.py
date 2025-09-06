from pydantic import BaseModel


class LlmCfg(BaseModel):
    """OCR 信息"""

    base_url: str
    """API URL"""

    api_key: str
    """API Key"""

    model: str
    """OCR 文本"""
