from pydantic import BaseModel


class LlmCfg(BaseModel):
    """OCR 信息"""

    api_key: str
    """API Key"""

    base_url: str
    """API URL"""

    model: str
    """OCR 文本"""
