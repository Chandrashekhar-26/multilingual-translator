from pydantic import BaseModel, Field
from typing import Literal


class TranslationRequest(BaseModel):
    text: str = Field(..., description='Text')
    source_language: Literal[
        'HINDI', 'ENGLISH', 'MARATHI'
    ] = Field(..., description="Source Language")
    target_language: Literal[
        'ENGLISH', 'HINDI', 'MARATHI'
    ] = Field(..., description="Target Language")
