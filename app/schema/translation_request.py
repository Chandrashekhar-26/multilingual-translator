from pydantic import BaseModel, Field
from typing import Literal


class TranslationRequest(BaseModel):
    text: str = Field(..., description='Text')
    source_language: Literal[
        'ENGLISH', 'HINDI', 'MARATHI', 'TAMIL', 'KANNADA'
    ] = Field(..., description="Source Language")
    target_language: Literal[
        'ENGLISH', 'HINDI', 'MARATHI', 'TAMIL', 'KANNADA'
    ] = Field(..., description="Target Language")
