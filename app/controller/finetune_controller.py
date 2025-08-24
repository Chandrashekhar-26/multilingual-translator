from fastapi import Response
from fastapi import APIRouter
from ..schema import TranslationRequest
from ..service import translation_service

translation_router = APIRouter(
    prefix="",
    tags=["Multilingual Translator FineTune"]
)


@translation_router.post("/")
def fine_tune():
    translation_service.init_finetune()

    return 'Model Finetuning initiated'
