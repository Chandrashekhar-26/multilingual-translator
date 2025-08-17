from fastapi import Response
from fastapi import APIRouter
from ..schema import TranslationRequest
from ..service import translation_service

translation_router = APIRouter(
    prefix="",
    tags=["Multilingual Translator"]
)


@translation_router.post("/")
def predict(translation_request: TranslationRequest):

    result = translation_service.translate(translation_request)

    return result
