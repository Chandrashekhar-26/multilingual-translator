from fastapi import FastAPI
from config import Config
from app.controller import translation_controller


def create_app():
    app = FastAPI()
    # CORS(app)
    register_routes(app, Config.BASE_URL, Config.API_VERSION)

    return app


def register_routes(app, base_url, api_version):
    app.include_router(translation_controller.translation_router, prefix=f"/{base_url}/{api_version}/translate")
