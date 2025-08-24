from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from datetime import datetime
import uvicorn
from app import create_app
from config import Config

app = create_app()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def startup_event():
    print(f' ***** App Running')


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.get('/')
# def redirect_to():
#     redirect_url = f'/{Config.BASE_URL}/{Config.API_VERSION}/info'
#     return RedirectResponse(url=redirect_url)


@app.get(f'/{Config.BASE_URL}/{Config.API_VERSION}/info')
def get_info():
    return {
        'App': 'Multilingual Translation API',
        'Version': '1.0.0',
        'Time': datetime.now()
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=Config.HOST, port=Config.PORT, reload=False)
