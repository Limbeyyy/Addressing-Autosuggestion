# local hosting entry point
from fastapi import FastAPI
from core.middleware import setup_middleware
from autocomplete_api import router as auto_router

app = FastAPI(title="Nepali Autosuggest API")

setup_middleware(app)
app.include_router(auto_router)

@app.get("/")
def root():
    return {"status": "API Running"}
