from fastapi import FastAPI

from app.routes.test import router as test_router

app = FastAPI()

app.include_router(test_router)




