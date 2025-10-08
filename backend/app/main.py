from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import health, predict, predict_multi

app = FastAPI(title = "AI Meal Calorie Estimator")

#Cross-Origin Resource Sharing (CORS): Server permits web page to requests resources from it
#React page (http://localhost:5173) can make requests to this FastAPI server (http://localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix = "")
app.include_router(predict.router, prefix = "")
app.include_router(predict_multi.router, prefix = "")