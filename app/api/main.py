from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up engine (loads model)
    try:
        from .dependencies import get_engine
        get_engine()
    except Exception as e:
        print(f"[warn] Engine not warmed: {e}")
    yield


app = FastAPI(
    title="Real-Time Anomaly Detection API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}