"""
main.py — FastAPI application entry point
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .encoder import get_model
from .routes import router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: load model weights once at startup
    print("⏳ Loading sentence-transformer model…")
    get_model()
    print("✅ Model ready.")
    yield


app = FastAPI(
    title="AttentionSeeker API",
    description="Interactive sentence transformer visualizer — backend",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow requests from the Vite dev server and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:80",     # Nginx / Docker
        "http://localhost",
        "*",                       # HF Spaces (public demo)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress large payloads (layer hidden states can be ~700 KB)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(router)
