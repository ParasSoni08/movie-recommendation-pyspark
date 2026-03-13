"""
api/app.py
──────────
FastAPI REST service for the Hybrid Movie Recommender.

Endpoints
---------
GET /health                      → service health check
GET /movies                      → paginated list of all movies
GET /recommend?user_id=1&n=10   → top-N hybrid recommendations
GET /similar?movie_id=1&n=10    → content-similar movies

Run locally
-----------
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing      import get_spark, run_preprocessing
from hybrid_recommender import HybridRecommender

# ── Globals (populated at startup) ────────────────────────────────────────────
_recommender: HybridRecommender | None = None
_movie_catalog: list[dict] = []
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; release resources on shutdown."""
    global _recommender, _movie_catalog, _start_time
    _start_time = time.time()

    print("🚀 Starting Movie Recommender API...")
    spark = get_spark("MovieRecommenderAPI")
    ratings, movies, train, test = run_preprocessing(spark)

    _recommender = HybridRecommender()
    _recommender.fit(train, ratings, movies)

    # Cache movie catalog as plain Python list
    _movie_catalog = [
        {
            "movie_id":    r["movie_id"],
            "title":       r["title"],
            "genres":      r["genres"],
            "genre_string": r["genre_string"],
        }
        for r in movies.collect()
    ]

    print(f"✅ API ready in {time.time() - _start_time:.1f}s")
    yield

    # Shutdown
    spark.stop()
    print("🛑 Spark stopped. API shut down.")


app = FastAPI(
    title       = "🎬 Movie Recommender API",
    description = "Hybrid ALS + Content-Based recommendations powered by PySpark",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Response Models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:   str
    uptime_s: float
    model:    str

class MovieItem(BaseModel):
    movie_id:    int
    title:       str
    genre_string: str

class RecommendationItem(BaseModel):
    rank:          int
    movie_id:      int
    title:         str
    hybrid_score:  float
    als_score:     float
    content_score: float

class SimilarMovieItem(BaseModel):
    rank:          int
    movie_id:      int
    title:         str
    content_score: float

class RecommendResponse(BaseModel):
    user_id:         int
    n:               int
    recommendations: list[RecommendationItem]

class SimilarResponse(BaseModel):
    movie_id:    int
    movie_title: str
    n:           int
    similar:     list[SimilarMovieItem]

class MoviesResponse(BaseModel):
    total:  int
    page:   int
    limit:  int
    movies: list[MovieItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Service liveness check."""
    return {
        "status":   "ok",
        "uptime_s": round(time.time() - _start_time, 1),
        "model":    "Hybrid ALS + Content-Based (MovieLens 100K)",
    }


@app.get("/movies", response_model=MoviesResponse, tags=["Catalog"])
def list_movies(
    page:  int = Query(1, ge=1,   description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Movies per page"),
):
    """
    Paginated list of all movies in the catalog.

    Example: `GET /movies?page=2&limit=10`
    """
    start  = (page - 1) * limit
    end    = start + limit
    subset = _movie_catalog[start:end]

    return {
        "total":  len(_movie_catalog),
        "page":   page,
        "limit":  limit,
        "movies": [
            {
                "movie_id":    m["movie_id"],
                "title":       m["title"],
                "genre_string": m["genre_string"],
            }
            for m in subset
        ],
    }


@app.get("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(
    user_id: int = Query(..., ge=1, description="Target user ID"),
    n:       int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Return top-N **hybrid** (ALS + content-based) movie recommendations for a user.

    Example: `GET /recommend?user_id=1&n=10`
    """
    if _recommender is None:
        raise HTTPException(503, "Model not loaded yet")

    try:
        recs = _recommender.recommend(user_id, n=n)
    except Exception as exc:
        raise HTTPException(500, f"Recommendation error: {exc}") from exc

    if not recs:
        raise HTTPException(404, f"No recommendations found for user_id={user_id}. "
                                  "The user may be unknown or have no ratings.")

    return {
        "user_id": user_id,
        "n":       n,
        "recommendations": [
            {**rec, "rank": i + 1}
            for i, rec in enumerate(recs)
        ],
    }


@app.get("/similar", response_model=SimilarResponse, tags=["Recommendations"])
def similar_movies(
    movie_id: int = Query(..., ge=1, description="Seed movie ID"),
    n:        int = Query(10, ge=1, le=50, description="Number of similar movies"),
):
    """
    Return *n* movies most similar to a given movie by content features.

    Example: `GET /similar?movie_id=1&n=10`
    """
    if _recommender is None:
        raise HTTPException(503, "Model not loaded yet")

    # Look up seed movie title
    seed_title = next(
        (m["title"] for m in _movie_catalog if m["movie_id"] == movie_id),
        None,
    )
    if seed_title is None:
        raise HTTPException(404, f"movie_id={movie_id} not found in catalog")

    try:
        similar = _recommender.recommend_similar(movie_id, n=n)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    return {
        "movie_id":    movie_id,
        "movie_title": seed_title,
        "n":           n,
        "similar": [
            {**rec, "rank": i + 1}
            for i, rec in enumerate(similar)
        ],
    }


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
