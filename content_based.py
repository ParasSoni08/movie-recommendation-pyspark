"""
content_based.py
────────────────
Content-Based Filtering using movie metadata (genres + title keywords).

Pipeline
--------
1. Build a TF-IDF feature matrix from genre strings (via PySpark ML)
2. Compute cosine similarity between movies using item vectors
3. Expose `recommend_similar_movies()` — given a movie, return the most similar
4. Expose `recommend_for_user()` — build a user profile from rated movies
   and score all candidate movies against that profile
"""

import os
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Normalizer
from pyspark.ml import Pipeline, PipelineModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cb_model")


class ContentBasedRecommender:
    """TF-IDF + cosine-similarity content-based recommender."""

    def __init__(self, num_features: int = 512):
        self.num_features = num_features
        self.pipeline_model: PipelineModel | None = None
        self._movie_vectors: dict[int, np.ndarray] = {}   # movie_id → L2-normed vector
        self._movie_titles:  dict[int, str]         = {}

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _build_feature_df(self, movies_df: DataFrame) -> DataFrame:
        """
        Turn each movie into a TF-IDF vector.
        Feature text = genre_string + title tokens.
        """
        # Combine genre tags and title into a single text field
        featured = movies_df.withColumn(
            "text",
            F.concat_ws(" ",
                F.col("genre_string"),
                F.regexp_replace(F.lower(F.col("title")), r"[^a-z0-9 ]", ""),
            )
        )
        return featured

    def fit(self, movies_df: DataFrame) -> "ContentBasedRecommender":
        """Build TF-IDF pipeline and cache movie vectors."""
        print("🔧 Building TF-IDF content features...")
        featured = self._build_feature_df(movies_df)

        tokenizer  = Tokenizer(inputCol="text", outputCol="tokens")
        hashing_tf = HashingTF(
            inputCol="tokens", outputCol="raw_tf",
            numFeatures=self.num_features
        )
        idf        = IDF(inputCol="raw_tf", outputCol="tfidf")
        normalizer = Normalizer(inputCol="tfidf", outputCol="features", p=2.0)

        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, normalizer])
        self.pipeline_model = pipeline.fit(featured)
        transformed = self.pipeline_model.transform(featured)

        # Cache as Python dicts for fast cosine similarity lookup
        rows = transformed.select("movie_id", "title", "features").collect()
        for row in rows:
            vec = row["features"]
            self._movie_vectors[row["movie_id"]] = np.array(vec.toArray())
            self._movie_titles[row["movie_id"]]  = row["title"]

        print(f"✅ Content model built for {len(self._movie_vectors):,} movies")
        return self

    # ── Cosine Similarity Helpers ─────────────────────────────────────────────

    def _cosine_sim(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity (vectors are already L2-normalised → just dot product)."""
        return float(np.dot(vec_a, vec_b))

    def _score_all_movies(self, query_vec: np.ndarray) -> list[tuple[int, float]]:
        """Score every movie in the index against *query_vec*."""
        scores = [
            (mid, self._cosine_sim(query_vec, vec))
            for mid, vec in self._movie_vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend_similar_movies(
        self, movie_id: int, n: int = 10, exclude_self: bool = True
    ) -> list[dict]:
        """
        Return *n* movies most similar to *movie_id* by content.
        Returns list of dicts with keys: movie_id, title, content_score.
        """
        if movie_id not in self._movie_vectors:
            raise ValueError(f"movie_id {movie_id} not found in content index")

        query_vec = self._movie_vectors[movie_id]
        ranked    = self._score_all_movies(query_vec)

        results = []
        for mid, score in ranked:
            if exclude_self and mid == movie_id:
                continue
            results.append({
                "movie_id":      mid,
                "title":         self._movie_titles.get(mid, ""),
                "content_score": round(score, 6),
            })
            if len(results) >= n:
                break
        return results

    def recommend_for_user(
        self,
        user_id: int,
        ratings_df: DataFrame,
        n: int = 10,
        rating_threshold: float = 3.5,
        already_seen: set[int] | None = None,
    ) -> list[dict]:
        """
        Build a user profile vector from movies the user liked (≥ threshold),
        then score all unseen movies.

        Parameters
        ----------
        user_id           : target user
        ratings_df        : Spark DataFrame with columns (user_id, movie_id, rating)
        n                 : number of recommendations
        rating_threshold  : minimum rating to count as 'liked'
        already_seen      : set of movie_ids to exclude (defaults to all rated movies)
        """
        # Collect this user's highly rated movies
        liked = (
            ratings_df
            .filter(
                (F.col("user_id") == user_id) &
                (F.col("rating") >= rating_threshold)
            )
            .select("movie_id", "rating")
            .collect()
        )

        if not liked:
            return []

        # Weighted average of liked movie vectors (weight = rating)
        profile_vec = np.zeros(self.num_features)
        total_weight = 0.0
        for row in liked:
            mid    = row["movie_id"]
            weight = row["rating"]
            if mid in self._movie_vectors:
                profile_vec  += weight * self._movie_vectors[mid]
                total_weight += weight

        if total_weight == 0:
            return []

        profile_vec /= total_weight
        # Re-normalise
        norm = np.linalg.norm(profile_vec)
        if norm > 0:
            profile_vec /= norm

        # Movies already interacted with
        if already_seen is None:
            seen_rows = (
                ratings_df.filter(F.col("user_id") == user_id)
                .select("movie_id").collect()
            )
            already_seen = {r["movie_id"] for r in seen_rows}

        ranked = self._score_all_movies(profile_vec)

        results = []
        for mid, score in ranked:
            if mid in already_seen:
                continue
            results.append({
                "movie_id":      mid,
                "title":         self._movie_titles.get(mid, ""),
                "content_score": round(score, 6),
            })
            if len(results) >= n:
                break
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        import pickle
        os.makedirs(path, exist_ok=True)
        self.pipeline_model.save(os.path.join(path, "pipeline"))
        with open(os.path.join(path, "vectors.pkl"), "wb") as f:
            pickle.dump({
                "movie_vectors": self._movie_vectors,
                "movie_titles":  self._movie_titles,
            }, f)
        print(f"💾 Content model saved → {path}")

    def load(self, path: str = MODEL_PATH) -> "ContentBasedRecommender":
        import pickle
        self.pipeline_model = PipelineModel.load(os.path.join(path, "pipeline"))
        with open(os.path.join(path, "vectors.pkl"), "rb") as f:
            data = pickle.load(f)
        self._movie_vectors = data["movie_vectors"]
        self._movie_titles  = data["movie_titles"]
        print(f"📂 Content model loaded ← {path}")
        return self


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import get_spark, load_ratings, load_movies

    spark = get_spark()
    ratings = load_ratings(spark)
    movies  = load_movies(spark)

    cb = ContentBasedRecommender()
    cb.fit(movies)

    print("\n🎬 Movies similar to 'Toy Story (1995)' [movie_id=1]:")
    for r in cb.recommend_similar_movies(1, n=10):
        print(f"  [{r['movie_id']:4d}] {r['title']:<45} score={r['content_score']:.4f}")

    print("\n👤 Content-based recommendations for user #1:")
    for r in cb.recommend_for_user(1, ratings, n=10):
        print(f"  [{r['movie_id']:4d}] {r['title']:<45} score={r['content_score']:.4f}")

    spark.stop()
