"""
hybrid_recommender.py
─────────────────────
Combines ALS collaborative filtering and content-based filtering into a single
weighted hybrid ranker.

Fusion Strategy
---------------
  hybrid_score = α × als_score_norm + (1 - α) × content_score

where α = ALS_WEIGHT (default 0.6).

Scores from both models are min-max normalised before blending so they are on
the same [0, 1] scale.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from preprocessing   import get_spark, run_preprocessing
from als_model       import ALSRecommender
from content_based   import ContentBasedRecommender

# ── Tuneable Hyper-parameters ─────────────────────────────────────────────────
ALS_WEIGHT     = 0.6   # weight on collaborative signal
CONTENT_WEIGHT = 1 - ALS_WEIGHT
TOP_N          = 10    # default number of final recommendations
# ─────────────────────────────────────────────────────────────────────────────


def _minmax_norm(values: list[float]) -> list[float]:
    """Normalise a list of floats to [0, 1]."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class HybridRecommender:
    """
    Hybrid recommender: wraps ALSRecommender + ContentBasedRecommender
    and blends their scores.
    """

    def __init__(
        self,
        als_weight: float    = ALS_WEIGHT,
        als_rank: int        = 20,
        als_max_iter: int    = 15,
        als_reg_param: float = 0.1,
        cb_num_features: int = 512,
    ):
        self.als_weight = als_weight
        self.cb_weight  = 1 - als_weight

        self.als = ALSRecommender(
            rank      = als_rank,
            max_iter  = als_max_iter,
            reg_param = als_reg_param,
        )
        self.cb = ContentBasedRecommender(num_features=cb_num_features)

        self._ratings_df: DataFrame | None = None
        self._movies_df:  DataFrame | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: DataFrame,
        ratings_df: DataFrame,
        movies_df: DataFrame,
    ) -> "HybridRecommender":
        print("=" * 55)
        print("  Training Hybrid Recommender")
        print(f"  ALS weight: {self.als_weight}  |  CB weight: {self.cb_weight}")
        print("=" * 55)

        self._ratings_df = ratings_df
        self._movies_df  = movies_df

        self.als.fit(train_df)
        self.cb.fit(movies_df)
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test_df: DataFrame):
        print("\n── Evaluation ───────────────────────────────────")
        rmse = self.als.evaluate(test_df)
        p, r = self.als.precision_recall_at_k(test_df, k=TOP_N)
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"   F1@{TOP_N}     : {f1:.4f}")
        print("─────────────────────────────────────────────────\n")
        return {"rmse": rmse, "precision": p, "recall": r, "f1": f1}

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend(self, user_id: int, n: int = TOP_N) -> list[dict]:
        """
        Return top-*n* hybrid recommendations for *user_id*.

        Steps
        -----
        1. Get ALS top-N*3 candidates (over-fetch to allow for blending)
        2. Get CB top-N*3 candidates
        3. Union candidate pools, normalise & blend scores
        4. Return top-N
        """
        candidate_n = n * 3

        # ── Step 1: ALS candidates ────────────────────────────────────────────
        als_df = self.als.recommend_for_user(user_id, candidate_n)
        als_rows = als_df.collect()
        als_map  = {r["movie_id"]: r["als_score"] for r in als_rows}

        # ── Step 2: CB candidates ─────────────────────────────────────────────
        cb_rows = self.cb.recommend_for_user(
            user_id, self._ratings_df, n=candidate_n
        )
        cb_map  = {r["movie_id"]: r["content_score"] for r in cb_rows}

        # ── Step 3: Union & blend ─────────────────────────────────────────────
        all_ids = set(als_map.keys()) | set(cb_map.keys())

        als_scores  = [als_map.get(mid, 0.0) for mid in all_ids]
        cb_scores   = [cb_map.get(mid, 0.0)  for mid in all_ids]

        als_norm = _minmax_norm(als_scores)
        cb_norm  = _minmax_norm(cb_scores)

        scored = []
        for mid, a, c in zip(all_ids, als_norm, cb_norm):
            hybrid = self.als_weight * a + self.cb_weight * c
            scored.append((mid, hybrid, a, c))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Collect movie titles
        movie_titles: dict[int, str] = {}
        if self._movies_df is not None:
            rows = self._movies_df.select("movie_id", "title").collect()
            movie_titles = {r["movie_id"]: r["title"] for r in rows}

        results = []
        for mid, hybrid, a, c in scored[:n]:
            results.append({
                "movie_id":     mid,
                "title":        movie_titles.get(mid, "Unknown"),
                "hybrid_score": round(hybrid, 6),
                "als_score":    round(a, 6),
                "content_score": round(c, 6),
            })
        return results

    def recommend_similar(self, movie_id: int, n: int = TOP_N) -> list[dict]:
        """Delegate to content-based similar movie lookup."""
        return self.cb.recommend_similar_movies(movie_id, n=n)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        self.als.save()
        self.cb.save()

    def load(self) -> "HybridRecommender":
        self.als.load()
        self.cb.load()
        return self


# ── End-to-End Pipeline ───────────────────────────────────────────────────────

def main():
    spark = get_spark()
    ratings, movies, train, test = run_preprocessing(spark)

    hybrid = HybridRecommender(
        als_weight   = ALS_WEIGHT,
        als_rank     = 20,
        als_max_iter = 15,
    )
    hybrid.fit(train, ratings, movies)

    metrics = hybrid.evaluate(test)
    print("Final Metrics:", metrics)

    # Demo recommendations for a few users
    for uid in [1, 42, 100]:
        print(f"\n🎬 Top-{TOP_N} hybrid recommendations for user #{uid}:")
        recs = hybrid.recommend(uid, n=TOP_N)
        for i, r in enumerate(recs, 1):
            print(
                f"  {i:2d}. [{r['movie_id']:4d}] {r['title']:<45} "
                f"hybrid={r['hybrid_score']:.4f} "
                f"(als={r['als_score']:.3f} cb={r['content_score']:.3f})"
            )

    spark.stop()


if __name__ == "__main__":
    main()
