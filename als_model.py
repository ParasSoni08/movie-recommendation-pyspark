"""
als_model.py
────────────
Alternating Least Squares (ALS) collaborative filtering via PySpark MLlib.

Key responsibilities
--------------------
- Train ALS on the ratings training split
- Evaluate with RMSE on the held-out test split
- Expose `recommend_for_user()` and `recommend_for_all_users()`
- Persist / load the trained model
"""

import os
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "als_model")


class ALSRecommender:
    """Thin wrapper around PySpark ALS for the movie recommendation pipeline."""

    def __init__(
        self,
        rank: int        = 20,
        max_iter: int    = 15,
        reg_param: float = 0.1,
        cold_start: str  = "drop",   # drop | nan
        seed: int        = 42,
    ):
        self.rank       = rank
        self.max_iter   = max_iter
        self.reg_param  = reg_param
        self.cold_start = cold_start
        self.seed       = seed
        self.model: ALSModel | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, train_df: DataFrame) -> "ALSRecommender":
        """Train ALS on *train_df* (requires columns: user_id, movie_id, rating)."""
        print(f"🔧 Training ALS  rank={self.rank}  iter={self.max_iter}  λ={self.reg_param}")
        als = ALS(
            rank            = self.rank,
            maxIter         = self.max_iter,
            regParam        = self.reg_param,
            userCol         = "user_id",
            itemCol         = "movie_id",
            ratingCol       = "rating",
            coldStartStrategy = self.cold_start,
            seed            = self.seed,
            implicitPrefs   = False,
        )
        self.model = als.fit(train_df)
        print("✅ ALS training complete")
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test_df: DataFrame) -> float:
        """Return RMSE on *test_df*."""
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate()")
        predictions = self.model.transform(test_df)
        evaluator = RegressionEvaluator(
            metricName  = "rmse",
            labelCol    = "rating",
            predictionCol = "prediction",
        )
        rmse = evaluator.evaluate(predictions.dropna(subset=["prediction"]))
        print(f"📊 ALS RMSE on test set: {rmse:.4f}")
        return rmse

    def precision_recall_at_k(self, test_df: DataFrame, k: int = 10, threshold: float = 3.5):
        """
        Compute Precision@K and Recall@K.
        Relevant = actual rating >= threshold.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate()")

        # Ground truth relevant items per user
        relevant = (
            test_df.filter(F.col("rating") >= threshold)
            .groupBy("user_id")
            .agg(F.collect_set("movie_id").alias("relevant_items"))
        )

        # Top-K predicted items per user
        recs = self.model.recommendForAllUsers(k)
        recs = recs.withColumn(
            "recommended_items",
            F.col("recommendations.movie_id")
        ).select("user_id", "recommended_items")

        joined = relevant.join(recs, "user_id")

        def intersect_size(rel, rec):
            return len(set(rel) & set(rec)) if rel and rec else 0

        intersect_udf = F.udf(intersect_size, "int")

        joined = joined.withColumn(
            "hits",
            intersect_udf(F.col("relevant_items"), F.col("recommended_items"))
        )

        metrics = joined.agg(
            (F.sum("hits") / (F.count("hits") * k)).alias("precision_at_k"),
            (F.sum("hits") / F.sum(F.size("relevant_items"))).alias("recall_at_k"),
        ).collect()[0]

        p, r = metrics["precision_at_k"], metrics["recall_at_k"]
        print(f"📊 ALS  Precision@{k}: {p:.4f}  |  Recall@{k}: {r:.4f}")
        return p, r

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend_for_user(self, user_id: int, n: int = 10) -> DataFrame:
        """Return top-*n* ALS recommendations for a single user."""
        if self.model is None:
            raise RuntimeError("Call fit() before recommending")
        spark = SparkSession.getActiveSession()
        user_df = spark.createDataFrame([(user_id,)], ["user_id"])
        recs = self.model.recommendForUserSubset(user_df, n)
        return (
            recs.withColumn("rec", F.explode("recommendations"))
                .select(
                    F.col("user_id"),
                    F.col("rec.movie_id").alias("movie_id"),
                    F.col("rec.rating").alias("als_score"),
                )
        )

    def recommend_for_all_users(self, n: int = 10) -> DataFrame:
        """Return top-*n* ALS recommendations for every user."""
        if self.model is None:
            raise RuntimeError("Call fit() before recommending")
        recs = self.model.recommendForAllUsers(n)
        return (
            recs.withColumn("rec", F.explode("recommendations"))
                .select(
                    F.col("user_id"),
                    F.col("rec.movie_id").alias("movie_id"),
                    F.col("rec.rating").alias("als_score"),
                )
        )

    def get_item_factors(self) -> DataFrame:
        """Return the learned item (movie) latent factor vectors."""
        return self.model.itemFactors  # columns: id, features

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"💾 ALS model saved → {path}")

    def load(self, path: str = MODEL_PATH) -> "ALSRecommender":
        self.model = ALSModel.load(path)
        print(f"📂 ALS model loaded ← {path}")
        return self


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import get_spark, run_preprocessing

    spark = get_spark()
    _, _, train, test = run_preprocessing(spark)

    rec = ALSRecommender(rank=20, max_iter=15, reg_param=0.1)
    rec.fit(train)
    rec.evaluate(test)
    rec.precision_recall_at_k(test, k=10)

    print("\n🎬 Top-10 recommendations for user #1:")
    rec.recommend_for_user(1, 10).show(truncate=False)

    spark.stop()
