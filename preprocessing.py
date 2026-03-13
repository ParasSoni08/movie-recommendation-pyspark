"""
preprocessing.py
────────────────
Loads and prepares the MovieLens 100K dataset using PySpark.

Outputs
-------
ratings_df  : Spark DataFrame  (user_id, movie_id, rating)
movies_df   : Spark DataFrame  (movie_id, title, genres, genre_string)
train_df    : 80 % split for ALS training
test_df     : 20 % split for ALS evaluation
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, LongType, StringType
)

# ── Genre list (MovieLens 100K ordering) ──────────────────────────────────────
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")


def get_spark(app_name: str = "MovieRecommender") -> SparkSession:
    """Return (or create) a local SparkSession."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_ratings(spark: SparkSession, path: str = None) -> "DataFrame":
    """
    Load u.data → (user_id INT, movie_id INT, rating FLOAT, timestamp LONG).
    """
    path = path or os.path.join(DATA_DIR, "u.data")
    schema = StructType([
        StructField("user_id",   IntegerType(), False),
        StructField("movie_id",  IntegerType(), False),
        StructField("rating",    FloatType(),   False),
        StructField("timestamp", LongType(),    False),
    ])
    return (
        spark.read
        .option("sep", "\t")
        .schema(schema)
        .csv(path)
    )


def load_movies(spark: SparkSession, path: str = None) -> "DataFrame":
    """
    Load u.item → (movie_id, title, release_date, genres list, genre_string).
    The pipe-delimited file has 24 columns; columns 5-23 are binary genre flags.
    """
    path = path or os.path.join(DATA_DIR, "u.item")

    # Read raw lines and parse manually (encoding issues with standard CSV reader)
    raw = spark.read.option("sep", "|").option("encoding", "ISO-8859-1").csv(path)

    # Rename first three useful columns
    df = raw.select(
        F.col("_c0").cast(IntegerType()).alias("movie_id"),
        F.col("_c1").alias("title"),
        F.col("_c2").alias("release_date"),
        *[F.col(f"_c{i + 5}").cast(IntegerType()).alias(GENRES[i])
          for i in range(len(GENRES))]
    )

    # Build a human-readable genre string  e.g. "Action|Comedy"
    genre_exprs = [
        F.when(F.col(g) == 1, F.lit(g)).otherwise(F.lit(None))
        for g in GENRES
    ]
    df = df.withColumn(
        "genre_string",
        F.concat_ws("|", *[
            F.when(F.col(g) == 1, F.lit(g)) for g in GENRES
        ])
    )

    # Collect genres into an array column for content-based filtering
    df = df.withColumn(
        "genres",
        F.array(*[F.when(F.col(g) == 1, F.lit(g)) for g in GENRES])
    ).withColumn(
        "genres",
        F.expr("filter(genres, x -> x is not null)")
    )

    return df.select("movie_id", "title", "release_date", "genres", "genre_string")


def normalize_ratings(ratings_df: "DataFrame") -> "DataFrame":
    """Mean-center ratings per user to reduce rating-scale bias."""
    user_mean = ratings_df.groupBy("user_id").agg(
        F.mean("rating").alias("user_mean")
    )
    return (
        ratings_df
        .join(user_mean, "user_id")
        .withColumn("rating_norm", F.col("rating") - F.col("user_mean"))
        .drop("user_mean")
    )


def train_test_split(ratings_df: "DataFrame", train_ratio: float = 0.8, seed: int = 42):
    """Stratified split: hold out the last 20 % of each user's ratings by time."""
    w = (
        __import__("pyspark.sql.window", fromlist=["Window"])
        .Window.partitionBy("user_id").orderBy(F.desc("timestamp"))
    )
    ranked = ratings_df.withColumn("rank", F.row_number().over(w))
    total  = ratings_df.groupBy("user_id").count().withColumnRenamed("count", "total")
    ranked = ranked.join(total, "user_id")
    ranked = ranked.withColumn("pct", F.col("rank") / F.col("total"))

    train = ranked.filter(F.col("pct") <= train_ratio).drop("rank", "total", "pct")
    test  = ranked.filter(F.col("pct") >  train_ratio).drop("rank", "total", "pct")
    return train, test


def run_preprocessing(spark: SparkSession = None):
    """End-to-end preprocessing; returns (ratings, movies, train, test)."""
    spark = spark or get_spark()

    print("📂 Loading ratings...")
    ratings = load_ratings(spark)
    print(f"   {ratings.count():,} ratings loaded")

    print("🎬 Loading movies...")
    movies = load_movies(spark)
    print(f"   {movies.count():,} movies loaded")

    print("📊 Normalizing ratings...")
    ratings_norm = normalize_ratings(ratings)

    print("✂️  Splitting train / test (80 / 20)...")
    train, test = train_test_split(ratings_norm)
    print(f"   Train: {train.count():,}  |  Test: {test.count():,}")

    # Basic stats
    print("\n── Dataset Summary ──────────────────────────────")
    ratings.describe("rating").show()
    print(f"Unique users : {ratings.select('user_id').distinct().count():,}")
    print(f"Unique movies: {ratings.select('movie_id').distinct().count():,}")
    print("─────────────────────────────────────────────────\n")

    return ratings, movies, train, test


if __name__ == "__main__":
    spark = get_spark()
    run_preprocessing(spark)
    spark.stop()
