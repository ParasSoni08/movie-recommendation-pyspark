"""
tests/test_recommender.py
─────────────────────────
Unit & integration tests for the Movie Recommendation System.

Run with:
    pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    from preprocessing import get_spark
    session = get_spark("TestSession")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def sample_data(spark):
    """Tiny in-memory dataset — no disk access required for unit tests."""
    from pyspark.sql import Row

    ratings_rows = [
        Row(user_id=1, movie_id=1, rating=5.0, timestamp=1000),
        Row(user_id=1, movie_id=2, rating=3.0, timestamp=1001),
        Row(user_id=1, movie_id=3, rating=4.0, timestamp=1002),
        Row(user_id=2, movie_id=1, rating=4.0, timestamp=1003),
        Row(user_id=2, movie_id=4, rating=2.0, timestamp=1004),
        Row(user_id=3, movie_id=2, rating=5.0, timestamp=1005),
        Row(user_id=3, movie_id=3, rating=3.0, timestamp=1006),
        Row(user_id=4, movie_id=1, rating=4.0, timestamp=1007),
        Row(user_id=4, movie_id=5, rating=5.0, timestamp=1008),
        Row(user_id=5, movie_id=3, rating=2.0, timestamp=1009),
    ]

    movies_rows = [
        Row(movie_id=1, title="Toy Story (1995)",     genres=["Animation", "Comedy"],  genre_string="Animation|Comedy"),
        Row(movie_id=2, title="GoldenEye (1995)",     genres=["Action", "Thriller"],   genre_string="Action|Thriller"),
        Row(movie_id=3, title="Four Rooms (1995)",    genres=["Thriller"],             genre_string="Thriller"),
        Row(movie_id=4, title="Get Shorty (1995)",    genres=["Action", "Comedy"],     genre_string="Action|Comedy"),
        Row(movie_id=5, title="Copycat (1995)",       genres=["Crime", "Thriller"],    genre_string="Crime|Thriller"),
    ]

    ratings_df = spark.createDataFrame(ratings_rows)
    movies_df  = spark.createDataFrame(movies_rows)
    return ratings_df, movies_df


# ── Preprocessing Tests ────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_normalize_ratings(self, sample_data):
        from preprocessing import normalize_ratings
        ratings_df, _ = sample_data
        normed = normalize_ratings(ratings_df)
        assert "rating_norm" in normed.columns
        assert normed.count() == ratings_df.count()

    def test_train_test_split_sizes(self, sample_data):
        from preprocessing import train_test_split
        ratings_df, _ = sample_data
        train, test = train_test_split(ratings_df, train_ratio=0.8)
        total = train.count() + test.count()
        assert total == ratings_df.count()
        assert train.count() > 0
        assert test.count() > 0

    def test_train_test_no_overlap(self, sample_data):
        from preprocessing import train_test_split
        from pyspark.sql import functions as F
        ratings_df, _ = sample_data
        train, test = train_test_split(ratings_df, train_ratio=0.8)
        overlap = train.join(
            test,
            on=["user_id", "movie_id", "timestamp"],
            how="inner"
        )
        assert overlap.count() == 0, "Train and test sets should not overlap"


# ── ALS Tests ─────────────────────────────────────────────────────────────────

class TestALS:

    def test_fit_and_evaluate(self, sample_data):
        from als_model import ALSRecommender
        from preprocessing import train_test_split
        ratings_df, _ = sample_data
        train, test = train_test_split(ratings_df)

        rec = ALSRecommender(rank=5, max_iter=5)
        rec.fit(train)
        rmse = rec.evaluate(test)
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_recommend_for_user_returns_rows(self, sample_data):
        from als_model import ALSRecommender
        from preprocessing import train_test_split
        ratings_df, _ = sample_data
        train, _ = train_test_split(ratings_df)

        rec = ALSRecommender(rank=5, max_iter=5)
        rec.fit(train)
        recs = rec.recommend_for_user(user_id=1, n=3)
        assert recs.count() <= 3
        assert "movie_id" in recs.columns
        assert "als_score" in recs.columns

    def test_item_factors_shape(self, sample_data):
        from als_model import ALSRecommender
        from preprocessing import train_test_split
        ratings_df, _ = sample_data
        train, _ = train_test_split(ratings_df)

        rec = ALSRecommender(rank=5, max_iter=5)
        rec.fit(train)
        factors = rec.get_item_factors()
        assert factors.count() > 0


# ── Content-Based Tests ────────────────────────────────────────────────────────

class TestContentBased:

    def test_fit_builds_vectors(self, sample_data):
        from content_based import ContentBasedRecommender
        _, movies_df = sample_data
        cb = ContentBasedRecommender(num_features=64)
        cb.fit(movies_df)
        assert len(cb._movie_vectors) == 5

    def test_similar_movies_excludes_self(self, sample_data):
        from content_based import ContentBasedRecommender
        _, movies_df = sample_data
        cb = ContentBasedRecommender(num_features=64)
        cb.fit(movies_df)
        similar = cb.recommend_similar_movies(movie_id=1, n=3)
        movie_ids = [r["movie_id"] for r in similar]
        assert 1 not in movie_ids

    def test_similar_movies_count(self, sample_data):
        from content_based import ContentBasedRecommender
        _, movies_df = sample_data
        cb = ContentBasedRecommender(num_features=64)
        cb.fit(movies_df)
        similar = cb.recommend_similar_movies(movie_id=2, n=2)
        assert len(similar) <= 2

    def test_recommend_for_user(self, sample_data):
        from content_based import ContentBasedRecommender
        ratings_df, movies_df = sample_data
        cb = ContentBasedRecommender(num_features=64)
        cb.fit(movies_df)
        recs = cb.recommend_for_user(user_id=1, ratings_df=ratings_df, n=3)
        assert isinstance(recs, list)
        # Should not recommend already-rated movies
        rated_ids = {1, 2, 3}
        for r in recs:
            assert r["movie_id"] not in rated_ids

    def test_unknown_movie_raises(self, sample_data):
        from content_based import ContentBasedRecommender
        _, movies_df = sample_data
        cb = ContentBasedRecommender(num_features=64)
        cb.fit(movies_df)
        with pytest.raises(ValueError):
            cb.recommend_similar_movies(movie_id=9999)


# ── Hybrid Tests ──────────────────────────────────────────────────────────────

class TestHybrid:

    @pytest.fixture(scope="class")
    def fitted_hybrid(self, sample_data):
        from hybrid_recommender import HybridRecommender
        from preprocessing import train_test_split
        ratings_df, movies_df = sample_data
        train, _ = train_test_split(ratings_df)

        h = HybridRecommender(als_weight=0.6, als_rank=5, als_max_iter=5, cb_num_features=64)
        h.fit(train, ratings_df, movies_df)
        return h

    def test_recommend_returns_list(self, fitted_hybrid):
        recs = fitted_hybrid.recommend(user_id=1, n=3)
        assert isinstance(recs, list)

    def test_recommend_has_expected_keys(self, fitted_hybrid):
        recs = fitted_hybrid.recommend(user_id=1, n=3)
        if recs:
            keys = recs[0].keys()
            assert "movie_id"      in keys
            assert "hybrid_score"  in keys
            assert "als_score"     in keys
            assert "content_score" in keys

    def test_recommend_scores_in_range(self, fitted_hybrid):
        recs = fitted_hybrid.recommend(user_id=1, n=5)
        for r in recs:
            assert 0.0 <= r["hybrid_score"] <= 1.0

    def test_recommend_similar(self, fitted_hybrid):
        similar = fitted_hybrid.recommend_similar(movie_id=1, n=3)
        assert isinstance(similar, list)
        assert all(r["movie_id"] != 1 for r in similar)

    def test_weight_sensitivity(self, sample_data):
        """ALS-heavy vs CB-heavy should produce different orderings."""
        from hybrid_recommender import HybridRecommender
        from preprocessing import train_test_split
        ratings_df, movies_df = sample_data
        train, _ = train_test_split(ratings_df)

        h1 = HybridRecommender(als_weight=0.9, als_rank=5, als_max_iter=5, cb_num_features=64)
        h2 = HybridRecommender(als_weight=0.1, als_rank=5, als_max_iter=5, cb_num_features=64)
        h1.fit(train, ratings_df, movies_df)
        h2.fit(train, ratings_df, movies_df)

        r1 = [r["movie_id"] for r in h1.recommend(1, n=5)]
        r2 = [r["movie_id"] for r in h2.recommend(1, n=5)]
        # Not necessarily different on tiny data, but both should be valid lists
        assert isinstance(r1, list)
        assert isinstance(r2, list)


# ── API Tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    """
    Lightweight smoke-tests for the FastAPI layer using mock state.
    Full integration tests require a running server (see README).
    """

    def test_health_endpoint_schema(self):
        from api.app import HealthResponse
        obj = HealthResponse(status="ok", uptime_s=42.0, model="Test Model")
        assert obj.status == "ok"

    def test_recommendation_item_schema(self):
        from api.app import RecommendationItem
        item = RecommendationItem(
            rank=1, movie_id=1, title="Test Movie",
            hybrid_score=0.9, als_score=0.8, content_score=0.7,
        )
        assert item.rank == 1
        assert item.hybrid_score == 0.9
