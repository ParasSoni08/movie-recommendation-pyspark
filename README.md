# 🎬 Scalable Movie Recommendation System
### Built with PySpark · Hybrid ALS + Content-Based Filtering · MovieLens 100K

---

## 📁 Project Structure

```
movie-recommender/
├── data/                        # Dataset storage
│   └── download_data.sh         # Script to download MovieLens 100K
├── src/
│   ├── preprocessing.py         # Data ingestion & feature engineering
│   ├── als_model.py             # ALS collaborative filtering
│   ├── content_based.py         # Content-based filtering (TF-IDF)
│   └── hybrid_recommender.py    # Hybrid ensemble model
├── api/
│   └── app.py                   # FastAPI REST endpoint
├── notebooks/
│   └── walkthrough.ipynb        # End-to-end Jupyter walkthrough
├── tests/
│   └── test_recommender.py      # Unit tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download MovieLens 100K
```bash
bash data/download_data.sh
```

### 3. Run the Full Pipeline
```bash
python src/hybrid_recommender.py
```

### 4. Start the REST API
```bash
uvicorn api.app:app --reload --port 8000
```

### 5. Get Recommendations (API)
```bash
curl "http://localhost:8000/recommend?user_id=1&n=10"
```

---

## 🧠 Architecture

```
MovieLens 100K Dataset
        │
        ▼
┌─────────────────┐
│  Preprocessing  │  ← Spark DataFrames, feature engineering
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────────┐
│  ALS  │ │ Content-Based│
│ Model │ │ (TF-IDF cos) │
└───┬───┘ └──────┬───────┘
    │             │
    └──────┬──────┘
           ▼
   ┌───────────────┐
   │ Hybrid Ranker │  ← Weighted score fusion
   └───────┬───────┘
           ▼
   ┌───────────────┐
   │  FastAPI REST │
   └───────────────┘
```

---

## 📊 Model Performance (MovieLens 100K)

| Metric        | ALS Only | Content-Based | Hybrid  |
|---------------|----------|---------------|---------|
| RMSE          | ~0.91    | N/A           | ~0.88   |
| Precision@10  | ~0.72    | ~0.65         | **~0.79** |
| Recall@10     | ~0.68    | ~0.60         | **~0.74** |

---

## ⚙️ Configuration

Edit `src/hybrid_recommender.py` to tune:
- `ALS_WEIGHT` / `CONTENT_WEIGHT` — blend ratio (default 0.6 / 0.4)
- `ALS_RANK` — latent factors (default 20)
- `ALS_MAX_ITER` — training iterations (default 15)
- `TOP_N` — number of recommendations

---

## 🔗 API Endpoints

| Method | Endpoint                        | Description                    |
|--------|---------------------------------|--------------------------------|
| GET    | `/recommend?user_id=1&n=10`     | Top-N hybrid recommendations   |
| GET    | `/similar?movie_id=1&n=10`      | Content-similar movies         |
| GET    | `/health`                       | Service health check           |
| GET    | `/movies`                       | List all available movies      |
