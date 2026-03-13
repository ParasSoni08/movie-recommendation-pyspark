#!/usr/bin/env bash
# Downloads MovieLens 100K dataset into the data/ directory

set -e

DATA_DIR="$(dirname "$0")"
URL="https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP="$DATA_DIR/ml-100k.zip"

echo "📥 Downloading MovieLens 100K..."
curl -L "$URL" -o "$ZIP"

echo "📦 Extracting..."
unzip -q "$ZIP" -d "$DATA_DIR"
rm "$ZIP"

echo "✅ Dataset ready at $DATA_DIR/ml-100k/"
echo "   Key files:"
echo "   - ml-100k/u.data     (100K ratings: user_id, movie_id, rating, timestamp)"
echo "   - ml-100k/u.item     (movie metadata: id, title, genres...)"
echo "   - ml-100k/u.user     (user demographics)"
