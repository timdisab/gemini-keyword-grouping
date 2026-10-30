"""Turn each keyword into an embedding vector with Gemini, caching results on disk."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai
from google.genai import errors, types

MODEL = "gemini-embedding-2"
# gemini-embedding-2 doesn't take a task_type setting, so the task goes into the text itself.
TASK_PREFIX = "task: clustering | query: "
INPUT_CSV = Path("data/human_grouping_v1.csv")
CACHE_FILE = Path("cache/embeddings.parquet")
BATCH_SIZE = 20
MAX_RETRIES = 6


def load_keywords():
    # Read only the two columns we need. The human grouping columns stay unread.
    return pd.read_csv(INPUT_CSV, usecols=["planner_rank", "keyword"])


def cache_key(keyword):
    # Key the cache on model + prefix + keyword, so changing either one forces fresh embeddings.
    return f"{MODEL}|{TASK_PREFIX}|{keyword}"


def load_cache():
    if CACHE_FILE.exists():
        df = pd.read_parquet(CACHE_FILE)
        return dict(zip(df["key"], df["vector"].map(np.asarray)))
    return {}


def save_cache(cache):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    df = pd.DataFrame({"key": list(cache.keys()), "vector": [v.tolist() for v in cache.values()]})
    df.to_parquet(CACHE_FILE, index=False)


def embed_batch(client, keywords):
    # A plain list of strings would give back ONE combined embedding.
    # Wrapping each keyword in its own Content object gives one embedding per keyword.
    contents = [
        types.Content(parts=[types.Part.from_text(text=TASK_PREFIX + kw)]) for kw in keywords
    ]

    # Retry with exponential backoff if we hit the rate limit (HTTP 429).
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(model=MODEL, contents=contents)
            break
        except errors.APIError as e:
            if e.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)

    # Fail loudly if we didn't get exactly one vector per keyword.
    if len(result.embeddings) != len(keywords):
        raise RuntimeError(
            f"Sent {len(keywords)} keywords but got {len(result.embeddings)} vectors back"
        )
    return [np.asarray(e.values, dtype=np.float32) for e in result.embeddings]


def get_embeddings():
    """Return the keyword table and a matrix with one embedding row per keyword."""
    df = load_keywords()
    cache = load_cache()

    # Only call the API for keywords we haven't embedded before.
    missing = [kw for kw in df["keyword"] if cache_key(kw) not in cache]
    missing = list(dict.fromkeys(missing))  # drop duplicates, keep order

    if missing:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i : i + BATCH_SIZE]
            for kw, vec in zip(batch, embed_batch(client, batch)):
                cache[cache_key(kw)] = vec
            # Save after every batch so a crash doesn't lose finished work.
            save_cache(cache)

    print(f"Embedded {len(missing)} keywords via API, loaded {len(df) - len(missing)} from cache")

    vectors = np.vstack([cache[cache_key(kw)] for kw in df["keyword"]])
    return df, vectors


if __name__ == "__main__":
    df, vectors = get_embeddings()
    print(f"Embedding matrix: {vectors.shape}")
