"""Embed the keywords, then cluster them.

Usage:
    python run.py                     # auto-pick thresholds nearest 8/15/24/40 clusters
    python run.py 0.04 0.06 0.09 0.11 # or choose your own thresholds from the sweep
"""

import sys

from dotenv import load_dotenv

# Load GEMINI_API_KEY from the .env file (kept out of git by .gitignore).
load_dotenv()

import cluster
import embed

thresholds = [float(t) for t in sys.argv[1:]] or None

df, vectors = embed.get_embeddings()
cluster.run(df, vectors, thresholds)
