# Gemini Keyword Grouping

Group Google Ads keywords by meaning with Gemini embeddings and agglomerative clustering in Python.

This is the code from the video **Group Google Ads Keywords with AI: Gemini Embeddings + Python**. It groups 100 Keyword Planner keywords for an electrician, so the results can be compared with a grouping done by hand.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
python run.py
```

No API key is needed for this. The embeddings for all 100 keywords are included in `cache/embeddings.parquet`, so the script makes no API calls and costs nothing.

It prints a threshold sweep, then writes one CSV to `output/` for each threshold it picks. To choose your own thresholds:

```bash
python run.py 0.04 0.06 0.09 0.11
```

A lower threshold gives more, smaller groups. A higher threshold gives fewer, bigger ones.

## Use your own keywords

1. Replace `data/human_grouping_v1.csv` with your own file. It needs `planner_rank` and `keyword` columns.
2. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/).
3. Copy the example env file and paste your key into `.env`:

   ```bash
   cp .env.example .env
   ```

4. Run `python run.py`. Only keywords that aren't already in the cache are sent to the API.

`.env` is listed in `.gitignore`. Never commit it.

Good thresholds depend on your keywords, so check the sweep before you trust the groups. `SWEEP` in `cluster.py` sets the range it tries.

## Files

| Path | What it is |
| --- | --- |
| `run.py` | Runs the embedding step, then the clustering step |
| `embed.py` | Gets embeddings from Gemini and caches them |
| `cluster.py` | Sweeps thresholds, clusters the keywords, and saves the results |
| `data/human_grouping_v1.csv` | The 100 keywords with the groups assigned by hand |
| `cache/embeddings.parquet` | Saved embeddings for those keywords |
| `output/clusters_t*.csv` | Cluster results at thresholds 0.04, 0.06, 0.09, and 0.11 |

## License

MIT. See [LICENSE](LICENSE).
