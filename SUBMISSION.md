# Submission — Hebrew RAG Retrieval Improvement

**Built:** hybrid BM25 + dense retrieval via weighted Reciprocal Rank Fusion (2:1).
**Result:** Recall@3 (the production metric) 0.833 → 0.933, with a documented
rank-1/MRR trade-off — details below.

## The improvement: Hybrid BM25 + Dense Retrieval

**Before:** `Engine.search_documents()` retrieved purely via cosine similarity against a
fine-tuned `multilingual-e5-large` bi-encoder — a single Elasticsearch `script_score`
query, with no lexical/keyword signal anywhere in the pipeline.

**Why this direction:** rather than pick a technique off a generic list, I inspected the
baseline's actual failure cases. A clear pattern emerged: queries containing a **rare,
specific, named term** — a medical procedure ("total laryngectomy"), a legal category
("workers from the territories"), a single distinctive noun ("widow") — got buried at
rank 6-8 behind pages that were only generically/topically similar. That's the textbook
failure mode of dense-only retrieval: a specific term gets diluted by broad semantic
similarity to unrelated pages. A standalone sanity check confirmed it before writing any
code — a bare BM25 query for "widow" (אלמנה) returned the correct page as its #1 hit.

**What was implemented:** the retrieval engine (`Webiks-Hebrew-RAGbot`, forked as a
sibling repo) now fuses the existing dense ranking with a new BM25 `match` query on the
same indexed `content` field, via weighted **Reciprocal Rank Fusion** (dense:BM25 =
2:1). No reindexing, no new model, no new dependency — BM25 is native to Elasticsearch.
Toggled by a `RETRIEVAL_MODE` env var (`dense` = original behavior, byte-for-byte
unchanged — confirmed by both an automated eval-harness regression run and unit tests;
`hybrid` = the improvement). The Demo's `.env` sets `RETRIEVAL_MODE=hybrid`.

I initially tried **unweighted** RRF: it fixed the target failures but reshuffled many
already-correct rank-1 results and caused one new regression (a long query where BM25's
OR-across-terms matching hit 436/450 paragraphs, diluting its own signal). Weighting
dense 2:1 over BM25 kept the recall gain while recovering most of the lost precision.
Full iteration detail, including a per-paragraph RRF scoring bug caught and fixed during
review, in `EVALUATION_METHODOLOGY.md`.

## Evaluation

**Metrics.** Each question has exactly one gold document, so:
- **Recall@k** (k=1,3,5,10) — did the gold page appear in the top-k, distinct-by-page
  results? **Recall@3 is the headline metric**, since `num_of_pages=3` is the app's
  actual production default fed to the LLM — it directly answers "does the LLM see the
  right page?"
- **MRR@10** — average of 1/rank of the first hit; distinguishes rank-1 from rank-9,
  which Recall@10 alone can't.
- **NDCG@k** — a softer, log-discounted complement to MRR's hard rank cutoff.
- **Not included: MAP.** With one relevant document per query, Average Precision equals
  `1/rank` exactly — identical to MRR here, so computing both is pure duplication.

**Eval set — why not just sample the QA CSV:**
- **No leakage.** That file *is* the retrieval model's training data (per the task
  brief), so I reconstructed the exact held-out validation split the training code
  (`Webiks-Hebrew-RAGbot-Trainer/utils.py: split_train_eval`) would have produced — a
  90/10 split on unique questions with a fixed seed (`random_state=42`) — and sampled 30
  questions from *only* that reconstructed held-out pool.
- **Real distractors.** The search index (450 paragraphs: the 30 questions' 161 real
  answer-paragraphs + ~289 random unrelated distractors) ensures genuine wrong answers
  are retrievable — avoiding an earlier dead end where a distractor-free index scored a
  meaningless perfect 1.000 on everything.

Full methodology, including both dead ends, in `EVALUATION_METHODOLOGY.md`.

**Results — Baseline (dense) vs. Improved (hybrid, weighted RRF), n=30 questions:**

| Metric | Baseline | Improved | Δ |
|---|---|---|---|
| Recall@1 | 0.700 | 0.567 | −0.133 |
| **Recall@3** | **0.833** | **0.933** | **+0.100** |
| Recall@5 | 0.833 | 0.933 | +0.100 |
| Recall@10 | 1.000 | 0.967 | −0.033 |
| NDCG@3 | 0.775 | 0.772 | −0.003 |
| MRR@10 | 0.780 | 0.721 | −0.059 |
| Mean latency | 1.24s | 1.19s | −0.05s |
| Not found (top 10) | 0/30 | 1/30 | +1 |

Recall@3 — what actually determines whether the LLM sees the right page under
production settings — improved from 0.833 to 0.933 (3 of 5 previously-missing
questions now surface correctly). This came with an honest cost: some rank-1 hits
got nudged lower by the fusion (lowering Recall@1/MRR), and one specific long,
low-specificity query (the "workers from the territories" query) still regresses out of
the top 10 — a known, documented limitation of plain-text BM25 without a Hebrew-aware
analyzer (noted as future work).
With n=30, each question is worth ~3.3 points, so these deltas are directional, not
statistically validated. Full per-question detail:
`EVALUATION_METHODOLOGY.md`, `results/baseline_metrics.json`, `results/improved_metrics.json`.

## Running the updated backend locally

Requires: Docker, Python 3.10 (exactly — not 3.11+), ~4GB free RAM headroom.

```bash
# 1. Clone the Demo repo. Its requirements.txt pulls the forked engine directly
#    from GitHub, pinned to a commit, so this single clone is enough to run it
#    (clone https://github.com/liorKreimer/Webiks-Hebrew-RAGbot.git separately
#    only if you want to review or modify the engine's own code):
git clone https://github.com/liorKreimer/Webiks-Hebrew-RAGbot-Demo.git

# 2. Elasticsearch
docker run -d --name es-rag \
  -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" -p 9200:9200 \
  -v es-rag-data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.2

# 3. Python environment (Demo repo)
cd Webiks-Hebrew-RAGbot-Demo
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # installs the forked engine automatically
```

**4. Model + config** (manual steps, not shell commands):
- Download the retrieval model and place it in `app/artifacts/` (see the original
  README for the Google Drive link).
- Create `app/.env` from `app/.env-example`; set `IS_MOCK_GPT_CLIENT=TRUE` (no OpenAI
  key needed) and `RETRIEVAL_MODE=hybrid` (or `dense` for the original, unmodified
  behavior).
- Also set `ES_EMBEDDING_INDEX=embedded_index` — it ships blank in `.env-example`, and
  blank isn't the same as unset; an empty value breaks the index pattern.
- Set `PATH_TO_ES_INITIAL_VALUES` to a paragraph-corpus JSON to index. For a fast
  reproduction of the numbers above, use the committed
  `downloads/paragraphs_corpus_holdout.json` (450 paragraphs) — the full corpus is much
  larger and can take hours on a memory-constrained machine.

```bash
# 5. Run (note: cwd must be app/src — see IMPORTANT note below)
cd app/src
../../.venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 5000
```

**Step 6 blocks until indexing finishes** — on the 450-paragraph holdout corpus above,
expect roughly 30-40 minutes on a memory-constrained (~8GB) machine, not a hang. Don't
run it concurrently with anything else memory-heavy.

```bash
# 6. Seed Elasticsearch (one-time, or whenever the corpus changes)
curl http://localhost:5000/initialize_elastic_from_json

# 7. Verify
curl http://localhost:5000/health
curl -X POST http://localhost:5000/search -H "Content-Type: application/json" \
     -d '{"query": "<Hebrew question>", "asked_from": "https://www.kolzchut.org.il/"}'
```

**IMPORTANT:** the original README's documented run command
(`uvicorn app.src.main:app` from the repo root) does not work — `main.py` uses flat
imports (`import config`, not `from app.src import config`) that only resolve when
`app/src/` itself is the working directory. Run `uvicorn main:app` from inside
`app/src/` as shown above. All `.env` relative paths are written relative to `app/src`
accordingly (this matches the original `.env-example` as shipped).

To compare against the original unmodified retrieval, set `RETRIEVAL_MODE=dense` and
restart the app — every other code path is untouched.

## Presentation

[`presentation.html`](./presentation.html) — a 3-slide deck (open it in a browser; arrow
keys or click to navigate) covering the diagnosis, the fix and workflow, and the results.
