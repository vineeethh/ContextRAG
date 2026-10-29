"""Live evaluation: run the golden docs questions through the real QueryPipeline.

Ingests data/raw/{fastapi,supabase} into an isolated eval tenant, asks every
question in eval/golden_dataset/docs/eval_v1.jsonl, then re-asks them to
measure semantic-cache hits. Collected per question: answer, retrieved
contexts, latency (end to end and per stage), tokens, cost and top rerank
score. Answers are scored with RAGAS (faithfulness, answer relevancy, context
precision, context recall) using Haiku 5.5 via OpenRouter as the judge.

Everything is written to eval/results/live_eval_<timestamp>.{json,csv}.

Needs Postgres, Redis Stack and Qdrant running (docker compose up -d postgres
redis qdrant) and OPENROUTER_API_KEY / COHERE_API_KEY in .env.

Usage:
  python eval/run_live_eval.py                 # full run, 78 questions
  python eval/run_live_eval.py --limit 3       # smoke test
  python eval/run_live_eval.py --skip-ragas    # latency / cost only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import statistics
import sys
import time
import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg  # noqa: E402
import redis.asyncio as aioredis  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.connectors.chunkers.heading_aware_chunker import HeadingAwareChunker  # noqa: E402
from backend.core.query_pipeline import QueryPipeline  # noqa: E402
from backend.models import ContextWindow, QueryResult, SourceType  # noqa: E402
from backend.observers.base import QueryObserver  # noqa: E402
from backend.repositories.postgres_conversation_repo import (  # noqa: E402
    PostgresConversationRepository,
)
from backend.repositories.postgres_source_hash_repo import PostgresSourceHashRepository  # noqa: E402
from backend.strategies.cache.redis_semantic_cache import RedisSemanticCache  # noqa: E402
from backend.strategies.embedding.openai_embedding import OpenAIEmbedding  # noqa: E402
from backend.strategies.embedding.tf_sparse_encoder import TFSparseEncoder  # noqa: E402
from backend.strategies.llm.openai_llm import OpenAILLM  # noqa: E402
from backend.strategies.llm.openrouter_llm import OpenRouterLLM  # noqa: E402
from backend.strategies.reranker.cohere_reranker import CohereReranker  # noqa: E402
from backend.strategies.vectordb.qdrant_db import QdrantDB, _collection  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
GOLDEN = ROOT / "eval" / "golden_dataset" / "docs" / "eval_v1.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

EVAL_TENANT_ID = "00000000-0000-0000-0000-000000000001"
EVAL_API_HASH = hashlib.sha256(b"ragas-eval-key").hexdigest()

# A single query this slow means the machine slept or the network hung, not that the
# pipeline is slow. Such rows are kept in the raw data but left out of latency stats.
SLOW_QUERY_S = 60

JUDGE_MODEL = "anthropic/claude-haiku-5.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


class _Capture(QueryObserver):
    """Keeps the last ContextWindow so the run can read stage latencies and cost."""

    def __init__(self) -> None:
        self.contexts: list[ContextWindow] = []

    async def notify(self, context: ContextWindow, result: QueryResult) -> None:
        self.contexts.append(context)


# ── setup ─────────────────────────────────────────────────────────────────────

async def _reset_state(pool: asyncpg.Pool, db: QdrantDB) -> None:
    """Fresh tenant, empty collection and empty semantic cache for a clean run."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES ($1, $2, $3) "
            "ON CONFLICT DO NOTHING",
            EVAL_TENANT_ID, "RAGAS Eval Tenant", EVAL_API_HASH,
        )
    if await db.collection_exists(EVAL_TENANT_ID):
        await db._client.delete_collection(_collection(EVAL_TENANT_ID))
    await db.create_collection(EVAL_TENANT_ID)

    r = aioredis.from_url(settings.redis_url)
    async for key in r.scan_iter(match="semcache:*"):
        await r.delete(key)
    await r.aclose()


async def _ingest(db: QdrantDB, hash_repo: PostgresSourceHashRepository) -> int:
    chunker = HeadingAwareChunker()
    embedder = OpenAIEmbedding()
    sparse = TFSparseEncoder()
    files = sorted(list(DATA_DIR.rglob("*.md")) + list(DATA_DIR.rglob("*.mdx")))
    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        source_url = str(path.relative_to(DATA_DIR))
        chunks = chunker.chunk(text, {
            "tenant_id": EVAL_TENANT_ID,
            "source_url": source_url,
            "source_type": SourceType.DOCS_SITE.value,
        })
        texts = [c.content for c in chunks]
        dense = await embedder.embed(texts)
        sparse_pairs = await sparse.encode(texts)
        for chunk, vec, (idx, vals) in zip(chunks, dense, sparse_pairs):
            chunk.tenant_id = EVAL_TENANT_ID
            chunk.source_url = source_url
            chunk.dense_vector = vec
            chunk.sparse_indices = idx
            chunk.sparse_values = vals
        await db.upsert(chunks)
        await hash_repo.upsert(
            EVAL_TENANT_ID, source_url,
            hashlib.sha256(text.encode()).hexdigest(), SourceType.DOCS_SITE.value,
        )
        total += len(chunks)
        print(f"  ingested {source_url:40s} {len(chunks):3d} chunks")
    return total


# ── query passes ──────────────────────────────────────────────────────────────

async def _ask(pipeline: QueryPipeline, capture: _Capture, question: str) -> dict:
    before = len(capture.contexts)
    t0 = time.perf_counter()
    try:
        result = await pipeline.handle(question, EVAL_TENANT_ID, uuid4())
    except Exception as exc:  # keep going; a failed question is reported, not hidden
        return {"question": question, "error": f"{type(exc).__name__}: {exc}"}
    latency = time.perf_counter() - t0

    # Observers run as fire-and-forget tasks; wait briefly for this query's context.
    for _ in range(100):
        if len(capture.contexts) > before:
            break
        await asyncio.sleep(0.02)
    ctx = capture.contexts[-1] if len(capture.contexts) > before else ContextWindow()

    return {
        "question": question,
        "answer": result.answer,
        "contexts": [c.content for c in result.source_chunks],
        "cached": result.cached,
        "latency_s": latency,
        "stages_s": dict(ctx.pipeline_stage_latencies),
        "tokens_in": ctx.tokens_in,
        "tokens_out": ctx.tokens_out,
        "cost_usd": float(ctx.cost_usd),
        "top_retrieval_score": result.top_retrieval_score,
    }


async def _run_pass(
    pipeline: QueryPipeline, capture: _Capture, questions: list[str], pause: float, label: str
) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        row = await _ask(pipeline, capture, q)
        if row.get("latency_s", 0) > SLOW_QUERY_S:
            row["suspect_slow"] = True
        rows.append(row)
        status = row.get("error") or f"{row['latency_s'] * 1000:8.0f} ms  cached={row['cached']}"
        print(f"  [{label} {i:2d}/{len(questions)}] {status}")
        if row.get("suspect_slow"):
            print(f"  WARNING: query took over {SLOW_QUERY_S}s (machine asleep?); "
                  "excluded from latency stats")
        if pause:
            await asyncio.sleep(pause)  # Cohere trial-key rate limit; not counted in latency
    return rows


# ── scoring ───────────────────────────────────────────────────────────────────

def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _latency_summary(rows: list[dict]) -> dict:
    lat = [r["latency_s"] * 1000 for r in rows]
    return {
        "n": len(lat),
        "p50_ms": round(_pct(lat, 50), 1),
        "p95_ms": round(_pct(lat, 95), 1),
        "p99_ms": round(_pct(lat, 99), 1),
        "mean_ms": round(statistics.fmean(lat), 1) if lat else None,
    }


def _stage_summary(rows: list[dict]) -> dict:
    out = {}
    for stage in ("embed", "retrieve", "rerank", "generate"):
        vals = [r["stages_s"][stage] * 1000 for r in rows if stage in r["stages_s"]]
        if vals:
            out[stage] = {
                "p50_ms": round(_pct(vals, 50), 1),
                "p95_ms": round(_pct(vals, 95), 1),
            }
    return out


def _ragas_scores(records: list[dict]) -> tuple[dict, list[dict]]:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    warnings.filterwarnings("ignore")
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=OPENROUTER_URL,
        api_key=settings.openrouter_api_key,
        temperature=0,
        max_tokens=4096,
        extra_body={"reasoning": {"enabled": False}},
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        base_url=OPENROUTER_URL,
        api_key=settings.openrouter_api_key,
        check_embedding_ctx_length=False,
    ))
    result = evaluate(
        dataset=Dataset.from_list(records),
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=8, timeout=180),
        raise_exceptions=False,
    )
    df = result.to_pandas()
    metric_cols = [c for c in ("faithfulness", "answer_relevancy", "context_precision",
                               "context_recall") if c in df.columns]
    agg = {c: round(float(df[c].mean()), 3) for c in metric_cols}
    agg["n_scored"] = {c: int(df[c].notna().sum()) for c in metric_cols}
    return agg, df.to_dict(orient="records")


# ── main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if args.limit:
        golden = golden[: args.limit]
    questions = [g["question"] for g in golden]
    print(f"Questions: {len(questions)} | LLM: {settings.llm_provider}/{settings.openrouter_model}"
          f" | embeddings: {settings.openai_embedding_model}")

    pool = await asyncpg.create_pool(settings.postgres_url.replace("+asyncpg", ""))
    db = QdrantDB()
    hash_repo = PostgresSourceHashRepository(pool)
    capture = _Capture()
    llm = OpenRouterLLM() if settings.llm_provider == "openrouter" else OpenAILLM()
    pipeline = QueryPipeline(
        llm=llm,
        embedder=OpenAIEmbedding(),
        sparse_encoder=TFSparseEncoder(),
        vector_db=db,
        reranker=CohereReranker(),
        cache=RedisSemanticCache(),
        conversation_repo=PostgresConversationRepository(pool),
        observers=[capture],
        source_hash_repo=hash_repo,
    )

    try:
        if args.warm_only:
            # Reuse the previous run's collection and cache; only re-measure cache hits.
            n_chunks, cold = None, []
        else:
            print("\n== Reset + ingest ==")
            await _reset_state(pool, db)
            n_chunks = await _ingest(db, hash_repo)

            print("\n== Pass 1: cold (empty cache) ==")
            cold = await _run_pass(pipeline, capture, questions, args.pause, "cold")
            await asyncio.sleep(2)  # let fire-and-forget cache writes land

        print("\n== Pass 2: warm (same questions, cache populated) ==")
        warm = await _run_pass(pipeline, capture, questions, 0, "warm")
    finally:
        await pool.close()

    ok_cold = [r for r in cold if "error" not in r]
    ok_warm = [r for r in warm if "error" not in r]
    cold_miss = [r for r in ok_cold if not r["cached"] and not r.get("suspect_slow")]
    cold_hit = [r for r in ok_cold if r["cached"]]
    warm_hit = [r for r in ok_warm if r["cached"] and not r.get("suspect_slow")]

    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "llm": f"{settings.llm_provider}:{settings.openrouter_model}",
        "reasoning": settings.openrouter_reasoning,
        "embedding_model": settings.openai_embedding_model,
        "judge_model": JUDGE_MODEL,
        "questions": len(questions),
        "chunks_ingested": n_chunks,
        "excluded_slow_queries": sum(
            1 for r in cold + warm if r.get("suspect_slow")),
        "errors": {"cold": len(cold) - len(ok_cold), "warm": len(warm) - len(ok_warm)},
        "empty_answers": sum(1 for r in ok_cold if not r["answer"].strip()),
        "cold_pass": {
            "full_pipeline_queries": len(cold_miss),
            "served_from_cache_in_cold_pass": len(cold_hit),
            "latency_full_pipeline": _latency_summary(cold_miss),
            "stage_latency": _stage_summary(cold_miss),
        },
        "warm_pass": {
            "cache_hits": len(warm_hit),
            "cache_hit_rate": round(len(warm_hit) / len(ok_warm), 3) if ok_warm else None,
            "latency_cache_hit": _latency_summary(warm_hit),
        },
        "tokens_and_cost": {
            "tokens_in_total": sum(r["tokens_in"] for r in cold_miss),
            "tokens_out_total": sum(r["tokens_out"] for r in cold_miss),
            "tokens_in_per_query": round(statistics.fmean(r["tokens_in"] for r in cold_miss), 1)
            if cold_miss else None,
            "tokens_out_per_query": round(statistics.fmean(r["tokens_out"] for r in cold_miss), 1)
            if cold_miss else None,
            "cost_usd_total": float(sum(Decimal(str(r["cost_usd"])) for r in cold_miss)),
            "cost_usd_per_1000_queries": round(
                1000 * statistics.fmean(r["cost_usd"] for r in cold_miss), 4
            ) if cold_miss else None,
        },
        "retrieval": {
            "top_rerank_score_mean": round(statistics.fmean(
                r["top_retrieval_score"] for r in cold_miss if r["top_retrieval_score"] is not None
            ), 3) if cold_miss else None,
            "below_gap_threshold": sum(
                1 for r in cold_miss
                if r["top_retrieval_score"] is not None
                and r["top_retrieval_score"] < settings.coverage_gap_threshold
            ),
            "gap_threshold": settings.coverage_gap_threshold,
        },
    }
    if cold_miss and cold_miss[0]["latency_s"] and warm_hit:
        summary["cache_speedup_x"] = round(
            summary["cold_pass"]["latency_full_pipeline"]["p50_ms"]
            / summary["warm_pass"]["latency_cache_hit"]["p50_ms"], 1)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    raw_path = RESULTS_DIR / f"live_eval_{stamp}_raw.json"
    raw_path.write_text(json.dumps({"summary": summary, "cold": cold, "warm": warm},
                                   indent=2, default=str))
    print(f"\nRaw answers + latency saved (before scoring): {raw_path}")

    per_question = []
    if not args.skip_ragas and not args.warm_only:
        print(f"\n== RAGAS (judge: {JUDGE_MODEL}) ==")
        by_q = {g["question"]: g for g in golden}
        records = [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": by_q[r["question"]]["answer"],
            }
            for r in ok_cold
        ]
        agg, per_question = await asyncio.to_thread(_ragas_scores, records)
        summary["ragas"] = agg

    out_json = RESULTS_DIR / f"live_eval_{stamp}.json"
    out_json.write_text(json.dumps(
        {"summary": summary, "cold": cold, "warm": warm, "ragas_per_question": per_question},
        indent=2, default=str,
    ))
    out_csv = RESULTS_DIR / f"live_eval_{stamp}.csv"
    with out_csv.open("w", newline="") as f:  # header-only in --warm-only mode
        w = csv.writer(f)
        w.writerow(["question", "cached_cold", "cold_latency_ms", "warm_latency_ms",
                    "tokens_in", "tokens_out", "cost_usd", "top_score"])
        for c, h in zip(cold, warm):
            if "error" in c:
                w.writerow([c["question"], "ERROR", c["error"]])
                continue
            w.writerow([c["question"], c["cached"], round(c["latency_s"] * 1000, 1),
                        round(h.get("latency_s", 0) * 1000, 1), c["tokens_in"],
                        c["tokens_out"], c["cost_usd"], c["top_retrieval_score"]])

    print("\n== Summary ==")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_json}\n       {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--pause", type=float, default=6.0,
                        help="seconds between cold queries (Cohere trial rate limit)")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--warm-only", action="store_true",
                        help="skip ingest and the cold pass; re-ask against the existing cache")
    parser.add_argument("--tag", default="", help="suffix for the result file names")
    asyncio.run(main(parser.parse_args()))
