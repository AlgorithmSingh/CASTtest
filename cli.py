from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import shutil

from cast_rag import (
    INDEX_ROOT,
    generate_answer,
    ingest_repository,
    query_index,
    query_repository,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repository RAG retrieval with cAST and RLM-inspired strategies")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("experiment", help="Run built-in synthetic experiment")
    exp.set_defaults(func=cmd_experiment)

    # --- ingest command ---
    ing = sub.add_parser("ingest", help="Clone/read a repo, chunk it, embed it, and save a FAISS index")
    ing.add_argument("source", help="Git URL (https/ssh) or path to a local repository")
    ing.add_argument("--name", default=None, help="Index name (default: derived from repo URL/path)")
    ing.add_argument("--strategy", choices=["cast", "fixed", "rlm"], default="cast", help="Chunking strategy")
    ing.set_defaults(func=cmd_ingest)

    # --- list command ---
    ls = sub.add_parser("list", help="List all saved indexes")
    ls.set_defaults(func=cmd_list)

    # --- delete command ---
    dl = sub.add_parser("delete", help="Delete a saved index")
    dl.add_argument("name", help="Name of the index to delete")
    dl.set_defaults(func=cmd_delete)

    # --- ask command ---
    ask = sub.add_parser("ask", help="Query a local repository or a saved index")
    group = ask.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="Path to local repository (ephemeral BM25-only search)")
    group.add_argument("--index", help="Name of a previously ingested index (hybrid vector+BM25 search)")
    ask.add_argument("--query", required=True, help="Natural language retrieval query")
    ask.add_argument("--strategy", choices=["cast", "fixed", "rlm"], default="cast", help="Chunking strategy (only for --repo)")
    ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    ask.add_argument("--json", action="store_true", help="Emit JSON output")
    ask.add_argument("--answer", action="store_true", help="Generate an answer using Gemini")
    ask.add_argument("--gemini-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env var)")
    ask.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use (default: gemini-2.5-flash)")
    ask.set_defaults(func=cmd_ask)
    return parser


def cmd_experiment(_args: argparse.Namespace) -> int:
    run_experiment()
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    if not INDEX_ROOT.exists():
        print("No indexes found.")
        return 0
    indexes = sorted(p for p in INDEX_ROOT.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if not indexes:
        print("No indexes found.")
        return 0
    for idx_dir in indexes:
        meta = json.loads((idx_dir / "meta.json").read_text())
        print(f"  {meta['name']:20s}  chunks={meta['num_chunks']:>5}  files={meta['num_files']:>4}  strategy={meta['strategy']}  source={meta['source']}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    idx_dir = INDEX_ROOT / args.name
    if not idx_dir.exists():
        print(f"error: no index named '{args.name}'", file=sys.stderr)
        return 1
    shutil.rmtree(idx_dir)
    print(f"Deleted index '{args.name}'.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        meta = ingest_repository(args.source, name=args.name, strategy=args.strategy)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Ingested '{meta['name']}' successfully.")
    print(f"  Source:  {meta['source']}")
    print(f"  Files:   {meta['num_files']}")
    print(f"  Chunks:  {meta['num_chunks']}")
    print(f"  Strategy: {meta['strategy']}")
    print(f"\nQuery with:  uv run python main.py ask --index {meta['name']} --query \"your question\"")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if args.repo:
        repo = Path(args.repo)
        if not repo.exists() or not repo.is_dir():
            print(f"error: repo path is invalid: {args.repo}", file=sys.stderr)
            return 2
        result = query_repository(str(repo), args.query, strategy=args.strategy, top_k=args.top_k)
    else:
        try:
            result = query_index(args.index, args.query, top_k=args.top_k)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    if args.answer:
        api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("error: --gemini-key or GEMINI_API_KEY env var required for --answer", file=sys.stderr)
            return 2
        if not result["results"]:
            print("No relevant chunks found to answer from.")
            return 0
        answer = generate_answer(args.query, result["results"], api_key, model=args.model)
        if args.json:
            result["answer"] = answer
            print(json.dumps(result, indent=2))
        else:
            print(answer)
        return 0

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    header = "REPO RAG"
    if args.index:
        header += f" (index: {args.index}, hybrid search)"
    print(f"=== {header} ===")
    if args.repo:
        print(f"Repo: {args.repo}")
    print(f"Strategy: {result['strategy']}")
    if "retrieval" in result:
        print(f"Retrieval: {result['retrieval']}")
    print(f"Files indexed: {result['num_files']}")
    print(f"Chunks indexed: {result['num_chunks']}")
    print(f"Query: {result['query']}")
    print()

    if not result["results"]:
        print("No results found.")
        return 0

    for idx, row in enumerate(result["results"], start=1):
        snippet = row["text"].strip().replace("\n", " ")
        snippet = snippet[:180] + ("..." if len(snippet) > 180 else "")
        score_str = f"score={row['score']}"
        if "source" in row:
            score_str += f", source={row['source']}"
        print(f"{idx}. {row['file_id']}  ({score_str})")
        print(f"   chunk={row['chunk_id']}")
        print(f"   {snippet}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
