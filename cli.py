from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cast_rag import query_repository, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cAST-style repository RAG retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("experiment", help="Run built-in synthetic experiment")
    exp.set_defaults(func=cmd_experiment)

    ask = sub.add_parser("ask", help="Query a local repository")
    ask.add_argument("--repo", required=True, help="Path to local repository")
    ask.add_argument("--query", required=True, help="Natural language retrieval query")
    ask.add_argument("--strategy", choices=["cast", "fixed"], default="cast", help="Chunking strategy")
    ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    ask.add_argument("--json", action="store_true", help="Emit JSON output")
    ask.set_defaults(func=cmd_ask)
    return parser


def cmd_experiment(_args: argparse.Namespace) -> int:
    run_experiment()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    if not repo.exists() or not repo.is_dir():
        print(f"error: repo path is invalid: {args.repo}", file=sys.stderr)
        return 2

    result = query_repository(str(repo), args.query, strategy=args.strategy, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=== CAST RAG repository query ===")
    print(f"Repo: {repo}")
    print(f"Strategy: {result['strategy']}")
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
        print(f"{idx}. {row['file_id']}  (score={row['score']})")
        print(f"   chunk={row['chunk_id']}")
        print(f"   {snippet}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
