from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepLense Pydantic-AI simulation agent.",
        epilog=(
            "This CLI adds ./DeepLenseSim to sys.path when present, so you usually do not need PYTHONPATH. "
            "If you set it manually: cmd.exe use  set PYTHONPATH=DeepLenseSim  "
            "(not PYTHONPATH=... as a standalone command name)."
        ),
    )
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Disable stdin confirmation (same as DEEPLENSE_AUTO_APPROVE=1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for .npy/.png outputs (default: ./deeplense_agent_outputs).",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="User message; if empty, read interactively line by line.",
    )
    args = parser.parse_args(argv)

    if args.no_hitl:
        os.environ["DEEPLENSE_AUTO_APPROVE"] = "1"

    # Ensure DeepLenseSim is importable when running from repo root
    repo = Path(__file__).resolve().parents[1]
    sim_root = repo / "DeepLenseSim"
    if sim_root.is_dir():
        sys.path.insert(0, str(sim_root))

    from deeplense_agent.agent_app import build_agent
    from deeplense_agent.tools_runtime import AgentDeps, SessionState

    agent = build_agent()
    deps = AgentDeps(
        interactive_hitl=not args.no_hitl,
        output_dir=args.output_dir,
        session=SessionState(),
    )

    if args.prompt:
        message = " ".join(args.prompt)
    else:
        message = input("You: ").strip()

    if not message:
        print("Empty prompt.")
        return 1

    async def _run() -> str:
        result = await agent.run(message, deps=deps)
        return result.output

    try:
        out = asyncio.run(_run())
    except Exception as e:
        print(f"Agent failed: {e}", file=sys.stderr)
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
