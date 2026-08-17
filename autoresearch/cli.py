"""One small command surface for design, seal, execute, judge, and synthesize."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .bank import BankIndex, rebuild_bank_views
from .campaign import CampaignError, CampaignQueue
from .chart import render_page, render_svg
from .doctor import diagnose
from .evidence import EvidenceEngine
from .execution import ExecutionService, NoPendingReplicate, NoResourceAvailable
from .exploration import CANDIDATE_TRACKS
from .knowledge import KnowledgeEngine
from .loop import SEARCH_SEED, FastLoop
from .protocol import PROMOTION_SEEDS
from .records import RecordError, canonical_json, read_json
from .research import ResearchEngine
from .science import ScientificLibrary
from .sealing import SealingAuthority
from .store import Store
from .workflow import V2Workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch")
    parser.add_argument(
        "--root",
        default=".autoresearch",
        help="state directory (default: .autoresearch)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize an empty immutable store")

    design = commands.add_parser("design", help="create an immutable ExperimentSpec")
    design.add_argument("proposal")

    review = commands.add_parser("review-template", help="render reviews bound to a spec")
    review.add_argument("spec_id")
    review.add_argument("--output")

    seal = commands.add_parser("seal", help="produce an immutable ExecutionManifest")
    seal.add_argument("spec_id")
    seal.add_argument("--execution", required=True)
    seal.add_argument("--reviews")

    execute = commands.add_parser("execute", help="run only from a sealed manifest")
    execute.add_argument("manifest_id")
    execute.add_argument("--all", action="store_true")
    execute.add_argument("--workers", type=int, default=1)

    judge = commands.add_parser("judge", help="create EvidenceDecisions")
    judge_target = judge.add_mutually_exclusive_group(required=True)
    judge_target.add_argument("--result")
    judge_target.add_argument("--all", action="store_true")

    commands.add_parser("synthesize", help="rebuild beliefs, portfolio, papers, and SOTA")

    paper = commands.add_parser("paper", help="register immutable paper coverage")
    paper.add_argument("declaration")

    inspect = commands.add_parser("inspect", help="print one immutable record")
    inspect.add_argument("kind")
    inspect.add_argument("record_id")

    commands.add_parser("status", help="show compact live state")
    commands.add_parser("validate", help="validate records, references, blobs, and inflight state")

    cycle = commands.add_parser("cycle", help="run the complete five-engine cycle")
    cycle.add_argument("proposal")
    cycle.add_argument("--execution", required=True)
    cycle.add_argument("--reviews")
    cycle.add_argument("--workers", type=int, default=1)

    screen = commands.add_parser("screen", help="fast loop: one seed, one arm, one run")
    screen.add_argument("label", help="short slug, becomes exp_<label>")
    screen.add_argument("--summary", required=True, help="human-readable change description")
    screen.add_argument("--execution", required=True)
    screen.add_argument("--argv", required=True, nargs=argparse.REMAINDER)
    screen.add_argument("--seed", type=int, default=SEARCH_SEED)
    screen.add_argument("--minimum-steps", type=int, default=1)
    screen.add_argument("--direction", default="unassigned")
    screen.add_argument("--subsystem", default="train_py")
    screen.add_argument("--source-ids", default="baseline")
    screen.add_argument("--workers", type=int, default=1)
    screen.add_argument(
        "--no-gpu", action="store_true", help="screen on a CPU resource (smoke tests only)"
    )
    screen.add_argument(
        "--isolation", default="continuous", choices=["none", "launch", "continuous"]
    )

    commands.add_parser("leaderboard", help="rebuild the leaderboard and progress chart")

    claims = commands.add_parser("claims", help="list or release in-flight execution claims")
    claims.add_argument("--release", help="claim token to release")
    claims.add_argument(
        "--confirm-dead",
        action="store_true",
        help="required with --release; asserts the process was verified dead",
    )

    enqueue = commands.add_parser(
        "enqueue", help="add sealed manifests to the durable resident-worker queue"
    )
    enqueue.add_argument("manifest_ids", nargs="+")

    run = commands.add_parser("run", help="drain the durable queue with resident workers")
    run.add_argument(
        "--workers",
        type=int,
        default=4,
        help="resident worker count (default: 4; resource/host/workdir leases still cap launches)",
    )
    run.add_argument("--follow", action="store_true", help="wait for newly enqueued work")
    run.add_argument("--poll-seconds", type=float, default=1.0)
    run.add_argument(
        "--idle-timeout-seconds",
        type=float,
        help="stop follow mode after this long without queue activity",
    )
    run.add_argument(
        "--ignore-health",
        action="store_true",
        help="override the rolling invalid-result circuit breaker",
    )
    run.add_argument(
        "--science-agenda",
        help="refresh this literature agenda concurrently while GPU workers drain",
    )
    run.add_argument("--literature-limit", type=int, default=25)
    run.add_argument("--openalex-mailto")

    queue = commands.add_parser("queue", help="inspect or reconcile durable queue state")
    queue.add_argument("--jobs", action="store_true", help="include every job record")
    queue.add_argument(
        "--reconcile",
        action="store_true",
        help="recover abandoned unstarted jobs and surface uncertain ones",
    )

    commands.add_parser("bank", help="rebuild per-GPU controls and promotion candidates")

    doctor = commands.add_parser(
        "doctor", help="check calibration freshness, timing bottlenecks, and campaign health"
    )
    doctor.add_argument("--bank-id")

    calibrate = commands.add_parser(
        "calibrate",
        help="stage one executable-seed control per declared resource/GPU",
    )
    calibrate.add_argument("bank_id", help="baseline revision name")
    calibrate.add_argument("label", help="unique calibration generation label")
    calibrate.add_argument("--scope", required=True)
    calibrate.add_argument("--execution", required=True)
    calibrate.add_argument("--mutable-code-path", action="append", default=[])
    calibrate.add_argument("--minimum-steps", type=int, default=1)
    calibrate.add_argument("--seed", type=int, default=SEARCH_SEED)
    calibrate.add_argument("--argv", required=True, nargs=argparse.REMAINDER)

    search = commands.add_parser(
        "search",
        help="stage a one-arm candidate against a frozen same-GPU control",
    )
    search.add_argument("label")
    search.add_argument("--bank-id", required=True)
    search.add_argument("--summary", required=True)
    search.add_argument("--scope", required=True)
    search.add_argument("--execution", required=True)
    search.add_argument("--mutable-code-path", action="append", default=[])
    search.add_argument("--minimum-steps", type=int, default=1)
    search.add_argument("--seed", type=int, default=SEARCH_SEED)
    search.add_argument("--direction", default="unassigned")
    search.add_argument("--subsystem", default="model")
    search.add_argument("--source-ids", default="banked_control")
    search.add_argument(
        "--hypothesis-id",
        action="append",
        default=[],
        help="immutable scientific hypothesis tested by this candidate (exactly one)",
    )
    search.add_argument(
        "--allow-overhead-dominated",
        action="store_true",
        help="explicitly bypass the profile-first model-search gate",
    )
    search.add_argument(
        "--allow-weak-science",
        action="store_true",
        help="explicitly test a hypothesis with foundation confidence below 0.55",
    )
    search.add_argument(
        "--track",
        default="mechanism",
        choices=list(CANDIDATE_TRACKS),
        help=(
            "exploration-budget classification: mechanism (new structure never run), "
            "knob (new value for an existing constant), throughput (buys steps)"
        ),
    )
    search.add_argument(
        "--family",
        help="mechanism family for the budget's breadth and repair-loop rules "
        "(defaults to --subsystem)",
    )
    search.add_argument(
        "--exploration-override",
        help="written reason, >=20 characters, recorded immutably, permitting a "
        "candidate the exploration budget would otherwise refuse",
    )
    search.add_argument("--argv", required=True, nargs=argparse.REMAINDER)

    promote = commands.add_parser(
        "promote", help="stage a gate-clearing candidate for distinct-seed confirmation"
    )
    promote.add_argument("candidate_spec_id")
    promote.add_argument(
        "--execution",
        help="optional execution file supplying only the resources/runtime shape",
    )
    promote.add_argument("--reviews", help="optional digest-bound review declaration")
    promote.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in PROMOTION_SEEDS),
        help="comma-separated distinct confirmation seeds",
    )
    promote.add_argument("--minimum-valid", type=int, default=4)
    promote.add_argument(
        "--proposal-only", action="store_true", help="print the proposal without registering it"
    )

    agenda = commands.add_parser("agenda", help="register an immutable research agenda")
    agenda.add_argument("declaration")

    literature_source = commands.add_parser(
        "literature-source", help="register metadata or a snapshotted full-text source"
    )
    literature_source.add_argument("declaration")

    literature_search = commands.add_parser(
        "literature-search", help="search OpenAlex and preserve the query/result snapshots"
    )
    literature_search.add_argument("query")
    literature_search.add_argument("--limit", type=int, default=25)
    literature_search.add_argument("--agenda-id")
    literature_search.add_argument("--topic-id")
    literature_search.add_argument("--mailto")

    literature_refresh = commands.add_parser(
        "literature-refresh", help="refresh every due topic in an immutable agenda"
    )
    literature_refresh.add_argument("agenda_id")
    literature_refresh.add_argument("--limit", type=int, default=25)
    literature_refresh.add_argument("--mailto")
    literature_refresh.add_argument("--force", action="store_true")

    scientific_claim = commands.add_parser(
        "scientific-claim", help="register one attributed and scoped claim"
    )
    scientific_claim.add_argument("declaration")

    mechanism = commands.add_parser(
        "mechanism", help="register a claim-supported causal mechanism graph"
    )
    mechanism.add_argument("declaration")

    hypothesis = commands.add_parser(
        "hypothesis", help="register a falsifiable mechanism-derived hypothesis"
    )
    hypothesis.add_argument("declaration")

    commands.add_parser(
        "science", help="cheaply rebuild beliefs, mechanisms, hypotheses, gaps, and ideas"
    )

    conclude = commands.add_parser(
        "conclude", help="materialize an experiment-supported or refuted scientific claim"
    )
    conclude.add_argument("hypothesis_id")
    conclude.add_argument("spec_id")
    conclude.add_argument("--proposal-only", action="store_true")
    return parser


def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": record["kind"],
        "id": record["id"],
        "digest": record["digest"],
        "created_at": record["created_at"],
    }


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _publish_board(store: Store) -> dict[str, Any]:
    """Rebuild the leaderboard and redraw the chart.

    Called after every screen: the chart is a projection of the registry, so it
    is never batched and can never drift from the records it is drawn from.
    """
    board = FastLoop(store).write_views()
    store.write_view("progress.svg", render_svg(board))
    store.write_view("progress.html", render_page(board))
    return board


def _payload_argv(raw: Sequence[str], command: str) -> list[str]:
    argv = [item for item in raw if item != "--"]
    if not argv:
        raise RecordError(f"{command} requires a non-empty --argv command")
    if argv[0] == "@python":
        argv[0] = sys.executable
    return argv


def _job_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "manifest_id": row["manifest_id"],
        "state": row["state"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = Store(args.root)
    research = ResearchEngine(store)
    sealing = SealingAuthority(store)
    execution = ExecutionService(store)
    evidence = EvidenceEngine(store)
    knowledge = KnowledgeEngine(store)
    science = ScientificLibrary(store)
    try:
        if args.command == "init":
            store.init()
            _print({"state": "initialized", "root": str(store.root)})
        elif args.command == "design":
            _print(_record_summary(research.create_from_file(args.proposal)))
        elif args.command == "review-template":
            template = sealing.review_template(args.spec_id)
            if args.output:
                Path(args.output).write_text(canonical_json(template) + "\n", encoding="utf-8")
                _print({"output": str(Path(args.output).resolve()), "spec_id": args.spec_id})
            else:
                _print(template)
        elif args.command == "seal":
            record = sealing.seal_from_files(args.spec_id, args.execution, args.reviews)
            _print(_record_summary(record))
        elif args.command == "execute":
            if args.all:
                rows, unfinished = execution.execute_all(args.manifest_id, workers=args.workers)
                _print(
                    {
                        "results": [_record_summary(row) for row in rows],
                        "unfinished_replicates": unfinished,
                    }
                )
                return 0 if not unfinished else 3
            _print(_record_summary(execution.execute_next(args.manifest_id)))
        elif args.command == "judge":
            rows = evidence.judge_all() if args.all else [evidence.judge(args.result)]
            _print({"decisions": [_record_summary(row) for row in rows]})
        elif args.command == "synthesize":
            _print(knowledge.synthesize())
        elif args.command == "paper":
            declaration_path = Path(args.declaration).resolve()
            record = research.register_paper(
                read_json(declaration_path), base=declaration_path.parent
            )
            _print(_record_summary(record))
        elif args.command == "agenda":
            _print(_record_summary(science.register_agenda(read_json(Path(args.declaration)))))
        elif args.command == "literature-source":
            declaration_path = Path(args.declaration).resolve()
            record = science.register_source(
                read_json(declaration_path), base=declaration_path.parent
            )
            _print(_record_summary(record))
        elif args.command == "literature-search":
            outcome = science.search_openalex(
                args.query,
                limit=args.limit,
                agenda_id=args.agenda_id,
                topic_id=args.topic_id,
                mailto=args.mailto,
            )
            _print(
                {
                    "search": _record_summary(outcome["search"]),
                    "sources": [_record_summary(row) for row in outcome["sources"]],
                }
            )
        elif args.command == "literature-refresh":
            _print(
                science.refresh_agenda(
                    args.agenda_id,
                    limit_per_query=args.limit,
                    mailto=args.mailto,
                    force=args.force,
                )
            )
        elif args.command == "scientific-claim":
            _print(_record_summary(science.register_claim(read_json(Path(args.declaration)))))
        elif args.command == "mechanism":
            _print(_record_summary(science.register_mechanism(read_json(Path(args.declaration)))))
        elif args.command == "hypothesis":
            _print(_record_summary(science.register_hypothesis(read_json(Path(args.declaration)))))
        elif args.command == "science":
            _print(science.synthesize())
        elif args.command == "conclude":
            declaration = science.conclusion_template(args.hypothesis_id, args.spec_id)
            if args.proposal_only:
                _print(declaration)
            else:
                _print(_record_summary(science.register_claim(declaration)))
        elif args.command == "inspect":
            _print(store.get(args.kind, args.record_id))
        elif args.command == "status":
            validation = store.validate()
            snapshot_path = store.views_dir / "KNOWLEDGE.json"
            snapshot = read_json(snapshot_path) if snapshot_path.exists() else None
            science_path = store.views_dir / "SCIENCE.json"
            science_snapshot = read_json(science_path) if science_path.exists() else None
            bank = BankIndex.from_store(store)
            _print(
                {
                    "root": str(store.root),
                    "counts": validation["counts"],
                    "inflight": validation["inflight"],
                    "leases": store.resource_leases(),
                    "queue": CampaignQueue(store).status(),
                    "bank": {
                        "controls": len(bank.controls),
                        "eligible_controls": sum(
                            bool(row["eligible_now"]) for row in bank.bank_view()["controls"]
                        ),
                        "promotion_due": len(bank.promotion_view()["promotion_queue"]),
                    },
                    "valid": validation["valid"],
                    "portfolio": snapshot.get("portfolio") if snapshot else None,
                    "paper_status": snapshot.get("paper_status") if snapshot else None,
                    "sota": snapshot.get("sota") if snapshot else None,
                    "science": science_snapshot.get("summary") if science_snapshot else None,
                }
            )
        elif args.command == "validate":
            result = store.validate()
            _print(result)
            return 0 if result["valid"] else 1
        elif args.command == "cycle":
            spec = research.create_from_file(args.proposal)
            manifest = sealing.seal_from_files(spec["id"], args.execution, args.reviews)
            results, unfinished = execution.execute_all(manifest["id"], workers=args.workers)
            decisions = [evidence.judge(row["id"]) for row in results]
            snapshot = knowledge.synthesize()
            _print(
                {
                    "spec": _record_summary(spec),
                    "manifest": _record_summary(manifest),
                    "results": [_record_summary(row) for row in results],
                    "decisions": [_record_summary(row) for row in decisions],
                    "unfinished_replicates": unfinished,
                    "knowledge": snapshot,
                }
            )
        elif args.command == "screen":
            execution_path = Path(args.execution).resolve()
            argv = [item for item in args.argv if item != "--"]
            if not argv:
                raise RecordError("screen requires a non-empty --argv command")
            outcome = FastLoop(store).screen(
                args.label,
                change_summary=args.summary,
                execution=read_json(execution_path),
                argv=argv,
                seed=args.seed,
                minimum_steps=args.minimum_steps,
                source_ids=[item for item in args.source_ids.split(",") if item],
                direction=args.direction,
                subsystem=args.subsystem,
                require_gpu=not args.no_gpu,
                isolation="none" if args.no_gpu else args.isolation,
                base=execution_path.parent,
                workers=args.workers,
            )
            board = _publish_board(store)
            _print(
                {
                    "spec": _record_summary(outcome["spec"]),
                    "manifest": _record_summary(outcome["manifest"]),
                    "results": [_record_summary(row) for row in outcome["results"]],
                    "decisions": [
                        {
                            **_record_summary(row),
                            "verdict": row["payload"]["measurement_verdict"],
                            "reasons": row["payload"]["reasons"],
                            "measurements": row["payload"]["measurements"],
                        }
                        for row in outcome["decisions"]
                    ],
                    "unfinished_replicates": outcome["unfinished"],
                    "running_best": board["best_value"],
                    "best_spec_id": board["best_spec_id"],
                    "audit_due": board["audit_due"],
                }
            )
        elif args.command == "leaderboard":
            board = _publish_board(store)
            _print(
                {
                    "experiments": board["experiments"],
                    "accepted": board["accepted"],
                    "invalid": board["invalid"],
                    "best_value": board["best_value"],
                    "best_spec_id": board["best_spec_id"],
                    "audit_due": board["audit_due"],
                    "views": sorted(
                        str(path.relative_to(store.root))
                        for path in store.views_dir.glob("LEADERBOARD*")
                    )
                    + [f"views/{name}" for name in ("progress.svg", "progress.html")],
                }
            )
        elif args.command == "claims":
            if args.release:
                if not args.confirm_dead:
                    raise RecordError(
                        "--release requires --confirm-dead: verify the local and remote "
                        "process are actually gone before removing a claim"
                    )
                _print({"released": store.release_claim(args.release)})
            else:
                _print({"claims": store.claims()})
        elif args.command == "enqueue":
            queue = CampaignQueue(store)
            jobs = [queue.enqueue(manifest_id) for manifest_id in args.manifest_ids]
            _print(
                {
                    "jobs": [
                        {
                            "job_id": row["job_id"],
                            "manifest_id": row["manifest_id"],
                            "state": row["state"],
                        }
                        for row in jobs
                    ],
                    "status": queue.status(),
                }
            )
        elif args.command == "run":
            queue = CampaignQueue(store)
            science_refresh = None
            science_error = None
            if args.science_agenda:
                # Literature/network work is CPU-side and cannot sit between GPU
                # launches.  A separate thread updates immutable records while the
                # resident queue retains its ordinary event-driven launch cadence.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        science.refresh_agenda,
                        args.science_agenda,
                        limit_per_query=args.literature_limit,
                        mailto=args.openalex_mailto,
                    )
                    outcome = queue.work(
                        workers=args.workers,
                        follow=args.follow,
                        poll_seconds=args.poll_seconds,
                        idle_timeout_seconds=args.idle_timeout_seconds,
                        ignore_health=args.ignore_health,
                    )
                    try:
                        science_refresh = future.result()
                    except RecordError as exc:
                        science_error = str(exc)
            else:
                outcome = queue.work(
                    workers=args.workers,
                    follow=args.follow,
                    poll_seconds=args.poll_seconds,
                    idle_timeout_seconds=args.idle_timeout_seconds,
                    ignore_health=args.ignore_health,
                )
            views = rebuild_bank_views(store)
            science_snapshot = science.synthesize()
            _print(
                {
                    **outcome,
                    "bank_controls": len(views["bank"]["controls"]),
                    "promotion_due": len(views["promotions"]["promotion_queue"]),
                    "science": science_snapshot["summary"],
                    "science_refresh": science_refresh,
                    "science_error": science_error,
                }
            )
        elif args.command == "queue":
            queue = CampaignQueue(store)
            reconciled = queue.reconcile() if args.reconcile else None
            result = {"status": queue.status()}
            if reconciled is not None:
                result["reconciled"] = [
                    {"job_id": row["job_id"], "state": row["state"]} for row in reconciled
                ]
            if args.jobs:
                result["jobs"] = queue.jobs()
            _print(result)
        elif args.command == "bank":
            views = rebuild_bank_views(store)
            _print(
                {
                    "controls": len(views["bank"]["controls"]),
                    "eligible_controls": sum(
                        bool(row["eligible_now"]) for row in views["bank"]["controls"]
                    ),
                    "candidates": len(views["promotions"]["candidates"]),
                    "promotion_due": len(views["promotions"]["promotion_queue"]),
                    "views": ["views/BANK.json", "views/PROMOTION_QUEUE.json"],
                }
            )
        elif args.command == "doctor":
            report = diagnose(store, bank_id=args.bank_id)
            _print(report)
            return 0 if report["model_search_ready"] else 4
        elif args.command == "calibrate":
            execution_path = Path(args.execution).resolve()
            scope_path = Path(args.scope).resolve()
            outcome = V2Workflow(store).stage_calibration(
                args.bank_id,
                args.label,
                read_json(execution_path),
                _payload_argv(args.argv, "calibrate"),
                read_json(scope_path),
                args.mutable_code_path,
                seed=args.seed,
                base=execution_path.parent,
                minimum_steps=args.minimum_steps,
            )
            _print(
                {
                    "bank_id": outcome["bank_id"],
                    "scope_id": outcome["scope"]["id"],
                    "staged": [
                        {
                            "resource_id": row["resource_id"],
                            "gpu": row["gpu"],
                            "spec": _record_summary(row["spec"]),
                            "manifest": _record_summary(row["manifest"]),
                            "job": _job_summary(row["job"]),
                        }
                        for row in outcome["staged"]
                    ],
                    "next": "run the queue, then use doctor before model search",
                }
            )
        elif args.command == "search":
            execution_path = Path(args.execution).resolve()
            scope = read_json(Path(args.scope).resolve())
            diagnostic_subsystems = (
                "calibration",
                "compile",
                "data",
                "evaluation",
                "evaluator",
                "input",
                "instrumentation",
                "tokenizer",
            )
            model_like = (
                not args.subsystem.lower().replace("-", "_").startswith(diagnostic_subsystems)
            )
            if model_like and scope.get("hardware_class") != "cpu-test" and not args.hypothesis_id:
                raise RecordError(
                    "model search requires --hypothesis-id; use a diagnostic subsystem for "
                    "bottleneck work or explicitly register the scientific chain first"
                )
            outcome = V2Workflow(store).stage_candidate(
                args.label,
                read_json(execution_path),
                _payload_argv(args.argv, "search"),
                scope,
                args.mutable_code_path,
                args.bank_id,
                seed=args.seed,
                summary=args.summary,
                direction=args.direction,
                subsystem=args.subsystem,
                source_ids=[item for item in args.source_ids.split(",") if item],
                hypothesis_ids=args.hypothesis_id,
                base=execution_path.parent,
                minimum_steps=args.minimum_steps,
                allow_overhead_dominated=args.allow_overhead_dominated,
                allow_weak_science=args.allow_weak_science,
                track=args.track,
                family=args.family,
                exploration_override=args.exploration_override,
            )
            _print(
                {
                    "spec": _record_summary(outcome["spec"]),
                    "manifest": _record_summary(outcome["manifest"]),
                    "job": _job_summary(outcome["job"]),
                    "resource_id": outcome["resource_id"],
                    "gpu": outcome["gpu"],
                    "control_result_id": outcome["reference_control"]["result_id"],
                    "gpu_key": outcome["reference_control"]["gpu_key"],
                    "next": "run the queue; `autoresearch bank` will score the result",
                }
            )
        elif args.command == "promote":
            try:
                seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
            except ValueError as exc:
                raise RecordError("--seeds must be comma-separated integers") from exc
            workflow = V2Workflow(store)
            if args.proposal_only:
                _print(
                    workflow.promotion_proposal(
                        args.candidate_spec_id,
                        seeds=seeds,
                        minimum_valid_replicates=args.minimum_valid,
                    )
                )
            else:
                execution_override = None
                if args.execution:
                    raw_execution = read_json(Path(args.execution).resolve())
                    execution_override = {
                        "resources": raw_execution.get("resources"),
                        "runtime": raw_execution.get("runtime"),
                    }
                reviews = read_json(Path(args.reviews).resolve()) if args.reviews else None
                outcome = workflow.stage_promotion(
                    args.candidate_spec_id,
                    execution=execution_override,
                    reviews=reviews,
                    seeds=seeds,
                    minimum_valid_replicates=args.minimum_valid,
                )
                _print(
                    {
                        "spec": _record_summary(outcome["spec"]),
                        "manifest": _record_summary(outcome["manifest"]),
                        "job": _job_summary(outcome["job"]),
                        "reviews_supplied": outcome["reviews_supplied"],
                        "next": "run the queue, then synthesize to evaluate promotion",
                    }
                )
        return 0
    except (CampaignError, RecordError, NoPendingReplicate, NoResourceAvailable) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
