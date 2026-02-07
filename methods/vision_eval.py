import argparse
import json
import os
import sys
import time
import logging
from typing import Any, Dict, List, Optional

import jsonlines
from ai2thor.controller import Controller

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.detail_evaluate import compute_SR_llm, compute_SR_object_state
from low_level_controller.low_level_controller import LowLevelPlanner
from methods.map_vlm import Agents

_LOGGER = logging.getLogger(__name__)
from methods.utils import all_objs, execute_low_level_plan, gen_low_level_plan


def _load_samples(dataset_path: str, num: int, start: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with jsonlines.open(dataset_path) as reader:
        for idx, item in enumerate(reader):
            if idx < start:
                continue
            samples.append(item)
            if len(samples) >= num:
                break
    return samples


def _run_single(
    sample: Dict[str, Any],
    model: str,
    headless: bool,
    server_timeout: float,
    server_start_timeout: float,
    skip_llm_eval: bool,
    verbose: bool,
) -> Dict[str, Any]:
    controller: Optional[Controller] = None
    try:
        if verbose:
            _LOGGER.info("Starting episode: scene=%s", sample.get("scene_name"))
            _LOGGER.info("Instruction: %s", sample.get("instruction"))

        controller = Controller(
            scene=sample["scene_name"],
            headless=headless,
            server_timeout=server_timeout,
            server_start_timeout=server_start_timeout,
        )

        objs_all = all_objs(controller)
        img = controller.last_event.frame
        agent = Agents(img, sample["instruction"], debug=verbose)
        env_info, plan = agent.multi_agent_vision_planning(objs_all)
        if verbose:
            _LOGGER.info("Planner returned plan_raw:\n%s", plan)

        low_level_plan = gen_low_level_plan(plan)
        if verbose:
            _LOGGER.info("Low-level plan (%d steps): %s", len(low_level_plan), low_level_plan)

        planner = LowLevelPlanner(controller)
        t0 = time.time()
        metadata_curr, sr_step = execute_low_level_plan(low_level_plan, planner)
        exec_time = time.time() - t0
        if verbose:
            _LOGGER.info("Execution finished: sr_step=%.3f exec_time_sec=%.2f", sr_step, exec_time)
        state_curr = metadata_curr["objects"]

        final_state = sample.get("final_state")
        exec_success = None
        exec_avg = None
        if final_state is not None:
            exec_success, exec_avg = compute_SR_object_state(state_curr, final_state)
            if verbose:
                _LOGGER.info("Object-state eval: exec_success=%s exec_avg=%.3f", exec_success, exec_avg)

        llm_score = None
        llm_exp = None
        retry_time = None
        if not skip_llm_eval:
            llm_score, llm_exp, retry_time = compute_SR_llm(
                sample["instruction"],
                low_level_plan,
                sample.get("step", []),
                model=model,
            )
            if verbose:
                _LOGGER.info("LLM eval: llm_score=%s llm_retry_time_sec=%s", llm_score, retry_time)

        return {
            "scene": sample["scene_name"],
            "instruction": sample["instruction"],
            "plan_raw": plan,
            "plan_low_level": low_level_plan,
            "sr_step": sr_step,
            "exec_success": exec_success,
            "exec_avg": exec_avg,
            "llm_score": llm_score,
            "llm_retry_time_sec": retry_time,
            "env_info": env_info,
            "exec_time_sec": exec_time,
        }
    finally:
        if controller is not None:
            controller.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision-based SafeAgentBench evaluator")
    parser.add_argument("--dataset", required=True, help="Path to jsonl dataset file")
    parser.add_argument("--num", type=int, default=1, help="Number of samples to run")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--headless", action="store_true", help="Run AI2-THOR headless")
    parser.add_argument("--server-timeout", type=float, default=120.0)
    parser.add_argument("--server-start-timeout", type=float, default=120.0)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--skip-llm-eval", action="store_true")
    parser.add_argument("--out", default="", help="Optional JSONL output path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed step logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    samples = _load_samples(args.dataset, args.num, args.start)
    if not samples:
        raise SystemExit("No samples found for the given dataset/start/num.")

    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if args.verbose:
            _LOGGER.info("Episode %d/%d", idx + 1, len(samples))
        result = _run_single(
            sample=sample,
            model=args.model,
            headless=args.headless,
            server_timeout=args.server_timeout,
            server_start_timeout=args.server_start_timeout,
            skip_llm_eval=args.skip_llm_eval,
            verbose=args.verbose,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=True))

    if args.out:
        with jsonlines.open(args.out, "w") as writer:
            for r in results:
                writer.write(r)


if __name__ == "__main__":
    main()
