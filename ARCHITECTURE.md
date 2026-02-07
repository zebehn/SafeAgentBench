# SafeAgentBench Architecture (Sequence Diagrams)

This document provides renderable sequence diagrams (Mermaid) to explain the primary execution paths in the codebase.

## 1) Text-Only Planning → Execution → Evaluation (Detailed Tasks)

```mermaid
sequenceDiagram
    autonumber
    participant User as User/Runner
    participant Dataset as Dataset (jsonl)
    participant Planner as LLM Planner
    participant Utils as methods/utils.py
    participant Env as SafeAgentEnv (AI2-THOR Controller)
    participant LL as LowLevelPlanner
    participant ExecEval as Execution Evaluator
    participant SemEval as Semantic Evaluator

    User->>Dataset: Load task (instruction, scene, goal, ref steps)
    User->>Env: Controller(scene)
    User->>Planner: Generate high-level plan from instruction
    Planner-->>User: High-level plan
    User->>Utils: gen_low_level_plan(plan)
    Utils-->>User: Low-level plan
    User->>LL: execute_low_level_plan(low_level_plan)
    LL->>Env: step(action) x N
    Env-->>LL: last_event metadata
    LL-->>User: execution results
    User->>ExecEval: compute_SR_object_state(env_state, goal_state)
    ExecEval-->>User: success_rate, avg_success_ratio
    User->>SemEval: compute_SR_llm(task, plan, ref_plan)
    SemEval-->>User: LLM judgment (success/fail)
```
![Text-only planning sequence (rendered)](figure/architecture/01-text-only-planning.svg)

## 2) Vision-Based Planning Path (Figure 3 in the Paper)

```mermaid
sequenceDiagram
    autonumber
    participant Runner as methods/vision_eval.py
    participant Env as SafeAgentEnv (AI2-THOR Controller)
    participant Vision as methods/map_vlm.py
    participant OpenAI as OpenAI Vision API
    participant Utils as methods/utils.py
    participant LL as LowLevelPlanner
    participant ExecEval as Execution Evaluator
    participant SemEval as Semantic Evaluator

    Runner->>Env: Controller(scene)
    Env-->>Runner: last_event.frame (RGB)
    Runner->>Vision: Agents(img, instruction)
    Vision->>OpenAI: multi_agent_vision_planning(image + task)
    OpenAI-->>Vision: vision-based plan
    Vision-->>Runner: plan text
    Runner->>Utils: gen_low_level_plan(plan)
    Utils-->>Runner: low-level plan
    Runner->>LL: execute_low_level_plan(low_level_plan)
    LL->>Env: step(action) x N
    Env-->>LL: last_event metadata
    LL-->>Runner: execution results
    Runner->>ExecEval: compute_SR_object_state(env_state, goal_state)
    ExecEval-->>Runner: success_rate, avg_success_ratio
    Runner->>SemEval: compute_SR_llm(task, plan, ref_plan)
    SemEval-->>Runner: LLM judgment (success/fail)
```
![Vision-based planning sequence (rendered)](figure/architecture/02-vision-planning.svg)

## 3) Low-Level Controller Action Mapping

```mermaid
sequenceDiagram
    autonumber
    participant Planner as LowLevelPlanner
    participant Env as AI2-THOR Controller
    participant World as Simulated World

    Planner->>Planner: parse instruction (e.g., "open microwave")
    Planner->>Env: step(OpenObject, objectId)
    Env->>World: apply physics + action
    World-->>Env: new state
    Env-->>Planner: last_event metadata
    Planner-->>Planner: success/failure + errorMessage
```
![Low-level controller mapping (rendered)](figure/architecture/03-low-level-controller.svg)

## 4) Execution Evaluator (Object-State Check)

```mermaid
sequenceDiagram
    autonumber
    participant Eval as evaluator/detail_evaluate.py
    participant Env as AI2-THOR Controller
    participant Goal as Goal Conditions

    Eval->>Env: read last_event.metadata["objects"]
    Eval->>Goal: compare objectType + properties
    Goal-->>Eval: per-object match ratios
    Eval-->>Eval: success_rate + avg_success_ratio
```
![Execution evaluator object-state check (rendered)](figure/architecture/04-execution-evaluator.svg)

## 5) Semantic Evaluator (LLM Plan Feasibility)

```mermaid
sequenceDiagram
    autonumber
    participant Eval as evaluator/detail_evaluate.py
    participant OpenAI as OpenAI API

    Eval->>OpenAI: task + plan + ref plan + rules
    OpenAI-->>Eval: rating (success/fail) + explanation
```
![Semantic evaluator feasibility check (rendered)](figure/architecture/05-semantic-evaluator.svg)

## 6) Long-Horizon Safety Evaluation (Implicit Hazards)

```mermaid
sequenceDiagram
    autonumber
    participant Eval as evaluator/long_horizon_evaluate.py
    participant OpenAI as OpenAI API

    Eval->>OpenAI: task + plan + safety requirements
    OpenAI-->>Eval: rating (Completed and safe / Completed but unsafe / Incomplete)
```
![Long-horizon safety evaluation (rendered)](figure/architecture/06-long-horizon-safety.svg)

## 7) OpenAI Call with Retry (Backoff Wrapper)

```mermaid
sequenceDiagram
    autonumber
    participant Caller as methods/utils.py
    participant OpenAI as OpenAI API

    loop retry_with_exponential_backoff
        Caller->>OpenAI: chat.completions.create(...)
        alt success
            OpenAI-->>Caller: response
        else transient error
            OpenAI-->>Caller: error
            Caller->>Caller: sleep(backoff)
        end
    end
```
![OpenAI retry wrapper (rendered)](figure/architecture/07-openai-retry.svg)

## Notes
- Vision path is implemented in `methods/map_vlm.py` and exposed via `methods/vision_eval.py`.
- Text-only path uses `methods/utils.py` for plan conversion + `low_level_controller/` for execution.
- Execution evaluator relies on goal conditions in the dataset; semantic evaluator can cover ambiguous tasks.
