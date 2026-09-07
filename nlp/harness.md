---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Harness

LLM orchestration through Claude Code, OpenAI Codex & GitHub Copilot: context → tools → loops → graphs → hooks → delivery.
The harness owns context, permitted execution & lifecycle; the model proposes actions within that runtime.

Date: September 7, 2026

Notations:
- $\mathcal{G}$: Task-dependency graph for an execution attempt.
- $\mathcal{V}$: Task nodes.
- $\mathcal{E}$: Required dependency edges.
- $v$: Task node.

&nbsp;

## Context

### Repo Instructions
- **What**: Persistently discovered project guidance.
- **Why**: Repeating conventions in every prompt is fragile.
- **How**:
    1. Put repo-wide invariants near the root.
    2. Scope local conventions to the relevant subtree.
    3. State exact commands, prohibited changes & completion evidence.
    4. Keep task-specific progress out of permanent instructions.

```{dropdown} Table: Instruction Entry Points
| Product surface | Native repo entry points | Important distinction |
|:--|:--|:--|
| Claude Code | `CLAUDE.md`; `.claude/rules/` | Instructions guide the model; Perms enforce allowed actions |
| Codex | `AGENTS.md`; `AGENTS.override.md` | Discovery walks toward the working dir; Nearer guidance overrides broader guidance |
| Copilot CLI | `AGENTS.md`; `.github/copilot-instructions.md`; `.github/instructions/**/*.instructions.md` | Supported instruction types & precedence differ across Copilot CLI, IDE & Cloud |
```

````{note} Example
:class: dropdown
- Instruction content for a hypothetical Python package:

```text
Use the existing unittest suite.
Parser changes belong in src/parser.py and tests/test_parser.py.
Preserve the public parse() signature.
Run python3 -m unittest discover -s tests -q after changing parser behavior.
Do not publish packages or push branches w/o approval.
Report the exact failed check if completion is blocked.
```
````

```{attention} Q&A
:class: dropdown
*Can an instruction enforce a security boundary?*

- No. “Never write outside this folder” must be backed by tool permissions or filesystem isolation when the boundary matters.

*Why not put the whole architecture guide here?*

- Always-loaded text competes with task evidence.
- Keep navigation & invariants here; retrieve detailed references when relevant.

*Are similarly named files interchangeable across products?*

- No. Discovery, scope, overrides & trust rules are product-specific.
- Share the invariant content; adapt the loader configuration.
```

&nbsp;

### Skills
- **What**: Discoverable packages of task instructions, references & optional scripts.
- **Why**: Repeated procedures need reuse w/o loading every procedure into every convo.
- **How**:
    1. Advertise a short description of when the skill applies.
    2. Load `SKILL.md` when selected.
    3. Read referenced material or run bundled scripts only as needed.
    4. Keep side-effecting procedures explicitly controlled.

````{note} Example
:class: dropdown
```text
---
name: parser-change
description: Fix identifier-parsing bugs.
---
Preserve parse()'s API. Reproduce → patch → run existing parser tests.
```
````

```{dropdown} Table: Repo Skill Locations
| Product | Conventional project location | Discovery vs execution |
|:--|:--|:--|
| Claude Code | `.claude/skills/<name>/SKILL.md` | Description enables discovery; Body loads on invocation |
| Codex | `.agents/skills/<name>/SKILL.md` | Explicit selection or task-based matching |
| Copilot | `.github/skills/<name>/SKILL.md` | Relevant skill content is loaded when needed |
```

```{attention} Q&A
:class: dropdown
*Skill vs subagent?*

- Skill = reusable procedure; subagent = separate execution context.
- A skill can instruct delegation; it is not inherently another model call or worker.

*Skill vs hook?*

- Skill selection is explicit or model-driven; a configured hook fires at a runtime event.

*Does a skill's tool field necessarily restrict tools?*

- No. In Claude Code, `allowed-tools` pre-approves listed tools for that turn; it is not a tool allowlist.
- Use the product's actual restriction mechanism rather than inferring semantics from the field name.
```

&nbsp;

### Just-in-Time Retrieval
- **What**: Load task-relevant evidence when needed, rather than preloading the corpus. {cite:p}`anthropic_context,codex_cli_features,copilot_cli_reference`
- **Why**: More context can mean more irrelevant/stale evidence, NOT more understanding.
- **How**:
    1. Keep lightweight pointers: paths, symbols, query handles & artifact IDs.
    2. Locate the relevant surface before reading its details.
    3. Retrieve bounded sections & inspect their dependencies.
    4. Expand only when the curr evidence leaves a real gap.

```{note} Example
:class: dropdown
- Parser bug → find `parse_identifier` → read its implementation, callers & relevant tests—not the whole repo.
```

```{attention} Q&A
:class: dropdown
*Is this exclusive to Claude Code?*

- No. All three can retrieve task evidence through search/read tools during execution; Anthropic explicitly names the pattern.
- This retrieves task content; deferred tool discovery retrieves tool definitions.

*Does a coding harness require a vector database?*

- No. Claude Code already uses filesystem navigation & targeted retrieval.
- Semantic retrieval is another tool, not the definition of a harness.

*When does retrieval fail?*

- Wrong query, stale index, clipped output or an omitted dependency.
- “No match” is evidence about the query, not proof that the behavior does not exist.
```

&nbsp;

### Compaction
- **What**: Reduce active convo context while retaining a continuation summary.
- **Why**: Long tool traces eventually crowd out the task or exceed the context budget.
- **How**:
    1. Drop or summarize expendable tool output.
    2. Preserve the goal, constraints, decisions, unresolved failures & next action.
    3. Keep precise evidence in retrievable artifacts rather than only in the summary.
    4. Re-read curr files before acting on an old conclusion.

```{attention} Q&A
:class: dropdown
*Compaction vs persistent memory?*

- Compaction maintains the active task; memory carries selected information into later tasks.
- Neither changes model weights.

*Compaction vs prompt caching?*

- Compaction changes what context is supplied.
- Prompt caching reuses computation for eligible repeated context; it does not itself remove irrelevant content.

*Can a summary be treated as authoritative state?*

- No. It is lossy, model-generated evidence.
- Exact file contents, test results & approval records must remain recoverable elsewhere.
```

&nbsp;

### Persistent Memory
- **What**: Selected knowledge retained across tasks & sessions. {cite:p}`claude_memory,codex_memories,copilot_memory`
- **Why**: A new session should not need to rediscover every stable project fact.
- **How**:
    1. Extract useful facts w/ their scope & supporting evidence.
    2. Store them separately from the active conversation.
    3. Retrieve relevant entries for later work; revalidate against current evidence.
    4. Correct or remove stale entries; keep mandatory rules in maintained instructions.

```{dropdown} Table: Native Memory Surfaces
| Product | Mechanism | Boundary |
|:--|:--|:--|
| Claude Code | Auto memory; `/memory` controls | Learned notes are context, not enforced policy; subagents can have their own memory |
| Local Codex | Opt-in memories; `/memories`; default storage under `~/.codex/memories/` | Separate controls for using memories & contributing future memory inputs |
| Copilot | Copilot Memory, public preview | Repository facts have code citations; user preferences have a separate scope |
```

```{attention} Q&A
:class: dropdown
*Memory vs instructions vs handoff?*

- **Instructions**: maintained rules for applicable work.
- **Memory**: reusable learnings; potentially stale.
- **Handoff**: current task's progress, artifacts & next action.

*Does local Codex memory equal ChatGPT web memory?*

- No. Local Codex has a separate store & controls.
- Local memory is off by default; enable the `memories` feature flag to use it. Generation runs in the background over eligible prior chats, not necessarily after every turn.

*Can memory preserve a prompt injection?*

- Yes. Persisting retrieved instructions can carry the attack into later sessions.
- Keep secrets & untrusted directives out; inspect provenance before promoting a finding to durable knowledge.
```

&nbsp;

### Durable Handoffs
- **What**: External task state sufficient to continue after context loss.
- **Why**: A new/resumed worker needs to distinguish finished work, partial work & unverified claims.
- **How**:
    1. Record:
        - Accepted scope & remaining criteria.
        - Changed artifacts & the revision they belong to.
        - Alive checks, their outcomes & known blockers.
    2. Resume by inspecting those artifacts & checking curr state.

````{note} Example
:class: dropdown
- Suggested handoff artifact; application data, not a vendor-required schema:

```json
{
  "task": "reject-empty-identifier",
  "state": "blocked",
  "changed_paths": ["src/parser.py", "tests/test_parser.py"],
  "verified": ["empty identifier is rejected"],
  "remaining": ["check callers that pass whitespace"],
  "next_action": "inspect whitespace-normalizing callers"
}
```
````

```{attention} Q&A
:class: dropdown
*Why not just restart the original prompt?*

- Repeats discovery, loses decisions & can redo side effects.

*What should become long-term memory?*

- Stable, evidenced project facts; not transient test status or an unfinished task.
- Keep secrets & untrusted instructions out of memory.

*Is an initializer/coding-agent split mandatory?*

- No. Anthropic demonstrates it for long-running work; a small task needs no separate initializer.
- Reuse existing project setup & task tracking before adding new artifacts.
```

&nbsp;

## Tools

### Tool Contracts
- **What**: Explicit action schemas with interpretable results & errors. {cite:p}`anthropic_tools`
- **Why**: Ambiguous tool choice or opaque results waste turns and obscure failure.
- **How**:
    1. Give tools distinct purposes & descriptive parameters.
    2. Validate arguments before performing side effects.
    3. Return actionable evidence: affected paths, IDs, exit status or a specific error.
    4. Bound results; preserve a way to fetch omitted detail.

```{note} Example
:class: dropdown
- Good search result: path, line range, matching excerpt & truncation indicator.
- Good test result: command, revision, exit status & failing cases.
- Bad test result: `"ok"` when the process timed out.
- Bad edit result: a success message w/o confirming the intended file was changed.
```

```{attention} Q&A
:class: dropdown
*Why keep shell access if typed tools exist?*

- Shell composes existing project tools; typed operations expose clearer contracts.
- Pick the smallest adequate surface rather than wrapping every command in a custom service.

*Can an automatic retry duplicate an action?*

- Yes. A timeout may occur after the server committed the operation.
- Check operation status or use an idempotent API before retrying a write.
```

&nbsp;

### MCP
- **Name**: Model Context Protocol {cite:p}`claude_mcp,codex_mcp,copilot_mcp`
- **What**: Client-server protocol for exposing tools, resources & prompts to a host.
- **Why**: External integrations need a reusable interface rather than one bespoke connection per assistant.
- **How**:
    1. Configure a trusted server & its authentication.
    2. Discover the capabilities exposed by that server.
    3. Let the harness mediate calls through its permission system.
    4. Return server results as observations, not higher-priority instructions.

```{attention} Q&A
:class: dropdown
*Does MCP run the agent loop?*

- No. The host harness chooses how discovery, authorization & tool results enter its loop.

*Does connecting a server make it safe?*

- No. Server code, credentials, remote data & returned text are separate trust concerns.
- A server may have permissions outside the local shell sandbox.

*Does a tool description authorize its use?*

- No. Discoverability ≠ authorization.
- Treat external descriptions & output as data that can contain prompt injection.
```

&nbsp;

### Deferred Tool Discovery
- **What**: Load tool definitions on demand rather than advertising every schema up front. {cite:p}`claude_mcp`
- **Why**: Large integration catalogs consume context before any useful action.
- **How**:
    1. Advertise a compact catalog.
    2. Search for capabilities relevant to the task.
    3. Load the exact returned schema before calling the tool.
    4. Keep frequently needed tools directly available when appropriate.

```{attention} Q&A
:class: dropdown
*Where is this concrete rather than hypothetical?*

- Claude Code documents MCP tool search with deferred schemas.
- Availability depends on model/provider support; some configurations load schemas up front.

*What is the failure mode?*

- Guessing a hidden tool's name or arguments instead of discovering its schema.
- Poor catalog descriptions can also make an existing capability effectively undiscoverable.
```

&nbsp;

### Permissions & Sandboxing
- **What**: Authorization decisions plus execution-level restrictions. {cite:p}`claude_permissions,claude_sandbox,codex_permissions,copilot_permissions`
- **Why**: An agent can propose harmful actions even when its assigned task is benign.
- **How**:
    1. Restrict available tools to the task.
    2. Decide which actions need approval.
    3. Constrain filesystem & network access where supported.
    4. Apply service-side authorization to remote tools.
    5. Keep high-impact actions behind an independent approval boundary.

```{dropdown} Table: Different Boundaries
| Mechanism | Answers | Does not imply |
|:--|:--|:--|
| Tool restriction | Can this agent invoke this capability? | Every allowed argument is safe |
| Permission rule | May this proposed action run w/o asking? | The resulting process is isolated |
| Sandbox | What can the process actually access? | External MCP servers share the same boundary |
| Git worktree | Which checkout receives edits? | Network, credentials or database isolation |
| Human approval | Has this action been authorized? | All future actions are authorized |
```

```{attention} Q&A
:class: dropdown
*Does unattended mode mean unrestricted mode?*

- No. A task can continue autonomously inside narrow permissions.
- Do not solve a blocked tool call by automatically disabling the boundary.

*Is a shell-command regex a sandbox?*

- No. Quoting, indirection, subprocesses & alternate tools defeat simplistic text matching.
- Use OS/service restrictions for hard boundaries; hooks can enforce narrower workflow rules.

*Are hooks contained by the agent's shell sandbox?*

- Not necessarily. Claude Code documents hooks & MCP servers as code that can run outside that sandbox.
- Review repo-supplied configuration before trusting or executing it.

*What curr product boundaries are easy to miss?*

- Codex permission profiles are beta; selecting legacy `sandbox_mode` uses the legacy configuration instead of composing with `default_permissions`.
- Codex profile domain rules need the network proxy enabled to restrict direct network access.
- Copilot CLI local sandboxing is preview/experimental & disabled by default; an allowlist alone is not an enabled sandbox.
```

&nbsp;

### Prompt Injection Boundaries
- **What**: Separation of untrusted content from authorized instructions & actions. {cite:p}`claude_permissions,claude_sandbox,codex_approval_security`
- **Why**: Files, web pages & tool results can contain instructions unrelated to the user's task.
- **How**:
    1. Treat retrieved text as evidence, not permission to expand the task.
    2. Preserve its origin when passing it to another worker or memory.
    3. Restrict capabilities, readable secrets & outbound destinations independently of the prompt.
    4. Require separate approval for sensitive actions; inspect the actual proposed operation.

```{note} Example
:class: dropdown
- A retrieved issue asks the agent to upload local credentials as a “diagnostic.”
- That text is task data, not user authorization; neither a summary nor a subagent handoff should promote it into an instruction.
```

```{attention} Q&A
:class: dropdown
*Does labeling text as untrusted solve the problem?*

- No. It helps interpretation but is not an execution boundary.
- Restrict what a mistaken decision can access or change.

*Can automatic monitoring replace pre-execution controls?*

- No. Codex documents model-dependent safety monitoring that may pause or end a task after the triggering activity.
- Monitoring, approval review & sandboxing act at different points; one does not substitute for the others.
```

&nbsp;

## Loop
- **What**: Design of an agent's repeated decision → action → feedback cycle. {cite:p}`claude_agent_sdk_blog,claude_cli`
- **Why**: Multi-step work needs feedback; uncontrolled repetition can stall or run forever.
- **How**:
    1. Define what each iteration receives: goal, relevant context, observations & progress.
    2. Let the model propose actions; execute permitted tools & feed results into the next decision.
    3. Define when to continue, wait, retry, stop or escalate using completion evidence & resource limits.

&nbsp;

### Planning & Approval
- **What**: Separation of investigation, proposed action & authorized execution. {cite:p}`claude_permissions,codex_approval_security,copilot_cli_reference`
- **Why**: A useful plan is not permission to carry out every action it describes.
- **How**:
    1. Investigate under restricted capabilities.
    2. Propose affected paths, steps, risks & acceptance checks.
    3. Obtain approval for the intended scope & necessary capabilities.
    4. Execute; pause again if the scope or required authority changes.

````{note} Example
:class: dropdown
- Native starting points for planning or read-only investigation:

```bash
claude --permission-mode plan
codex --sandbox read-only "Plan the parser change; do not implement it."
copilot --mode=plan
```

- These are separate product controls, not interchangeable security guarantees.
````

```{attention} Q&A
:class: dropdown
*Plan approval vs tool approval?*

- Plan approval agrees on the approach; tool approval authorizes a particular capability or operation.
- Neither silently authorizes a later deployment, broader data access or a changed task.

*Is plan mode a hard sandbox?*

- A plan-and-approve workflow is not, by itself, filesystem or network isolation.
- Shell commands & external tools still need their own access controls.
- Inspect effective permissions rather than relying only on a mode name.

*What happens when unattended work needs new authority?*

- Fail or return a blocker unless a trusted approval path exists.
- Lack of an available human is not permission to bypass the boundary.
```

&nbsp;

### Completion Contract
- **What**: Observable conditions separating completion from a plausible final answer. {cite:p}`claude_best,anthropic_agent_evals`
- **Why**: An agent can stop after a partial fix or report success w/o exercising the changed behavior.
- **How**:
    1. Specify scope, preserved behavior & required outputs.
    2. Choose checks that distinguish the requested change from the old behavior.
    3. Run them against the final candidate, not an earlier revision.
    4. Report completion only when the required evidence exists.

```{note} Example
:class: dropdown
- Request: reject empty identifiers w/o changing valid identifiers.
- Contract: `""` rejected; `"alpha"` still accepted; public API unchanged; existing suite passes; no unrelated edits.
- Insufficient: parser file changed; model says “fixed”; unrelated tests pass.
- A zero exit code from the agent process establishes successful process execution, not satisfaction of this contract.
```

```{attention} Q&A
:class: dropdown
*Why give tests or screenshots in the prompt?*

- They make the desired outcome observable.
- For a UI change, unit tests alone may not exercise the interaction that actually matters.

*Can the implementer also define success?*

- It can propose checks; it should not silently weaken accepted requirements to make them pass.
- Keep the acceptance contract and critical check configuration outside its write permissions when adversarial robustness matters.
```

&nbsp;

(bounded-repair)=
### Bounded Repair
- **What**: Verification-driven retries with explicit exhaustion & failure outcomes. {cite:p}`claude_cli,codex_exec,copilot_modes`
- **Why**: Unbounded “keep trying” can repeat a failing strategy or consume resources indefinitely.
- **How**:
    1. Run the acceptance check.
    2. On a repairable failure, return a concise failure report to the agent.
    3. Allow a bounded repair attempt, then check the new candidate.
    4. Stop successfully on evidence; otherwise stop as blocked or failed.

````{important} Code
:class: dropdown
- Minimal outer controller around an existing native harness; it does not reimplement model/tool dispatch.
- The check must fail when no required tests run. Both commands are trusted, preconfigured argument lists.

```python
import subprocess
import sys


class BoundedRepair:
    def __init__(self, agent_prefix, check_command, max_repairs=2, timeout=120):
        if not agent_prefix or not check_command:
            raise ValueError("Both command lists are required")
        if max_repairs < 0 or timeout <= 0:
            raise ValueError("Invalid repair budget")
        self.agent_prefix = list(agent_prefix)
        self.check_command = list(check_command)
        self.max_repairs = max_repairs
        self.timeout = timeout

    def run(self):
        for attempt in range(self.max_repairs + 1):
            result = subprocess.run(
                self.check_command,
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode == 0:
                return "verified"
            evidence = (result.stdout + "\n" + result.stderr)[-4000:]
            if attempt == self.max_repairs:
                raise RuntimeError(f"Repair budget exhausted:\n{evidence}")
            ## Evidence is untrusted task data, not permission to expand scope.
            prompt = (
                "Repair the accepted task within its existing scope. "
                "Do not weaken or delete the checks. Diagnostic data follows:\n"
                + evidence
            )
            subprocess.run(
                self.agent_prefix + [prompt],
                check=True, timeout=self.timeout,
            )


## Example: already-passing check; no model call or file edit.
controller = BoundedRepair(
    agent_prefix=["claude", "-p", "--max-turns", "6"],
    check_command=[sys.executable, "-c", "assert 2 + 2 == 4"],
)
assert controller.run() == "verified"
```
````

```{attention} Q&A
:class: dropdown
*What does the example bound?*

- Repair invocations & each subprocess wait; example numbers are chosen budgets, not vendor defaults.
- Native agent calls can contain many tool calls. Configure their own limits & permissions too.
- A subprocess timeout is not a distributed cancellation protocol; detached descendants or remote jobs need explicit lifecycle management.

*Which failures should not trigger another model attempt?*

- Missing credentials, denied authorization, exhausted budget or a broken runtime.
- The code propagates agent-process failures & timeouts instead of presenting them as successful repairs.

*Why not keep retrying identical failures?*

- No changed evidence or strategy → no reason to expect progress.
- Stop & expose the blocker; retry transient infrastructure errors separately from semantic repair.
```

&nbsp;

### Resource Budgets & Model Selection
- **What**: Allocation of models, reasoning effort & execution limits across a task. {cite:p}`claude_cli,claude_subagents,codex_agents,copilot_cli_reference`
- **Why**: More reasoning, retries or workers can increase cost without improving the outcome.
- **How**:
    1. Match each role's model & effort to its uncertainty and evidence requirements.
    2. Set independent limits for turns, retries, concurrent workers & elapsed time.
    3. Track actual usage across the parent and workers.
    4. Escalate a difficult task deliberately; terminate when the accepted budget is exhausted.

```{dropdown} Table: Non-Interchangeable Limits
| Limit | Bounds | Does not necessarily bound |
|:--|:--|:--|
| Turns/continuations | Repeated model turns | Tool calls or duration within a turn |
| Concurrency | Simultaneously active workers | Total workers created over the run |
| Total invocations | Overall fan-out or repair attempts | Cost of an individual invocation |
| Timeouts/deadlines | Local waiting or execution interval | Remote side effects & detached descendants |
| Cost/credit limit | Accounted usage under the product's policy | Exact spend when enforcement is a soft threshold |
```

```{attention} Q&A
:class: dropdown
*Must every worker use the coordinator's model?*

- No. Claude, Codex & Copilot expose model selection for custom agents; supported effort controls depend on the model & surface.
- Use an inexpensive worker only when it can meet the role's acceptance contract.

*Why not maximize reasoning effort everywhere?*

- It can increase latency & usage; simple retrieval may not need deeper reasoning.
- Decide from representative task outcomes, not a belief that more tokens always help.
```

&nbsp;

### Goals, Continuations & Schedules
- **What**: Distinct triggers for starting the next agent turn. {cite:p}`claude_goal,claude_schedules,copilot_modes,copilot_cli_reference,codex_automations`
- **Why**: “Continue until done” and “check again later” require different control.
- **How**:
    1. **Completion-driven**: continue after an unsatisfied goal or stop gate.
    2. **Time-driven**: wake at a scheduled interval.
    3. **Event-driven**: wake when a background result or external event arrives.
    4. Cancel continuing work when the task completes or becomes impossible.

```{dropdown} Table: Native Continuation Surfaces
| Surface | Concrete mechanism | Boundary |
|:--|:--|:--|
| Claude Code | `/goal <condition>` | Model-evaluated completion condition; does not change permissions |
| Claude Code | `/loop 5m <prompt>` | Session scheduling; not a permanent cloud daemon |
| Claude Code | `Stop` hook | Custom continuation rule after a turn |
| Copilot CLI | `/autopilot` | Continues toward a goal rather than awaiting each user prompt |
| Copilot CLI | `/every` | Experimental scheduled prompts; separate from autopilot |
| ChatGPT desktop/web | Scheduled tasks | Desktop can use local Codex projects; web cannot directly access local folders |
```

```{attention} Q&A
:class: dropdown
*Does a goal evaluator prove correctness?*

- No. Claude's `/goal` uses a separate model judgment; give it concrete evidence and keep deterministic acceptance checks.

*Should a waiting task repeatedly say “continue”?*

- No. Await its completion event or schedule an appropriately spaced check.
- Busy polling spends turns w/o changing the information available.

*Does a saved schedule run while the machine is off?*

- Only if its execution host supports that.
- Claude session loops require a running session; cloud routines are a different surface.
- Distinguish persisted schedule configuration from a live executor.
- Codex CLI & IDE can prepare a task but do not supply the Scheduled management interface.
```

&nbsp;

### External Events & Hosted Runs
- **What**: Inbound events that wake an existing session or start a separate agent job. {cite:p}`claude_channels,claude_routines,codex_automations,copilot_cloud,copilot_cli_about`
- **Why**: CI results, repository events & messages may arrive after the initiating user turn.
- **How**:
    1. Authenticate the event source & restrict which events may trigger work.
    2. Route to an existing session or a new isolated run.
    3. Pass the event as scoped task input, not unrestricted authority.
    4. Track the run & its result; avoid duplicating side effects if delivery repeats.

```{dropdown} Table: Event Ingress vs Execution Host
| Mechanism | Destination | Boundary |
|:--|:--|:--|
| Claude channels, research preview | Existing running session, via MCP | Session must remain open; authenticate & restrict senders |
| Claude routines, research preview | Cloud or configured self-hosted run | Saved prompt, repos, connectors & triggers; can run while the laptop is closed |
| ChatGPT scheduled/event-triggered tasks | Time schedules: desktop/web; event triggers: web/mobile | Event triggers require an eligible plan; unavailable in desktop, Codex CLI & IDE |
| Copilot cloud agent | Separate hosted repository task | Environment & available configuration differ from the local CLI |
| Copilot CLI cloud sandbox, public preview | Cloud-hosted CLI session via `copilot --cloud` | Distinct from a delegated cloud-agent job; inherits cloud-agent policies |
```

```{attention} Q&A
:class: dropdown
*Channel vs hook?*

- A channel brings an external event into a session.
- A hook reacts to an event inside the harness lifecycle.

*Remote control vs cloud execution?*

- Remote control changes where the user interacts; it need not move execution away from the original host.
- A hosted job runs in its configured remote environment; local files & credentials do not automatically follow it.
```

&nbsp;

## Graph
- **What**: Design of task decomposition, dependencies & result routing. {cite:p}`claude_workflows,copilot_fleet`
- **Why**: Multi-stage work needs coordination across tasks, not just repetition within one task.
- **How**:
    1. Define nodes as bounded tasks, tool operations or agent loops, each w/ explicit inputs & outputs.
    2. Connect nodes through prerequisites, artifact handoffs & conditional success/failure routes.
    3. Schedule independent branches concurrently; join required results before dependent work proceeds.

&nbsp;

### Task Graph
- **What**: Explicit dependencies & conditional transitions between units of work. {cite:p}`claude_workflows,copilot_fleet`
- **Why**: A flat task list cannot express “review this exact implementation before integration.”
- **How**:
    1. Make each node produce a concrete artifact or decision.
    2. Add edges only for real input dependencies.
    3. Route verification failure back to repair.
    4. Route missing authorization or exhausted budget to a blocked outcome.
    5. Keep completion contingent on every required node.

```{note} Math
:class: dropdown

$$
\mathcal{G}=(\mathcal{V},\mathcal{E})
$$

- A node is ready when it is pending & all required predecessors are complete:

$$
\operatorname{ready}(v)
\iff
\operatorname{pending}(v)
\land
\bigwedge_{(u,v)\in\mathcal{E}}\operatorname{complete}(u)
$$

- $u$: Predecessor node.
- Dependency edges can form a directed acyclic graph within an attempt.
- Repair feedback makes the overall control flow cyclic; do not call that whole flow a DAG.
```

````{note} Example
:class: dropdown
```text
inspect -> agree contract -> implement -> verify -> review -> deliver
                               ^           |         |
                               +-- fail ---+         |
                               +-- confirmed issue --+

denial / exhausted budget / missing prerequisite -> blocked
```

- Review receives the verified candidate's diff.
- A review-driven edit invalidates affected checks → re-verify before delivery.
- This diagram specifies a controller design, not a shared graph configuration language for the three products.
````

```{attention} Q&A
:class: dropdown
*Does “plan first, then test” in a prompt enforce a graph?*

- No. It is guidance until a scheduler or gate enforces prerequisites.
- A skill can describe the plan; code or native runtime controls enforce the transitions.

*What should an edge carry?*

- Artifact ID/path, revision, relevant findings & outcome.
- Not the entire predecessor convo unless the next node actually needs it.
```

&nbsp;

### Subagents & Handoff Contracts
- **What**: Delegated workers with separate contexts & explicit task boundaries. {cite:p}`claude_subagents,codex_agents,copilot_agents`
- **Why**: Independent investigations can overwhelm the coordinator's context or require different capabilities.
- **How**:
    1. Delegate one self-contained task.
    2. Supply allowed files, inputs, constraints & expected output.
    3. Restrict tools & select an appropriate model.
    4. Receive evidence & unresolved questions, not only a verdict.
    5. Reuse the existing worker for follow-up when its context remains useful.

````{note} Example
:class: dropdown
```text
Role: parser compatibility reviewer.
Inputs: candidate diff and the parse() contract below.
Allowed files: src/parser.py, tests/test_parser.py.
Task: find behavior changes outside empty-identifier rejection.
Restrictions: read-only; no edits, subprocesses, or external requests.
Return: confirmed issue, relevant lines, counterexample, proposed fix.
If no issue is found, say so. Do not invent one to justify the role.
```
````

```{attention} Q&A
:class: dropdown
*Does the worker know the coordinator's whole convo?*

- Do not assume so. Claude supports fresh workers & forks; other surfaces have their own context propagation.
- Explicit handoff inputs work even when convo inheritance changes.

*Is a role prompt enough to make a worker read-only?*

- No. Remove write-capable tools or use an appropriate sandbox.
- A shell tool can write files even when an edit tool is absent.

*When should work stay in the main agent?*

- Small tasks, one continuous dependency chain or work needing constant shared context.
- Delegation adds startup, token & integration costs.
```

&nbsp;

### Background Task Lifecycle
- **What**: Launch, observation, steering, completion & cancellation of asynchronous work. {cite:p}`claude_subagents,codex_agents,codex_app_server,copilot_cli_reference`
- **Why**: A worker that has started is not a result the coordinator can safely consume.
- **How**:
    1. Launch a bounded task; retain its task/thread identifier.
    2. Continue independent work while it runs.
    3. Await completion events or use bounded status checks where events are unavailable.
    4. Consume the terminal result & associated artifacts before advancing dependencies.
    5. On cancellation, stop the specific work, reconcile partial effects & update dependent tasks.

```{note} Example
:class: dropdown
- Launch caller analysis → inspect parser independently → receive analysis result → implement.
- Not: launch analysis → immediately assume its findings, or repeatedly poll while doing nothing else.
- Claude & Copilot expose task views; Codex exposes agent threads through `/agent` & structured turn controls through app-server.
```

```{attention} Q&A
:class: dropdown
*Steer, resume or restart?*

- **Steer**: add input to active work when supported.
- **Resume**: continue stored context.
- **Restart**: create a new attempt; check for prior side effects before repeating actions.

*Does interrupting the model undo its tools?*

- No. Files may already be edited; a remote operation may already have committed.
- Cancellation must account for the actual worker/process/service lifecycle.

*Should a failed optional branch block everything?*

- Only if its output is required by the contract.
- Distinguish required dependencies from optional evidence; propagate failures instead of silently treating them as empty results.
```

&nbsp;

### Parallel Work & Isolation
- **What**: Concurr independent tasks with controlled ownership of mutable state. {cite:p}`claude_worktrees,codex_worktrees,copilot_fleet`
- **Why**: Parallelism helps independent work; competing edits can erase that gain.
- **How**:
    1. Parallelize independent reads freely within the budget.
    2. Assign disjoint write ownership or separate worktrees.
    3. Wait for required results before consuming them.
    4. Integrate through one owner; check the combined result.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $d_v$: Execution duration of node $v$.
    - $p$: Available worker count.

- Ignoring startup, coordination & contention, execution time obeys:

$$
T \geq
\max\left(
\frac{\sum_{v\in\mathcal{V}}d_v}{p},
\max_{\rho}\sum_{v\in\rho}d_v
\right)
$$

- $T$: Graph makespan.
- $\rho$: Dependency path in an acyclic execution attempt.
- More workers cannot shorten the critical dependency path.
```

```{attention} Q&A
:class: dropdown
*Separate context vs separate worktree?*

- Separate context isolates model-visible history.
- Separate worktree isolates checkout/index state for edits.
- Neither automatically isolates ports, databases, credentials or remote services.

*Why not give every worker the same files?*

- Concurr writes create conflicts & ambiguous ownership.
- Parallel read-only critiques of the same candidate can be intentional; duplicate implementation usually is not.

*Do passing worker tests establish integration success?*

- No. Tests in separate candidates do not establish correctness of the merged candidate.
```

&nbsp;

### Scripted Orchestration & Teams
- **What**: Code-controlled routing or coordinated peer sessions above individual agent loops. {cite:p}`claude_workflows,claude_teams,copilot_fleet,copilot_sdk_fleet,codex_agents`
- **Why**: Repeated branching & fan-out can outgrow a coordinator's convo.
- **How**:
    1. Start with one agent & a few focused workers.
    2. Use peers when they need to communicate and coordinate.
    3. Move routing into code when repeatability & explicit dependencies matter.
    4. Bound concurrency, total work & runtime separately.

```{dropdown} Table: Who Holds the Plan?
| Mechanism | Next-step authority | Appropriate use |
|:--|:--|:--|
| Skill | Model following reusable instructions | Repeatable procedure |
| Coordinator + subagents | Coordinator model | A few delegated tasks |
| Claude agent team | Lead & communicating teammates | Work needing peer coordination |
| Claude dynamic workflow | JavaScript orchestration script | Explicit loops, branches & large fan-out |
| Copilot fleet | CLI planning/delegation machinery | Parallel subagent work |
| Native harness under an external controller | Application code | Cross-product routing or deterministic release gates |
```

```{attention} Q&A
:class: dropdown
*Are Claude teams the same as dynamic workflows?*

- No. Teams coordinate model-led sessions; dynamic workflows put orchestration in a script.
- Teams are documented as experimental & require interactive sessions; do not assume they work in `claude -p` or Agent SDK runs.

*Is Copilot SDK fleet the same stability contract as CLI `/fleet`?*

- No. SDK `session.rpc.fleet.start()` is experimental; pin the SDK & CLI runtime together.
- A TUI feature name does not establish a stable programmatic API.

*Can I paste a generic graph YAML into all three products?*

- No. Shared concepts do not imply compatible configuration or scheduler APIs.
- Use native features where sufficient; an external controller owns any cross-product graph.

*Does more agents mean better results?*

- Not necessarily. More workers add correlated mistakes, cost & coordination.
- A new role needs a distinct evidence-gathering or execution responsibility.
```

&nbsp;

## Hooks
- **What**: Design of event-triggered callbacks around agent execution. {cite:p}`claude_hooks_guide,codex_hooks,copilot_hooks`
- **Why**: Required checks & automation should not depend on the model remembering to request them.
- **How**:
    1. Bind callbacks to lifecycle events: before a tool, after a result or when the agent tries to stop.
    2. Process the event payload & return a supported decision or observation; the runtime invokes the callback, not the model.
    3. Define timeout/failure behavior & guard against repeated continuations; keep hard restrictions in permissions or sandboxing.

&nbsp;

### Lifecycle Hooks
- **What**: Runtime callbacks at named execution events. {cite:p}`claude_hooks_guide,codex_hooks,copilot_hooks`
- **Why**: Required behavior should not depend on the model remembering to request it.
- **How**:
    1. Select the event that occurs before or after the relevant action.
    2. Register a handler using that product's configuration.
    3. Parse the event's structured input.
    4. Return the event-specific decision or observation.

```{dropdown} Table: Where a Hook Belongs
| Intent | Lifecycle point | Key limitation |
|:--|:--|:--|
| Supply session context | Start/resume | Load only relevant, trusted information |
| Deny an action | Before tool execution | A post-tool callback is already too late |
| Observe an edit or command | After tool execution | Completion of the tool ≠ completion of the task |
| Require more work | Agent's attempted stop | Must terminate on failure or exhaustion |
| Record teardown | Session end | Not generally a way to veto task completion |
```

```{dropdown} Table: Native Hook Dialects
| Detail | Claude Code | Codex | Copilot CLI |
|:--|:--|:--|:--|
| Project configuration | `.claude/settings.json` | `.codex/hooks.json` or inline `.codex/config.toml` | `.github/hooks/*.json` |
| Before tool | `PreToolUse` | `PreToolUse` | `preToolUse` |
| After tool | `PostToolUse` | `PostToolUse` | `postToolUse` |
| Attempted stop | `Stop` | `Stop` | `agentStop` |
| Command field | `command` | `command` | `bash`, `powershell` or `command` |
| Timeout field | `timeout` | `timeout` | `timeoutSec` |
| Continuation guard input | `stop_hook_active` | `stop_hook_active` | `stop_hook_active` |
| Trust caveat | Review discovered project code | New/changed non-managed definitions need hook trust review | Prompt-mode repo loading has a separate trust gate |
```

```{attention} Q&A
:class: dropdown
*Is every hook deterministic?*

- Triggering a configured handler is runtime-controlled; its decision need not be deterministic.
- Claude supports command, HTTP, MCP-tool, prompt & agent handlers. Prompt/agent handlers introduce model judgment.
- Use machine checks for exact invariants; use model checks only for genuinely semantic judgments.

*Are the schemas portable?*

- No. Event casing, nesting, arguments & output fields differ.
- Native CLI hooks & SDK callbacks may also use different names within the same product.
```

&nbsp;

### Pre-Execution Decisions
- **What**: Event-specific allow, deny or approval decisions before a tool runs. {cite:p}`claude_hooks,codex_hooks,copilot_hooks`
- **Why**: Some workflow rules need inspection of the proposed action before side effects occur.
- **How**:
    1. Match the intended tool/event.
    2. Parse arguments as data, not shell text to execute.
    3. Return a documented decision & a useful reason.
    4. Leave normal permissions in force when the hook has no decision.

````{note} Example
:class: dropdown
- A Claude `PreToolUse` handler's denial response, printed as JSON to stdout with exit code `0`:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "This action requires the approved change window."
  }
}
```

- Codex supports the same denial shape; its curr `PreToolUse` does not support `"ask"` and does not cover hosted tools such as web search.
- Copilot CLI uses top-level decision fields instead:

```json
{
  "permissionDecision": "deny",
  "permissionDecisionReason": "This action requires the approved change window."
}
```

- The window decision must come from trusted policy data, not a model-generated `"approved": true`.
- An empty successful response means no objection; it is not automatic authorization.
````

```{attention} Q&A
:class: dropdown
*Why not implement all access control here?*

- Some hook failures are non-blocking; timeout & malformed-output semantics differ by product and event.
- Keep hard restrictions in native permissions, sandboxing & service authorization.
- Copilot `preToolUse` timeouts fail open; Codex unsupported `PreToolUse` fields can fail the hook and let the tool call proceed.

*Can a post-tool denial undo execution?*

- No. It may affect the next step, not erase the side effect.

*Should a hook rewrite arbitrary shell commands?*

- Not with ad hoc string substitutions.
- Prefer a typed tool or a reviewed command wrapper when arguments need policy-aware transformation.
```

&nbsp;

(completion-hooks)=
### Completion Hooks
- **What**: Stop-time checks that request another turn or terminate with a blocker. {cite:p}`claude_hooks,codex_hooks,copilot_hooks`
- **Why**: A proposed final response can precede required verification.
- **How**:
    1. Run the accepted check against the curr candidate.
    2. Pass → allow the turn to end.
    3. Fail with repair budget remaining → feed back the failure.
    4. Fail after exhaustion → terminate as incomplete, not successful.

````{note} Example
:class: dropdown
- Claude configuration in `.claude/settings.json`; the handler below is saved as `.claude/hooks/stop_gate.py`.
- The project supplies `scripts/check_parser.py`: a trusted acceptance command that exits nonzero for failed or missing required checks.
- Invoke from the project root; `CLAUDE_PROJECT_DIR` anchors the handler path.

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/stop_gate.py\" --hook",
            "timeout": 75
          }
        ]
      }
    ]
  }
}
```
````

````{important} Code
:class: dropdown
```python
import json
from pathlib import Path
import subprocess
import sys


class StopGate:
    def __init__(self, check_command, timeout=60):
        if not check_command or timeout <= 0:
            raise ValueError("A check command and positive timeout are required")
        self.check_command = list(check_command)
        self.timeout = timeout

    def __call__(self, event):
        if not isinstance(event, dict) or event.get("hook_event_name") != "Stop":
            raise ValueError("Expected a Stop event")
        active = event.get("stop_hook_active")
        if type(active) is not bool:
            raise ValueError("Expected boolean stop_hook_active")
        result = subprocess.run(
            self.check_command, capture_output=True,
            text=True, timeout=self.timeout,
        )
        if result.returncode == 0:
            return {}
        reason = "Acceptance check failed:\n" + (
            result.stdout + "\n" + result.stderr
        )[-4000:]
        if active:
            ## Ending a failed task is not accepting its candidate.
            return {"continue": False, "stopReason": "Incomplete. " + reason}
        return {"decision": "block", "reason": reason}


## Example: one failed check requests work; a repeated failure ends as incomplete.
if __name__ == "__main__":
    if sys.argv[1:] == ["--hook"]:
        root = Path(__file__).resolve().parents[2]
        gate = StopGate([sys.executable, str(root / "scripts/check_parser.py")])
        try:
            output = gate(json.load(sys.stdin))
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            output = {"continue": False, "stopReason": f"Gate failed: {error}"}
        print(json.dumps(output))
    else:
        gate = StopGate([sys.executable, "-c", "raise SystemExit(1)"])
        event = {"hook_event_name": "Stop", "stop_hook_active": False}
        assert gate(event)["decision"] == "block"
        event["stop_hook_active"] = True
        assert gate(event)["continue"] is False
```
````

```{attention} Q&A
:class: dropdown
*Why inspect `stop_hook_active`?*

- Blocking a stop creates another opportunity to stop.
- This example permits one hook-driven repair continuation; an already-active continuation that still fails ends explicitly as incomplete.
- With several stop hooks, the flag describes the continuation context, not this script's private retry counter.

*Is this a fail-closed release gate?*

- No. This handler reports its own expected failures, but failure to launch it or a runtime hook timeout can follow non-blocking product semantics.
- A separate trusted check must authorize merge, publication or deployment.

*Why not just return an empty object after the retry?*

- That silently removes the gate while the check still fails.
- Terminate with a visible blocker instead.

*What if background work is still running?*

- Do not declare completion from an intermediate checkout.
- Join required workers before acceptance; stop hooks can also inspect product-provided background-task state where available.
```

&nbsp;

### Hook Reliability
- **What**: Correct execution, failure handling & observation of the callback itself. {cite:p}`claude_hooks_guide,codex_hooks,copilot_hooks`
- **Why**: A configured hook can be absent, mismatched, timed out or ignored because its output is malformed.
- **How**:
    1. Confirm discovery in the actual CLI/IDE/cloud surface.
    2. Exercise the exact event with a harmless fixture.
    3. Keep structured stdout separate from diagnostic stderr.
    4. Test pass, deny/block, malformed input, missing executable & timeout.
    5. Record whether the runtime honored the decision, not merely whether the script ran.

```{attention} Q&A
:class: dropdown
*Synchronous vs asynchronous?*

- **Synchronous gate**: the runtime waits before deciding the next action.
- **Asynchronous observer**: useful for logs or background feedback; cannot retroactively veto the action.
- Claude's `async: true` command hooks do not honor blocking decision fields.
- Codex asynchronous hooks likewise cannot block, authorize, rewrite or force continuation.

*Can hooks run concurrly?*

- Product/event rules differ; Claude can run matching handlers in parallel.
- Do not rely on registration order for shared-file updates. Combine dependent operations into one handler or use explicit synchronization.

*Why avoid running a full suite after every edit?*

- Overlapping runs can inspect different intermediate states & flood the agent with obsolete failures.
- Use targeted feedback during editing, then a fresh acceptance check after the candidate stabilizes.

*Are hook scripts untrusted input?*

- They are executable code with real privileges.
- Review scripts & configuration changes; do not interpolate event-provided strings into shell commands.

*Does every worker emit the same lifecycle hooks?*

- No. Copilot's built-in `general-purpose` worker does not emit `subagentStart`/`subagentStop`; its custom agents & other documented built-in YAML agents do.
- Test the actual worker type rather than inferring hook coverage from the task UI.
```

&nbsp;

## Running & Evaluating the System

### Programmatic Harnesses
- **What**: Native agent runtimes exposed through commands, event streams or SDKs. {cite:p}`claude_headless,codex_exec,codex_sdk,codex_app_server,copilot_sdk`
- **Why**: Automation needs machine-readable outcomes & lifecycle control rather than terminal-text scraping.
- **How**:
    1. Use the native noninteractive CLI for a single bounded job.
    2. Use an SDK or app-server interface for persistent sessions & interactive event handling.
    3. Handle approvals, failures, cancellation & terminal outcomes explicitly.
    4. Validate structured output, then independently verify its claims.

```{dropdown} Table: Integration Boundaries
| Product | Programmatic surface | Who owns the inner loop? |
|:--|:--|:--|
| Claude Code | `claude -p`; Claude Agent SDK | Claude Code runtime |
| Codex | `codex exec`; Codex SDK; app server | Codex runtime |
| Copilot | Programmatic CLI; GitHub Copilot SDK | Copilot CLI engine |
| Direct model API | Model calls & tool-call messages | Your application, unless another runtime supplies it |
```

```{attention} Q&A
:class: dropdown
*Codex SDK vs OpenAI Agents SDK?*

- Codex SDK embeds the Codex agent runtime.
- OpenAI Agents SDK is a different orchestration library; using it is not automatically using Codex's coding harness.

*Does JSON output guarantee a correct result?*

- No. A valid schema establishes shape, not factual or behavioral correctness.
- Treat `"tests_passed": true` as a claim unless tied to trusted test evidence.

*Why not parse whatever the CLI prints?*

- Human-readable rendering can mix progress, diagnostics & answers.
- Use the supported structured stream; distinguish partial output from terminal success.
- Codex `--json` & Copilot `--output-format=json` emit JSON Lines, not one JSON document.

*What changes in noninteractive mode?*

- Human prompts may be unavailable; configure required permissions in advance w/o disabling all restrictions.
- Initialization & trust behavior can also differ. Claude `-p` auto-discovers project extensions unless configured otherwise; `--bare` skips most discovery and has different authentication requirements.
```

&nbsp;

### Session & Event Protocols
- **What**: Structured lifecycle contracts between an application & a persistent agent runtime. {cite:p}`claude_headless,codex_app_server,copilot_sdk_sessions`
- **Why**: Starting a turn, receiving text & completing a task are different events.
- **How**:
    1. Initialize the client connection; create or resume an identified session.
    2. Register tools, permission handling & event consumers before starting work.
    3. Correlate requests, turns, tool calls & results by their identifiers.
    4. Distinguish partial output from terminal success, failure or interruption.
    5. Disconnect, resume or delete state deliberately; restore external dependencies separately.

````{note} Example
:class: dropdown
- Codex app-server lifecycle, using actual protocol method names:

```text
initialize -> response -> initialized
thread/start or thread/resume -> thread ID
turn/start -> turn/started -> item events -> turn/completed
                         \-> turn/steer or turn/interrupt
```

- `turn/completed` carries final status; its arrival does not by itself mean success.
- A failed tool item can be repaired later in the same turn; a successful item does not establish the whole turn's outcome.
- Consume returned identifiers; do not invent thread IDs or parse them out of rendered prose.
````

```{attention} Q&A
:class: dropdown
*Does reopening a session restore every dependency?*

- No. Copilot documents that provider credentials & in-memory tool state are not persisted.
- Re-register required callbacks/tools & restore credentials through the supported configuration.

*Can two clients safely write to the same session?*

- Do not assume so. Copilot SDK does not provide a built-in lock for concurrent resumes.
- Use one coordinating owner or an application-level queue; forks are separate histories, not automatic merge operations.

*Can a JSON-RPC server be exposed like an ordinary local CLI?*

- No. Its network transport needs authentication & protected access.
- Codex WebSocket transport is experimental/unsupported; local stdio avoids exposing a network listener.
```

&nbsp;

### Checkpoints & Recovery
- **What**: Saved convo or workspace state for controlled continuation & rollback. {cite:p}`claude_checkpoint,codex_cli_features,copilot_context`
- **Why**: A failed approach should not require reconstructing all prior work.
- **How**:
    1. Save a meaningful task boundary.
    2. Preserve the associated workspace revision & evidence.
    3. Resume reasoning or restore files using the appropriate mechanism.
    4. Reconcile external side effects before retrying.

```{attention} Q&A
:class: dropdown
*Does restoring a convo restore the world?*

- No. Conversation state, files, background processes & remote services have different lifecycles.

*Can Claude rewind any filesystem change?*

- No. Its checkpointing tracks supported direct file edits, not arbitrary shell modifications.
- Subagent restoration also depends on execution mode; do not assume every delegated edit is covered.

*Are Copilot compaction checkpoints rewind points?*

- No. `/session checkpoints` exposes summaries, not filesystem rollback.
- `/rewind` operates on convo/file rewind points; `/fork` forks history w/o creating a worktree.

*What about a pushed branch, email or database write?*

- Local checkpoints cannot undo remote effects.
- Use the service's cancellation/compensation mechanism or require approval before the original action.
```

&nbsp;

### Observability
- **What**: Correlated records of decisions, actions, outcomes & resource use. {cite:p}`claude_monitor,codex_exec,copilot_cli_reference`
- **Why**: A final answer cannot reveal where the harness lost context, stalled or bypassed an intended gate.
- **How**:
    1. Correlate session, worker, tool-call & artifact identifiers.
    2. Record arguments safely, outcome, duration, retries & permission decisions.
    3. Preserve the candidate revision associated with each check.
    4. Inspect the failed transition before changing prompts or models.

```{dropdown} Table: Symptom to Evidence
| Symptom | Inspect first |
|:--|:--|
| Wrong tool repeatedly chosen | Available schemas, descriptions & actual discovery results |
| Agent stops early | Stop event, continuation decision & uncompleted criteria |
| Repeated identical repair | Failure fingerprint, candidate changes & retry budget |
| Lost requirement | Instruction discovery, compaction boundary & handoff artifact |
| Conflicting edits | Worker ownership, worktrees & integration order |
| Unexpected spend | Worker count, model/effort choice, repeated context & polling |
| “Passing” result but broken feature | Which candidate and behavior the check exercised |
```

```{attention} Q&A
:class: dropdown
*What is available natively?*

- Claude exposes telemetry through OpenTelemetry.
- Codex's structured execution stream reports lifecycle/tool events.
- Copilot exposes session/task information & structured CLI output.

*Should every prompt and tool payload be logged?*

- No. They may contain secrets, private source or user data.
- Redact or omit sensitive payloads; restrict retention & access.
```

&nbsp;

### Harness Evaluation
- **What**: Repeatable task trials measuring the whole agent system. {cite:p}`anthropic_agent_evals`
- **Why**: A prompt, hook or graph change can improve one example while breaking another.
- **How**:
    1. Collect representative tasks & past failures.
    2. Start trials from equivalent workspace/environment states.
    3. Grade resulting behavior with machine checks where possible.
    4. Inspect traces for hidden failures, not only the final text.
    5. Compare success, cost & latency under the same task distribution.

```{note} Example
:class: dropdown
- Regression cases for the parser workflow:
    - Empty input rejected; valid input preserved.
    - Acceptance command fails to start.
    - Tool permission denied.
    - Worker returns partial output.
    - Stop hook emits malformed JSON.
    - Repair never fixes the failing check.
    - Review changes the candidate after an earlier passing check.
- Desired outcomes include honest blockage, not only successful completion.
```

```{attention} Q&A
:class: dropdown
*Why evaluate more than once?*

- Model choices & tool/environment conditions can vary between trials.
- A single successful trace does not establish reliable behavior.

*What must stay fixed for a fair comparison?*

- Task inputs, repo revision, environment, permission policy & acceptance checks.
- Record model/runtime/config versions & resource budgets.

*What does a reviewer add?*

- A distinct attempt to find unsupported claims or missed behavior.
- Reviewer agreement is not proof; verify findings against artifacts & reproducible evidence.
```

&nbsp;

### Packaging & Reproducibility
- **What**: Versioned distribution of instructions, skills, workers, hooks & integration configuration. {cite:p}`claude_plugins,codex_plugins,copilot_plugins`
- **Why**: A workflow that works only in one developer's global configuration is hard to reproduce.
- **How**:
    1. Start with project-local configuration.
    2. Package repeated cross-project components using the product's plugin support.
    3. Record the runtime version & configuration used for each evaluation.
    4. Re-run regression cases after runtime or plugin updates.

```{attention} Q&A
:class: dropdown
*Does “supports skills/plugins” mean compatible packages?*

- No. Shared `SKILL.md` conventions do not imply identical hooks, manifests, tool names or permission behavior.

*What should not be packaged?*

- Credentials, transient session output & machine-specific paths.
- Keep environment-dependent values explicit.

*Why avoid copying all global configuration into a project?*

- It may introduce unrelated tools, hidden hooks & broader permissions.
- Reproduce the required capabilities, not one person's entire assistant environment.
```

&nbsp;

## Putting It Together

### Claude Code Workflow
- **What**: Project-configured execution with optional workers, hooks & scripted orchestration. {cite:p}`claude_subagents,claude_headless,claude_workflows,claude_cli`
- **Why**: The native runtime already supplies the inner loop; customize the task boundaries instead of rebuilding it.
- **How**:
    1. Add concise project guidance in `CLAUDE.md`.
    2. Reuse a task skill when the procedure repeats.
    3. Add a restricted reviewer when independent inspection is useful.
    4. Use a [completion hook](#completion-hooks) for feedback; retain an independent delivery check.
    5. Opt into a scripted workflow only when the dependency graph justifies it.

````{note} Example
:class: dropdown
- Reviewer definition in `.claude/agents/parser-reviewer.md`:

```text
---
name: parser-reviewer
description: Check parser changes for unintended compatibility breaks.
tools: Read, Grep, Glob
model: inherit
maxTurns: 6
---
Read only the supplied parser files.
Check the candidate against the supplied compatibility contract.
Return confirmed issues with a counterexample and relevant lines.
Do not edit files or invent findings.
```

- Start with interactive planning:

```bash
claude --permission-mode plan
```

- Inspect a trusted repo noninteractively with only read/search tools:

```bash
claude -p "Explain identifier validation in src/parser.py. Do not edit." \
  --tools "Read,Glob,Grep" \
  --output-format json \
  --max-turns 6
```

- Tool restriction does not disable separately discovered hooks or MCP startup code; review project configuration before launching.
- For substantial scripted fan-out, request an interactive dynamic workflow:

```text
Use a workflow to inspect the independent parser backends.
Give each worker one backend and read-only tools.
Collect compatibility counterexamples, then have one worker verify them.
Return only findings that survive verification.
```

- Read & approve the generated orchestration before execution; inspect it through `/workflows`.
- The interactive workflow opt-in is not a portable `-p` keyword or a published cross-vendor JavaScript API.
````

```{attention} Q&A
:class: dropdown
*Why not enable teams for this small parser fix?*

- A writer + focused reviewer is sufficient.
- Teams are for communicating peers; dynamic workflows are for scripted orchestration. Neither is necessary for every task.

*What survives a new task?*

- Project instructions, installed skills & configuration.
- Revalidate transient findings instead of turning them into permanent rules.
```

&nbsp;

### Codex Workflow
- **What**: Stateful coding runs with configured subagents, trusted hooks & structured events. {cite:p}`codex_agents,codex_hooks,codex_exec,codex_worktrees,codex_app_server`
- **Why**: Codex can own tool execution & continuation while an outer controller owns acceptance.
- **How**:
    1. Put project invariants in `AGENTS.md`.
    2. Define specialized roles in `.codex/agents/*.toml`.
    3. Review project configuration & hook trust in the actual client.
    4. Run the task through the CLI or a persistent SDK/app-server session.
    5. Verify the candidate independently before integrating it.

````{note} Example
:class: dropdown
- `.codex/agents/parser-reviewer.toml`:

```toml
name = "parser-reviewer"
description = "Find unintended parser compatibility changes."
sandbox_mode = "read-only"
developer_instructions = """
Inspect only the supplied parser files.
Compare the candidate with the supplied compatibility contract.
Return confirmed issues, counterexamples, and relevant lines.
Do not change files.
"""
```

- Bound local subagent concurrency in `.codex/config.toml`:

```toml
[agents]
max_concurrent_threads_per_session = 2
```

- Ask the coordinator to use `parser-reviewer` after implementation & verification; wait for its findings before integration.
- The role's `sandbox_mode` is a configuration default, not an immutable override of the parent session's live permission choices.
- For Codex hooks, save the [StopGate implementation](#completion-hooks) as `.codex/hooks/stop_gate.py`; its `Stop` input & decision subset are supported by both products.
- `.codex/hooks.json`, invoked from the repo root:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 .codex/hooks/stop_gate.py --hook",
            "timeout": 75
          }
        ]
      }
    ]
  }
}
```

- Hooks are enabled by default in curr releases; `/hooks` reviews new/changed non-managed definitions before they run.
- Inspect w/o granting workspace writes:

```bash
codex exec --json --sandbox read-only \
  "Inspect identifier validation in src/parser.py. Do not modify files."
```

- For an approved implementation, explicitly select `--sandbox workspace-write` instead; that grants more capability, not evidence of success.
- Resume a stored run:

```bash
codex exec resume --last "Check the candidate against the accepted contract."
```
````

```{attention} Q&A
:class: dropdown
*Does Codex still only support `notify`?*

- No. Current hooks cover session, tool, approval, subagent, compaction & turn lifecycles.
- `notify` is a separate notification mechanism, not the lifecycle hook system.

*Which details should not be copied from Claude?*

- Codex supports command & MCP-tool handlers; parsed prompt/agent handler types are not executed.
- Its `PreToolUse` does not support all Claude decisions.
- Current documentation, not a schema from unreleased `main`, defines the supported behavior.

*Does the CLI automatically manage app-style worktrees?*

- Do not assume so. Managed worktree creation & Handoff are documented for Codex in the ChatGPT desktop app.
- A CLI workflow can use ordinary Git worktrees under its own controller.

*Does every app-server command use the thread sandbox?*

- No. An agent `turn/start` & `command/exec` have different roles; `thread/shellCommand` is an explicit full-access user command outside the thread sandbox.
- Do not treat a generic JSON-RPC connection as an isolation guarantee.
```

&nbsp;

### Copilot CLI Workflow
- **What**: Independent continuation & parallel-delegation controls around a configurable coding session. {cite:p}`copilot_cli_reference,copilot_modes,copilot_fleet,copilot_agents,copilot_agent_config,copilot_hooks,copilot_programmatic`
- **Why**: Autonomy, concurrency & permissions need separate configuration.
- **How**:
    1. Load repo guidance & required skills.
    2. Use plan mode when the change needs an agreed approach.
    3. Use autopilot for continued execution; fleet only for genuinely independent work.
    4. Inspect loaded configuration through `/env` & workers through `/tasks`.
    5. Use a restricted reviewer & an independent acceptance command.

````{note} Example
:class: dropdown
- `.github/agents/parser-reviewer.agent.md`:

```text
---
name: parser-reviewer
description: Review completed parser changes against their compatibility contract.
disable-model-invocation: true
tools:
  - view
  - grep
  - glob
---
Inspect only the supplied parser files.
Return confirmed compatibility issues with counterexamples and relevant lines.
Do not modify files.
```

- Use `@parser-reviewer` for explicit delegation.
- The custom-agent reference recommends `disable-model-invocation` instead of retired `infer`; the CLI reference still lists the older spelling. Confirm selection behavior in the installed runtime.
- For independent investigations, not concurr edits:

```text
/fleet Inspect parser input handling and parser callers in parallel.
Keep both workers read-only. Wait for both, then propose one focused change.
```

- A read-only programmatic inspection needs no autopilot continuation:

```bash
COPILOT_TASK_WAIT_TIMEOUT_SECONDS=120 \
copilot --prompt="Inspect identifier validation in src/parser.py. Do not edit." \
  --output-format=json \
  --available-tools="view,grep,glob" \
  --allow-tool="read" \
  --no-ask-user
```

- The wait timeout bounds pending background work at exit, not total run duration.
- For an approved implementation, bound continuation & grant the specific required operations:

```bash
copilot --prompt="Reject empty identifiers, preserve valid input and the API, then run the acceptance check." \
  --autopilot \
  --max-autopilot-continues=2 \
  --output-format=json \
  --allow-tool="read,write(src/parser.py),write(tests/test_parser.py),shell(python3 scripts/check_parser.py)" \
  --no-ask-user
```

- This keeps the runtime's normal control-tool set instead of assuming an undocumented `task_complete` availability flag.
- Continuation count does not bound tool calls inside each turn; use an outer process deadline if a wall-clock limit is required.
- For an implementation run that permits the acceptance command, `.github/hooks/final-check.json` can register the reminder below:

```json
{
  "version": 1,
  "hooks": {
    "agentStop": [
      {
        "type": "command",
        "bash": "python3 .github/hooks/final_check.py --hook",
        "timeoutSec": 5
      }
    ]
  }
}
```

- Save the following handler as `.github/hooks/final_check.py`.
- Omit this reminder from a read-only inspection that cannot execute the check.
- Prompt-mode repo hooks must actually be loaded: trust the folder or explicitly set `GITHUB_COPILOT_PROMPT_MODE_REPO_HOOKS=true` after reviewing it.
- Do not enable repo hooks merely to make an unknown checkout's example run.
````

````{important} Code
:class: dropdown
- One extra verification/reporting turn; a reminder, not an acceptance gate.

```python
import json
import sys


class FinalCheckReminder:
    def __init__(self, command):
        if not command:
            raise ValueError("A verification command is required")
        self.command = command

    def __call__(self, event):
        if not isinstance(event, dict) or type(event.get("stop_hook_active")) is not bool:
            raise ValueError("Expected boolean stop_hook_active")
        if event["stop_hook_active"]:
            return {"decision": "allow"}
        return {
            "decision": "block",
            "reason": (
                f"Run {self.command} against the final candidate. "
                "Report the actual outcome; if it fails or cannot run, "
                "report the task as incomplete. Do not claim an unrun check passed."
            ),
        }


## Example: request once, then return control to the outer acceptance check.
if __name__ == "__main__":
    reminder = FinalCheckReminder("python3 scripts/check_parser.py")
    if sys.argv[1:] == ["--hook"]:
        print(json.dumps(reminder(json.load(sys.stdin))))
    else:
        assert reminder({"stop_hook_active": False})["decision"] == "block"
        assert reminder({"stop_hook_active": True})["decision"] == "allow"
```
````

```{attention} Q&A
:class: dropdown
*Why is the reminder weaker than StopGate?*

- It asks the model to verify; it does not itself execute the check.
- Its `"allow"` means “end this turn,” not “candidate accepted.” Use the [bounded outer check](#bounded-repair) or CI for actual acceptance.

*Which defaults should automation avoid assuming?*

- Current autopilot concept & command-reference pages disagree on the implicit continuation limit.
- Set `--max-autopilot-continues` explicitly rather than silently choosing one account.

*Which operations still need maturity/surface qualifiers?*

- `/every`, worktree commands, local sandboxing & SDK fleet have experimental/preview status.
- Native CLI `agentStop` output is not automatically compatible with every VS Code hook field.
- Cloud agent only loads repo hooks; a local user hook is not cloud deployment configuration.
```

&nbsp;

### Verified Change Walkthrough
- **What**: One implement–verify–review graph using a chosen native product profile. {cite:p}`claude_best,anthropic_long_harness,copilot_sdk_fleet`
- **Why**: Every extension needs a clear job in an actual delivery path.
- **How**:
    1. **Initialize**: select one profile above; inspect its discovered instructions, tools & hooks.
    2. **Contract**: reject empty identifiers, preserve valid inputs & the public API, restrict changed paths.
    3. **Reproduce**: run the existing parser checks & add the missing failing case.
    4. **Implement**: one writer owns `src/parser.py` & its targeted tests.
    5. **Verify**: run the acceptance command against the resulting candidate.
    6. **Review**: a read-only worker receives the contract, candidate diff & actual check results.
    7. **Reconcile**: reproduce each finding; repair confirmed errors & rerun affected checks.
    8. **Deliver**: return the persistent artifact & evidence, or a precise blocker.

```{dropdown} Table: Transition Contract
| Current node | Success | Failure |
|:--|:--|:--|
| Initialize | Known runtime/config & accessible project | Stop for missing prerequisite |
| Reproduce | Failing case distinguishes old behavior | Improve reproduction before editing |
| Implement | Candidate within scope | Return to implementation within budget |
| Verify | Accepted checks pass on candidate | Repair; stop when budget is exhausted |
| Review | No confirmed blocker remains | Reproduce finding; repair & re-verify |
| Deliver | Artifact & evidence refer to the same candidate | Do not deliver stale or incomplete evidence |
```

```{attention} Q&A
:class: dropdown
*What should be omitted from this small task?*

- A vector database, a separate graph framework, a swarm of writers & a long-lived scheduler.
- Native tools, one optional reviewer & a bounded check already cover the contract.

*When does this become a larger graph?*

- Independent backends, separate migrations or multiple integration targets create real dependencies.
- Add workers & nodes for those dependencies, not merely because the runtime permits them.

*What is the smallest reliable harness?*

- The native loop + relevant context + constrained tools + observable acceptance + honest failure.
- Add context management, delegation, hooks & persistence only where they solve a demonstrated lifecycle problem.
```

&nbsp;
