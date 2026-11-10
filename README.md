# AI Agent Pro

A general-purpose, local AI agent platform built with Python, Streamlit, the official OpenAI Python SDK and SQLite. Create specialist assistants, keep separate conversations, and run a bounded draft-and-review workflow with durable execution records.

Phase 3 adds a persistent PDF/text knowledge base with agent-scoped retrieval and source citations. Answers support Markdown formatting. Phase 4B adds explicitly approved read-only tool calling, arithmetic and syntax inspection. Phase 4C adds action-specific human approval for bounded practice text replacements. Phase 4D coordinates versioned sequential multi-agent workflows. Phase 4E adds an interactive visual workflow studio; browsing, submitted-code execution and external actions remain unavailable.

**AI assistance:** This project was developed with substantial AI coding assistance. I’m reviewing the implementation to deepen my understanding and build my ability to maintain and extend it independently.

## Features

- Create, edit, select and delete agents with names, descriptions and instructions.
- Configure models, recent context length, output token caps, task token budgets, request counts and task deadlines per agent.
- Chat with persistent, agent-specific conversations. Start a new conversation or explicitly confirm clearing its messages and task records.
- Execute one answer stage, a draft plus review, or a draft plus two reviews. Reviews improve the previous answer; they do not constitute independent verification or hidden reasoning inspection.
- View task status, failures, elapsed time, request attempts, configuration snapshots, stage records and token usage when available.
- SQLite foreign keys, parameterized values, atomic task/message persistence and one active task per conversation.
- Deadline-based recovery for task records left running by a crashed application.
- Private credential configuration, bounded inputs, safe text display and best-effort secret redaction.
- Secure PDF/text uploads, named collections, agent assignments, persistent embeddings, exact cosine search and cited document-grounded answers.
- Atomic re-indexing, embedding reuse, document/collection deletion and interrupted indexing recovery.

## Install

The package declares Python 3.11 or newer. Release validation uses Python 3.14 on macOS; Python 3.11–3.13 and Linux have not been exercised in this release check. Full PDF extraction, resource-limited tools and practice file changes target macOS/Linux and rely on POSIX resource and descriptor APIs. Windows is not a supported full-feature platform; use a Linux environment such as WSL rather than native Windows. From this project directory on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip>=26.2'
python -m pip install -r requirements.txt
```

Installation uses an editable package so the small `app.py` launcher can import the implementation in `src/ai_agent_pro`.

The initial validation environment used Python 3.14, Streamlit 1.66.1, OpenAI SDK 2.54.0, NumPy 2.5.4, psutil 7.2.2 and pytest 8.4.2. The release requires pypdf 7.0.0 or newer within major version 7: version 7 fixes PDF parsing/resource-consumption issues affecting the previously tested 6.21.4. See the [pypdf maintainer advisory](https://github.com/py-pdf/pypdf/security/advisories/GHSA-59jv-x5x2-gg4c) and [7.0.0 release notes](https://github.com/py-pdf/pypdf/releases/tag/7.0.0). Resource limits remain enabled and are not a substitute for dependency security updates. The development extra requires pytest 9.0.3 or newer within major version 9 to include its [temporary-directory security fix](https://github.com/pytest-dev/pytest/releases/tag/9.0.3). Upgrade pip as shown above before installing dependencies. Dependency ranges are declared in `pyproject.toml`; other Python/package combinations have not all been exercised.

## Private API configuration

Choose one method; do not put a real key in source code, chat messages, agent instructions, screenshots or Git.

1. Set `OPENAI_API_KEY` in your local environment or in a private PyCharm run configuration. Avoid shared run configurations containing credentials.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the example value locally:

```toml
OPENAI_API_KEY = "your-private-key"
```

The real secrets file and `.env` files are ignored by Git. The application does **not** load `.env` files automatically. Environment configuration takes precedence over Streamlit secrets. No credential entry field or API key is stored in SQLite. Credentials are held in process memory and used only by the SDK. The provider explicitly uses `https://api.openai.com/v1` rather than inheriting an alternate `OPENAI_BASE_URL`.

The UI works without a key; chat stays disabled. A working key, API billing and model access are needed for actual responses. The model selector offers `gpt-4.1-mini`, `gpt-4.1` and `gpt-4.1-nano`; availability depends on your account. These are deliberate supported identifiers, not a promise that every account can access them.

## Launch in PyCharm

Select this project's `.venv` as the interpreter. Install dependencies using the PyCharm terminal, then run:

```bash
python -m streamlit run app.py
```

Alternatively create a Python run configuration with **Module name** `streamlit`, **Parameters** `run app.py`, and **Working directory** set to this project root. Open `http://127.0.0.1:8501` if the browser does not open automatically. Running `app.py` as a plain Python script is not the Streamlit launch procedure.

1. Open **Agents** and create an agent.
2. Select it in the sidebar and open **Chat**.
3. Submit a message. Each submission authorizes sending that message, bounded recent context and agent instructions to OpenAI.
4. Inspect results and usage under **Activity**.

`src/main.py` is the user's preserved PyCharm sample, not the application entry point.

## Architecture

```text
Streamlit UI → AgentEngine → OpenAIProvider → OpenAI Responses API
                    ↓
               Repository → SQLite
```

- `app.py`: small Streamlit launcher.
- `config.py`: agent/settings data classes, model allowlist, validation bounds and safe application error type.
- `safety.py`: input checks, platform instruction boundary, conservative token reservation and log/content redaction.
- `migrations.py`: transactional schema migrations through v5, startup serialization and future-schema rejection.
- `database.py`: SQLite repository, agent CRUD, conversation ownership, transactional messages/runs, recovery and cascaded deletion.
- `provider.py`: asynchronous official SDK adapter; genuine Responses API streaming, SDK error translation, output validation and optional usage.
- `engine.py`: history selection, sequential draft/review orchestration, token reservations, task deadline/cancellation and execution persistence.
- `execution.py`: session-owned chat/workflow worker, latest-only progress mailbox, cooperative Stop and abandoned-session cleanup.
- `approvals.py`, `approval_resume.py` and `practice.py`: exact-action approval, durable continuation and bounded practice file replacements.
- `workflow_contracts.py`, `workflow_store.py` and `workflow_engine.py`: versioned sequential definitions, durable checkpoints, isolated child executions and global budgets.
- `ui.py`: application shell, credential lookup, navigation and safe error boundary.
- `presentation/`: overview charts, agent cards/forms, chat, activity details, shared styling and progress views.
- `analytics.py`: parameterized, read-only aggregates over retained records.
- `telemetry.py`: safe execution events, measured stage durations and isolated progress observers.

The provider boundary supports controlled Responses API function calls through the Stage 4A registry and executor. The repository and engine are independent of Streamlit, enabling isolated tests and future service entry points. The `rag/` package isolates upload validation/storage, extraction, chunking, embeddings, indexing, scoped retrieval, evidence formatting and knowledge-base UI. It uses the same repository connections and SQLite transactions. Tool execution is bounded and sequential. The application coordinates sequential multi-agent workflows through its own workflow engine; no third-party orchestration framework is required.

### Execution semantics

Agent settings are validated in both forms and backend code. Maximum answer/review stages are 1–3; SDK retries are disabled. Without approved tools, each stage makes at most one SDK API attempt. With tools, the task permits at most 8 model requests and 5 sequential tool calls across all stages. Output caps are 64–16,384 tokens per request (default 1,200; existing agent settings are retained); task budgets are 1,024–100,000 input-plus-output tokens; deadlines are 5–180 seconds.

Before a request, UTF-8 byte length plus a framing allowance and the output cap conservatively reserve token capacity. This is intentionally more restrictive than exact tokenization and can reject a task that would fit by token count. Completed-call usage is used when available; otherwise reserved capacity is charged to the local budget. Execution stops if capacity cannot cover the next request. These are application limits, not an account spending cap.

The engine sends up to 30 recent messages and trims history to 24,000 UTF-8 bytes before adding the current message. Full retained history stays in SQLite; the model does not see all of it. Failed tasks' messages remain visible but are excluded from later model context. Drafts are held in memory during execution; only the validated final answer is saved as a chat message. The final generation stage streams a provisional preview; earlier draft/review stages show progress without displaying intermediate drafts. Task history records stage names and usage, not private chain-of-thought.

A single overall deadline bounds awaited network calls; remaining time is passed to the SDK and `asyncio.wait_for`. Cancellation closes the async client. Network/server cancellation and provider billing cannot be guaranteed. Failed/incomplete responses are reported as failures and partial answers are not added to chat. Such calls may still be billed. Failed-task aggregate usage is marked unavailable; completed-stage usage remains in the stage records.

A unique SQLite index prevents overlapping runs in one conversation. Interrupted tasks can be explicitly recovered after their configured deadline plus ten seconds. Ordinary tasks are marked interrupted; recoverable approval-publication checkpoints return to the paused state for explicit reconciliation. Recovery never automatically retries an uncertain model request. Editing settings does not change an already-running task's in-memory configuration.

## Data and safety

The default database is `data/agentpro.sqlite3`, excluded from Git. It contains agent instructions/settings, conversation metadata, redacted messages and task records. `AI_AGENT_PRO_DB` can point to a different local database for tests or deployment. Clearing a conversation deletes its messages and task records; deleting an agent also deletes its conversations. Confirmations guard these UI actions. SQLite deletion is not a secure disk erasure mechanism.

- Platform instructions are supplied separately through the Responses API `instructions` parameter; untrusted messages stay in user/assistant roles.
- Function tools are absent by default. Only exact versions explicitly approved for the selected agent and execution are supplied. File tools require an explicit eligible-file manifest. Practice text replacements require a separate human decision bound to the exact proposed change. Shell commands, code runners, network connectors, arbitrary writes and external action adapters remain unavailable.
- Inputs have length/control-character checks. Assistant replies use Streamlit’s native Markdown renderer with `unsafe_allow_html=False`: headings, bold text, lists and code blocks render while raw HTML is escaped. No model output is passed to HTML components or executed as code. A Mistune parsing pass permits only HTTP, HTTPS and mailto link destinations; other destinations become inert labels, and Markdown images become text labels to avoid remote image requests; user messages and retrieved document excerpts remain plain text. Stored messages and model context are unchanged.
- Recognizable API keys, bearer tokens and labeled secrets are redacted before message/config persistence. Logs contain safe error categories and task IDs; third-party debug logging is suppressed. Pattern-based redaction cannot identify every secret: do not enter sensitive information.
- `store=False` disables response storage for later API retrieval. It does not imply zero provider retention; review the provider's applicable data policies.
- Agent instructions and prompt boundary language help resist injection but cannot eliminate it. Model answers can be wrong. These safeguards do not make the application fully secure.

This is a trusted local application, bound to `127.0.0.1`. It has no login, tenant isolation, encrypted database or hardened production hosting. Anyone who can read the project data or access the running app can access retained conversations. Keep the host account and database private. Application users can manage local agent data, and approved model calls can read only eligible explicitly listed project files and assigned knowledge sources.

## Tests and verification

```bash
python -m pytest
```

Tests cover agent/settings validation, SQLite persistence and isolation, cascade/clear behavior, active-run locking/recovery, redaction, context retention, request/token/deadline limits, safe failures, actual SDK serialization through an HTTP mock transport, and Streamlit create/edit/delete/chat workflows. All OpenAI requests in tests are mocked: no API credentials or paid calls are needed.

The startup smoke test starts a real headless Streamlit process on a temporary localhost port, checks `/_stcore/health`, and terminates it. It requires permission to bind a local socket; restrictive sandboxes must allow localhost networking to run the complete suite. Streamlit AppTest separately executes and interacts with the actual application script.

Release verification on 2026-11-08 used Python 3.14.5 on macOS, Streamlit 1.67.0, pypdf 7.0.0, pytest 9.2.0 and Playwright 1.64.1. A disposable environment installed the final declared dependencies from public PyPI with an empty home directory and personal pip configuration disabled. Fresh-install release-safety/RAG checks passed (99 tests), and the browser-enabled full suite passed (648 tests). `pip check` found no broken requirements; an OSV query of 55 installed public packages returned no known advisory matches at that time. Test isolation blocks external networking, scrubs inherited credentials, disables private Streamlit secrets loading and uses disposable databases. Synthetic SQLite backup/restore checks cover integrity, foreign keys, retained records/uploads/workflow data and restored-database independence. This does not establish compatibility across every allowed dependency or platform version.

## Limitations and roadmap

A prior manual Phase 4B calculator smoke test was reported successful (6375, two model requests). That limited observation is not a release-wide evaluation of model access, latency, billing, answer quality or retrieval quality. Release verification uses mocked OpenAI responses and makes no paid API calls; live use requires a configured account. Tests verify adapter behavior with SDK-backed mocked HTTP responses. Assistant output supports safe Markdown and streaming; direct chat attachments are not implemented. Upload documents through Knowledge base instead. Chat history and conversation lists currently load from local SQLite without pagination; Activity uses SQL pagination, and agent cards and Knowledge base listings are paginated. There is no account-wide spending limit, authentication, automatic retry or background job queue. Abrupt termination may leave running records requiring recovery.

Sequential multi-agent workflows, a visual workflow studio and operational monitoring are implemented. Parallel or branching workflows, OCR, larger-scale vector backends, Docker and cloud hosting remain future work.

## API references

Implementation follows the [official OpenAI Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create) and [GPT-4.1 mini model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini). API fields, access and model availability may evolve.

## Phase 3: document knowledge base

### Start and use

From the project root, install/update this project's environment and start the app:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Configure `OPENAI_API_KEY` privately as described above. Uploading and managing documents works without a key; indexing and document chat require one. The sidebar reports credential presence, not verified connectivity.

1. Open **Knowledge base**, create a collection, and upload a PDF or UTF-8 `.txt` file.
2. Select **Index document**. This authorizes sending extracted text to OpenAI for embeddings. Wait for **Ready for retrieval**.
3. Select an agent in the sidebar, choose its collections, then select **Save collection assignments**.
4. In **Agents → Edit selected agent**, enable **Document-grounded answers** and save.
5. In **Chat**, ask a standalone question. Answers cite `[S1]`, etc.; expand **Retrieved sources** to see filenames, page/line locations and excerpts.
6. Use **Re-index document**, **Delete document**, or **Delete collection** in Knowledge base. Deletion requires confirmation. Interrupted indexing can be recovered after 190 seconds and explicitly retried.

Existing agents default to document mode off. Their ordinary chat, history, budget controls and review stages retain their previous behavior. Document mode uses the current question and fresh source excerpts, omitting earlier chat context to avoid reintroducing deleted or revoked document material. Ask complete questions rather than relying on previous pronouns or references.

### Processing and persistence

Original file bytes, sanitized extracted text, vectors, collection assignments, indexing jobs and citation snapshots are stored together in the existing local SQLite database. There is no external vector service or file search tool. The schema upgrades transactionally to version 5 and refuses databases from a newer application. Before upgrading an existing installation, back up the database with the app stopped. Newly created database files use mode `0600`; newly created data directories use `0700`.

Limits are enforced independently of the UI: PDF/text only, 10 MiB per upload, 200 PDF pages, 1 MB extracted UTF-8 text, 2,500 chunks per document, 100 documents per collection, 100 collections, 100 MiB total original uploads and 20,000 indexed chunks across the knowledge base. Filenames cannot contain paths or control characters. MIME and extension alone do not establish validity. Malformed/encrypted PDFs and textless scans fail safely during indexing; partial scans receive a warning. OCR and table-layout reconstruction are outside this phase.

PDF extraction runs in a credential-free subprocess, with a 20-second wall deadline, a CPU limit and a memory watchdog. Linux additionally uses an address-space limit; macOS uses monitored resident memory because it rejects that limit. Monitoring can overshoot its threshold between checks. This is resource isolation, **not an operating-system security sandbox**. PDF links, scripts and embedded attachments are never executed or followed by the application. psutil provides memory monitoring; the PDF worker currently targets macOS/Linux.

Chunks preserve PDF page numbers or text line ranges. They use up to 600 UTF-8 bytes with up to 80 bytes of overlap, preferring whitespace boundaries. Byte length conservatively upper-bounds text token usage, matching the existing engine's budget design; these are not 600-token chunks. No tokenizer files are downloaded.

The embedding adapter uses `text-embedding-3-small` with 1,536 dimensions, batches of at most 32 texts, the fixed OpenAI API endpoint, bounded timeouts and no automatic retries. Index jobs have a 180-second deadline and a conservative 1,100,000-token reservation ceiling. Identical chunks reuse embeddings only within the same document and model/processing version. Re-indexing builds and validates a replacement generation before publishing; failure retains the previous ready generation. Corrupt cached vectors are regenerated.

Search filters to assigned collections **before** scoring, compares normalized vectors by exact cosine similarity, deduplicates identical chunks and supplies up to five excerpts above a starting threshold of 0.25. Query embeddings accept at most 6,000 UTF-8 bytes. Model/dimension/processing mismatches are excluded until re-indexed. Exact search is bounded for this local dataset, not intended as a large-scale ANN service. Similarity scores are not confidence probabilities; tune the threshold with representative evaluations.

### Citations, safety and deletion

Platform instructions remain separate from document content. Filename, location and excerpt are JSON-serialized as untrusted user-role data; only explicitly approved read-only tool calls are available; external actions remain unavailable. Unknown citation labels or missing citations fail safely, except for the explicit insufficient-evidence response. No relevant sources returns that response without a generation request. Documents can still contain convincing misinformation or injection attempts: these safeguards cannot guarantee factual support or eliminate prompt injection.

Answers and source snapshots are saved atomically. Sources are revalidated before each generation request and at final persistence, so deletion/re-indexing during a task cannot commit stale citations. An already in-flight API request cannot be recalled when a source is deleted. Old citation snapshots remain tied to their original generation after re-indexing. Deleting documents removes original bytes, chunks, vectors, jobs and retained citation excerpts; collection deletion cascades to its documents and assignments. Deleting an agent removes its assignments while retaining shared collections. Sources from revoked collections are hidden from that agent's source panels.

**Original uploads retain their original bytes, including any secrets.** Only extracted text is subject to best-effort redaction. Do not upload sensitive documents unless you intend that local retention and provider processing. Historical chat answers can already quote a deleted document; clear those conversations separately. Turning document mode off restores ordinary retained-history behavior. SQLite deletion is not secure disk erasure. The application remains trusted/local, without authentication or encryption at rest.

Query and indexing embedding usage are recorded separately from generation usage. Query embeddings also count toward the task's token budget and overall deadline. Failed API calls can be billed even when usage is unavailable. Neither local budget is an account spending cap. Indexing is a bounded foreground operation; there is no background queue or automatic paid retry.

### Tests and retrieval evaluation

```bash
.venv/bin/python -m pytest
```

The full offline suite includes real subprocess PDF extraction and Streamlit startup, SDK-backed HTTP mocks, UI document workflows, labeled deterministic retrieval fixtures, cross-agent isolation, citations, prompt-injection payload serialization, upload validation, quotas, transactional migrations, concurrent startup, deletion/re-index races, cancellation, timeouts, partial-batch failure and corrupted-vector repair. Tests use temporary databases and make no paid API calls. The startup smoke test needs localhost socket access.

Deterministic fake embeddings verify ranking and integration contracts; they do **not** establish semantic quality of the live model. An opt-in evaluation uses synthetic documents and ten labeled questions, an unassigned canary collection, and a disposable database under `data/evaluations/`:

```bash
# Requires OPENAI_API_KEY in the environment; authorizes paid embedding calls.
.venv/bin/python scripts/evaluate_rag.py --live
```

It reports Recall@5 and checks zero leakage from the unassigned collection, returning failure below 90% recall. The script reopens the database to exercise persistence and removes its disposable data afterward. It never reads or changes the application's knowledge base. Its lifecycle is also tested offline. Live retrieval quality and embedding-account access/billing have not been established by the offline release checks.

API behavior follows the [official embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) and [Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

## Modernised interface

The sidebar provides **Overview**, **Agents**, **Chat**, **Knowledge base**, **Activity**, **Workflows**, and **Help**. Agent selection is explicit; the overview and activity screens have their own **All agents** or individual-agent scope. Search filters, selected conversations and collections survive navigation within the session.

- **Overview:** real retained task counts, operational completion rate, median recorded task time, known generation usage, activity/error/usage/duration charts, current document readiness and recent-task shortcuts. Indexing performance uses measured v3 job durations only. No sample data is inserted into the application database.
- **Agents:** searchable, paginated cards with real chat/edit/collection actions; structured identity, behaviour, execution-limit and document settings; explicit deletion confirmation. Cards distinguish configuration from retained task counts.
- **Chat:** agent/settings shortcuts, persistent conversation selection, safely rendered Markdown replies, progressive final-stage previews, Stop, measured stage feedback and selectable source excerpts. Ordinary chat and document-grounded context rules are preserved.
- **Knowledge base:** searchable, filtered, paginated metadata table; selected-document controls; separate active-index readiness and latest-attempt status; duplicate-upload feedback; indexing history; eligible recovery and confirmed deletion. Listings do not load original document BLOBs.
- **Activity:** scoped/date/status/title filters, SQL pagination, task inspection, conversation navigation, source records and workflows derived only from recorded events. Recovery is disabled until the real deadline plus ten seconds; use Refresh records to update eligibility. Refresh library serves the same role for indexing.
- **Workflows:** visual sequential agent nodes, immutable definition versions, separate step permissions, structured handoffs, global budgets and exact-action approval controls.
- **Help:** private setup, data processing, troubleshooting and actual capability limits.

Chart dates and filter boundaries use **UTC**. Completion rate excludes running tasks and describes operational status, not answer accuracy. Task-duration statistics exclude recovered interrupted records without a meaningful duration. Generation token totals sum known completed-stage usage exactly once, including completed stages from failed tasks; run aggregate totals are not added a second time. Unreported/failed-call usage is not zero. Query and indexing embedding usage are separate and report their known-record coverage; reservations are not consumed tokens. Clearing conversations and deleting documents removes their associated records, so totals are not lifetime or account billing totals. Document inventory is current, not a historical snapshot of the selected task dates.

Schema v3 adds execution events and nullable indexing completion/duration fields. Earlier records are preserved without invented stage timings or indexing durations. Started-only events remain visibly unresolved. New progress events record retrieval, answer/draft/review, saving, extraction, chunking, actual embedding batches and publication. A progress observer failure does not repeat requests or change a successfully persisted answer into a failed run. Events contain fixed labels, counts and safe error categories, not prompts, source excerpts, credentials or model reasoning. Clearing/deleting an owning task/document cascades to its events.

The interface uses native Streamlit layout/widgets and its Altair chart stack. Tables accompany charts, controls have visible labels, state is expressed with text as well as colour, and layouts are checked at mobile/tablet/desktop widths. Assistant replies use native Markdown with HTML disabled; user messages and document excerpts remain plain text. CSS is limited to static application-owned containers. This is a bounded foreground application: no durable background queue, automatic retry, bill estimates, confidence score or persisted retrieval-quality dashboard has been added. Chat uses a bounded in-process worker so the UI can display previews and handle Stop; it is not a persistent job service.

### Browser-level interface tests

The normal suite uses Streamlit AppTest plus backend/aggregation tests. Optional Playwright smoke tests run against disposable localhost databases, without credentials or paid requests, at **390px**, **768px** and **1440px**. They cover keyboard form submission, navigation, upload, disabled indexing/chat, chart rendering and horizontal overflow. These are smoke checks, not a full accessibility certification.

```bash
.venv/bin/python -m pip install -e '.[dev,browser]'
PLAYWRIGHT_BROWSERS_PATH="$PWD/data/browser-cache" .venv/bin/python -m playwright install chromium
AI_AGENT_PRO_BROWSER_TESTS=1 .venv/bin/python -m pytest
```

The browser cache is ignored beneath this project's `data/`. Without the opt-in variable, browser cases are skipped; ordinary tests still run. The full suite and Streamlit startup checks need localhost socket access. Browser tests save screenshots in pytest's temporary artifact directories and never use your application database.


### Streaming chat, copying code and stopping

Launch with `.venv/bin/python -m streamlit run app.py`. Select an agent in Chat and ask for an explanation with a fenced Python code example. The final generation stage appears progressively in a provisional assistant bubble. Earlier stages display Drafting/Reviewing; retrieval and Saving have separate states. After successful provider completion, budget checks and (for RAG) citation/source validation, the final answer is saved once and the UI shows Completed. Stored history uses the same safe Markdown renderer and syntax highlighting.

Hover over a code block or focus it with the keyboard to reveal Streamlit's native **Copy to clipboard** control. It copies the displayed code with indentation, internal newlines and blank lines, without Markdown fences (the fence-boundary newline is omitted by the native renderer). Copied code is never executed by the application.

Select **Stop generation** while waiting for or receiving output. The active async request and SDK stream are closed, later review stages are skipped and no partial assistant answer is stored. The UI shows Cancelled; Activity retains the task as `interrupted` with a cancellation reason to preserve existing analytics. Stop is disabled during the short atomic save; a completed save wins a late Stop. Disconnects, incomplete streams, timeouts and invalid citations show Failed and remove the provisional preview. The original user message remains visible but failed/cancelled turns are excluded from subsequent model context. Failed or cancelled requests can still incur provider usage; unavailable usage is never reported as zero.

Submitting again and changing/clearing conversations are disabled while the session owns an active request, with SQLite's existing one-running-task constraint as a second guard across sessions. Navigating away from Chat cancels the owned request. A missing UI heartbeat cancels an abandoned worker after about 15 seconds; the overall agent deadline also applies. Browser reloads do not resume a stream, and a process crash still uses the existing deadline-based recovery. Workers have no Streamlit context, persist no deltas and retain only a bounded latest preview.

Every preview is redacted before it reaches the UI. The unfinished final word is withheld until a boundary, preventing a secret token split across network chunks from being exposed before the existing best-effort redactor can recognize it. Safe Markdown is applied on every refresh, including unfinished links/fences; HTML remains disabled and remote Markdown images remain inert. These are the same best-effort secret patterns used for stored messages, not a guarantee that arbitrary sensitive data can be detected.

Foreground streaming keeps `store=False`, no tools unless explicitly approved, no automatic retries and the fixed API endpoint. Stop closes the foreground connection rather than enabling provider background mode. See the [official streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses) and [foreground/background cancellation guidance](https://developers.openai.com/api/docs/guides/background).

Offline tests cover real SDK SSE parsing through HTTP mocks, stalled/disconnected streams, timeouts, cancellation and save races, worker cleanup, split-secret previews, citation validation and source deletion during generation. Browser tests use an explicitly separate test launcher with fake responses, disposable databases and no paid calls; they verify progressive display before completion, Stop, failure cleanup, syntax highlighting and actual keyboard/pointer clipboard copying.


### Incomplete answers and output limits

The per-request output cap and the total task budget are separate. A detailed ten-step tutorial with code can outgrow the default 1,200 output tokens even with plenty of task budget remaining. Open **Chat → Agent settings → Output tokens per request** (or **Agents → Edit selected agent**) and try 4,096–8,192 for a longer answer, or split the request into smaller parts. The application bounds the cap at 64–16,384; existing agents and the economical default are not silently changed. Larger caps can increase cost and latency. Input, previous stages and RAG query usage still count toward the existing 1,024–100,000-token task budget, and the 5–180-second task deadline still applies. The 100,000-character local response bound also remains in force.

For both streamed and ordinary responses, a terminal `status=incomplete` is not success. The provider now examines `incomplete_details.reason`: `max_output_tokens` produces an **Output limit reached** error, while `content_filter` and absent/unrecognized reasons produce different messages without claiming that more tokens will fix them. The safe error retained in Chat/Activity includes the API status, normalized termination reason, requested output cap and reported output tokens when the API supplies them. Unknown status/reason values and raw API error messages are never echoed. Older failures cannot be retroactively diagnosed when their termination details were discarded.

Incomplete responses remain failed tasks. No partial assistant message or RAG citation snapshot is committed, no subsequent review is started, and no automatic retry/continuation spends more tokens. The user message remains visible but is excluded from later model context with the failed turn. Stop cancellation remains interrupted, and the active stream is closed in either case. Diagnostics may show reported output from an incomplete call, but it is not counted as a completed stage or silently added to dashboard usage aggregates; failed-run aggregate usage remains unavailable.

See the [official Responses streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events) for the incomplete status/reason fields. Tests cover real SDK SSE and non-streamed output-limit responses, missing/filtering reasons, sanitized diagnostics, no partial persistence, RAG citation safety, bounded configuration, manual adjustment, budget enforcement and browser error recovery.


## Phase 4A: controlled read-only tool foundation

Phase 4A introduced the versioned registry and deny-by-default execution service in `src/ai_agent_pro/tools/` as a backend-only foundation. In the current application, Phase 4B connects explicitly selected tools to the chat engine and OpenAI; Phase 4C adds exact-action approval controls; and Phases 4D–4E add sequential workflow execution and its visual studio. The five read-only registry operations below remain separate from human-approved practice replacements.

The five initial version `1.0.0` operations are:

| Tool | Input fields (all required) | Actual operation |
|---|---|---|
| `calculator` | `expression` | Arithmetic AST interpreter supporting `+ - * / // % **` and unary signs; no names, calls, attributes, assignments or code evaluation |
| `python_inspect` | `source` | Parse Python syntax and report imports, function/class declarations and safe syntax locations; never execute submitted source |
| `project_read` | `path`, `start_line`, `line_count` | Read up to 50 lines from an explicitly approved UTF-8 source/document file |
| `project_search` | `query`, `paths`, `max_results` | Literal, case-insensitive search of approved files; `paths=[]` selects the current explicit manifest, not the entire directory |
| `knowledge_search` | `query`, `max_results` | Local lexical AND-keyword search over the agent's assigned, active indexed text, with source/document/generation references |

Knowledge search makes no query-embedding or generation request, uses no network and changes no SQLite records. It is deterministic keyword retrieval, not semantic ranking; use short keywords such as `London deployment`. The existing semantic RAG chat remains unchanged. This tool only reads assigned active generations with the current text-processing version, does not select original upload BLOBs or embedding vectors, and revalidates returned sources against current assignments/generations before releasing the result. Its source labels are retrieval metadata, not verification of factual support or saved chat citations.

### Host-controlled permissions

Only trusted application code can register vetted implementations, choose the project/database binding, grant an exact tool version, and approve an explicit file manifest. The model-facing payload cannot contain permissions, agent identity, root paths, database paths, handler names or extra fields. The `agent_id` passed to the executor must come from the trusted caller's active-agent context; it is not an authentication mechanism. There is no remote tool endpoint. Chat and Workflows supply UI/model integration through explicit per-execution or per-step selections; model text cannot create grants.

A manual, trusted-host example (use an actual active agent ID):

```python
import asyncio
from ai_agent_pro.tools.executor import ToolExecutor
from ai_agent_pro.tools.permissions import ToolBudget

executor = ToolExecutor()  # Binds this installation's project root and local database.
agent_id = "your-active-agent-id"
executor.permissions.allow(agent_id, "calculator", "1.0.0")
result = asyncio.run(executor.execute(
    agent_id, "calculator", "1.0.0", {"expression": "(12 + 8) / 4"},
    budget=ToolBudget(max_calls=5, seconds=10, result_bytes=32768),
))
# result.ok, result.data, result.error_code, result.error, result.elapsed_seconds

executor.permissions.approve_files(agent_id, ["README.md", "src/ai_agent_pro/engine.py"])
executor.permissions.allow(agent_id, "project_search", "1.0.0")
# Search args: {"query": "max_output_tokens", "paths": [], "max_results": 10}
# project_read needs its own separate version grant.
```

Grants and manifests are in memory, scoped to the `ToolPermissions` instance, and disappear on restart. Importing or listing the registry grants nothing. Revocation or manifest changes during execution invalidate the returned result. A default `ToolExecutor()` permits no tool calls; changing an agent's natural-language instructions cannot alter that. Later integration must preserve this boundary rather than derive grants or identity from model text. Initialise one executor/authority for the application rather than creating an independently granted service for each model request.

### Filesystem, data and resource boundaries

Project bindings default to this AI Agent Pro installation. Tools accept relative, explicitly approved filenames, never a caller-selected root or glob. Reads use directory-relative descriptors with `O_NOFOLLOW`; symlink directories/files, multi-linked files, directories, devices and FIFOs are denied. Traversal, absolute/Windows/UNC paths, encoded paths, hidden components, Git internals, private/environment/data directories and credential filenames/formats are denied even if included in a manifest. Only `.py`, `.md`, `.rst`, `.txt`, `.js`, `.jsx`, `.ts`, `.tsx`, `.css`, `.html` and `.sql` source/document files are eligible. JSON/YAML/TOML/INI and binary/configuration/key formats are excluded. Other projects are outside the binding. These operations never modify files.

At most 32 files can be approved per agent, each up to 256 KiB; project search reads at most 1 MiB per call and returns up to 20 matches. Lines are bounded and report truncation; redaction is applied before line selection/search while preserving source line numbers. Known key/bearer/labeled-secret patterns are redacted in tool results; raw exceptions, source and credentials are not logged. Error codes use fixed messages. Redaction remains best-effort and cannot recognise every arbitrary secret in an otherwise approved source file; approval is not permission to read private configuration.

Schemas reject remote references, conditional/regex schema keywords, unknown fields, wrong types (including floats for integers and booleans for numbers), non-finite numbers, overlong strings, deep structures and unbounded definitions. Registry versions cannot be overwritten; catalog/definition copies cannot mutate registered schemas. Runtime results are schema-validated, redacted again by the parent, size-checked and permission-checked before release.

Each operation runs in a **fixed, credential-free Python worker**, without an inherited API key, cloud credential variables or `PYTHONPATH`. There is no generic command, shell, network-fetch, package-install, file-write or submitted-code execution tool. The registry accepts handler references only to vetted code inside the project's tools package. Implementing a new adapter is a trusted code change and requires review and tests; arbitrary third-party Python is not made safe by this process boundary.

Default per-call deadlines are 2 seconds for arithmetic and 3 seconds for other operations, with output limits of 1 KiB and 32 KiB respectively. Processes have a CPU limit and a monitored 256 MiB RSS limit; Linux also uses an address-space limit. Monitoring can overshoot between 20 ms checks. At most two workers run per executor. An optional shared `ToolBudget` bounds calls, elapsed time and cumulative returned bytes (defaults: 10 calls, 30 seconds, 128 KiB); it never grants permissions or increases tool limits.

A cancellation event returns a safe `cancelled` result. Cancelling the coroutine propagates `CancelledError` after killing/reaping the child. Timeouts, oversized output and resource-limit failures likewise clean up workers; no automatic retries are made. These resource controls currently target macOS/Linux and are **not an OS sandbox for untrusted executable code**. Running generated tests, network connectors, persistent approvals and visual multi-agent workflows remain outside Stage 4A.

### Verification

`tests/test_tools.py` covers registry versioning, strict contracts, grants, real calculations, source parsing without execution, file manifests, traversal, hidden/credential/config paths, symlink/hardlink escapes, FIFO/binary/large files, secret redaction and agent-scoped local knowledge search. `tests/test_tool_execution.py` exercises actual child-process timeouts/cancellation, output and memory limits, environment scrubbing, malformed results, revocation races, result budgets and concurrency. Tests use disposable project roots/databases; they do not read `API.key`, modify other real projects or call paid APIs.

```bash
.venv/bin/python -m pip install -e '.[dev,browser]'
PYTHONDONTWRITEBYTECODE=1 AI_AGENT_PRO_BROWSER_TESTS=1 .venv/bin/python -m pytest -p no:cacheprovider -ra
```


## Phase 4B: controlled AI tool calling

In **Chat → Tools for the next message**, select exact `1.0.0` tools for the active agent. Nothing is selected by default. `project_read` and `project_search` additionally require one eligible project-relative filename per line. Selections remain visible in that session until changed and are copied into each submitted execution. There is no model-supplied grant, principal, version, project root or database path. A cached host authority owns the executor and in-memory grants; another grant/manifest revision during a task invalidates that task. Restarting the application clears grants. Knowledge tools require the bound project database; a custom database outside this project does not acquire knowledge-tool access.

`tool_calling.py` freezes the selected definitions and permission revisions for the execution, maps each version to an API function name (for example `calculator__v1_0_0`), and supplies strict schemas. The provider disables parallel function calls. Only a completed response with one complete function call may execute; streamed calls also require matching argument-done and item-done events. Argument deltas are never displayed or executed. The host rejects unknown names/versions, extra or duplicate JSON fields, invalid types, non-finite values, malformed/deep arguments, duplicate call IDs and parallel batches before starting a worker.

Explicit tool selection requires at least one successful selected-tool call before saving a generated answer. The engine passes `tool_choice="required"` until a real result exists, then `"auto"` for subsequent calls and answer/review stages. A text-only response to a required-call request fails safely without saving an assistant answer or retrying. Selection remains the sole UI approval source; this policy neither grants tools nor widens the selected schemas. With no selection, tools and tool-choice settings are omitted. Document-grounded mode still returns its existing no-evidence response without a model request when retrieval finds no sources.

The engine calls the existing `ToolExecutor`, then sends a JSON `function_call_output` with the original `call_id`. Response output items, including reasoning items when present, are replayed in the execution's in-memory context. SDK-only parsed fields are excluded. All requests retain `store=False`, the fixed API endpoint and zero automatic retries. No provider response IDs or background runs are used. Tool outputs are untrusted data and cannot grant further access. Permissions and knowledge generations/assignments are revalidated before subsequent requests and final persistence. An already in-flight API request cannot be recalled after revocation.

Limits apply to the entire task, including draft/review stages:

- At most **8 model requests**, **5 sequential tool calls**, **64 KiB cumulative tool results** and **128 KiB serialized request context**. One response containing multiple calls is rejected.
- The existing configured token budget, output cap and overall deadline remain in force. Reservations include function schemas, protocol items and returned results. Unknown token usage charges the conservative reservation; reported completed-call usage is counted once in the existing analytics.
- The shared tool budget additionally expires after 60 seconds; shorter per-tool CPU, wall-time, RSS, result-size and concurrency limits still apply. Stop cancels the current API request or tool wait and kills/reaps tool workers and snapshot descendants. A failed tool stops the task safely without retrying or saving a partial assistant message.

**Actual tool activity** displays registry names, exact versions, real statuses and sanitized JSON results for the owning chat session. Knowledge results show the actual source count rather than retaining source excerpts in a second UI cache. SQLite execution events retain tool names, statuses and elapsed times, without arguments, source excerpts, results or private reasoning. Activity's recorded workflow shows these summaries after reload. Transient tool results disappear with the session; final answers follow the existing conversation retention rules. No schema migration is needed, and the existing RAG citation, Markdown, persistence, analytics and dashboard paths are retained.

Security remains bounded: redaction and prompt-injection resistance are best-effort, the local app has no authentication, and resource-limited workers are not an OS sandbox. The earlier manual calculator smoke test is described under Limitations; automated release checks use mocked responses. Phase 4C approvals and Phase 4D–4E sequential multi-agent workflows are implemented and documented below. Network connectors and parallel or branching orchestration remain unavailable.

Offline verification includes real SDK HTTP/SSE mocks, real Stage 4A subprocesses, sequential calls, version/agent isolation, revoked grants, malformed and incomplete calls, token/result/context/request/time limits, cancellation, safe activity/results, knowledge deletion and unchanged RAG citations. Opt-in Chromium cases exercise actual tool execution, sequential activity/results, streamed Markdown, Stop and safe rejection of malformed, incomplete and unapproved calls. The original SQLite path-race and CR-only line-number regressions remain in the complete suite.

Function-call protocol follows the [official OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling).

## Phase 4C: human approvals and safe practice file changes

Chat now includes **Safe practice file changes**. Use **Create example practice file** to explicitly create `practice_workspace/example.txt` with the fixed text `This is a practice file.` and a final newline (existing files are never overwritten). Alternatively, prepare public practice files yourself in that directory. Enable proposals and explicitly select existing files before sending a message. These selections authorize proposals only. They never approve a replacement. The practice directory is ignored by Git.

The sole new model function is `practice_replace__v1_0_0`, with strict `path` and `content` arguments. It proposes a complete replacement of one selected existing file. Each execution supports one proposed file change. A second change requires a new execution and decision. Other selected read-only tools may finish before the proposal; their completed calls are saved in the checkpoint and are not repeated on continuation. No shell, arbitrary Python, directory creation, executable adapter, deletion, rename, or external project modification is exposed to the model.

A proposal durably pauses the original run in `waiting_approval`. The UI shows the full absolute path, complete before/after text, a unified diff (including missing-final-newline markers), hashes, exact tool version, execution/action identity and expiry. **Approve file change**, **Deny file change**, and **Cancel paused execution** are functional human controls. Only Approve permits publication; messages, model output, documents, selected tools and restored widget values cannot supply that decision. The UI renders text and diffs as code, without interpreting embedded markup. Pending decisions survive reload/restart. Runs with unresolved proposals block another execution or deletion of the owning conversation/agent.

Schema version 4 transactionally adds action requests and checkpoints. Each request binds canonical validated arguments, before/after SHA-256 hashes, file/workspace/project identities, agent/run/call identity, exact tool version, a checkpoint digest and a 15-minute expiry. SQLite transactions serialize decisions and single-consumer checkpoint claims. Expired, changed, denied, cancelled, already-decided, foreign, malformed and replayed actions fail closed. Decisions remain in the local application database; it retains proposal before/after text for audit. Recognizable credentials are refused, rather than copied into proposals. Existing `.env`, `API.key`, Streamlit secrets, database and credential ignore rules remain in force. Do not place sensitive data in practice files: redaction is best-effort and this local application has no authentication or encryption at rest.

Only single-level public ASCII basenames ending in `.txt` or `.md` are eligible, with at most 16,000 UTF-8 bytes. Hidden/private/credential names, binary/control data, symlinks, hard links, FIFOs and directories are refused. Descriptor-relative no-follow reads and atomic same-directory replacement keep publication inside the fixed workspace; the source and workspace are rechecked immediately before publication. A nonblocking OS directory lock serializes publication across application processes; a competing approved action fails safely rather than overwriting a concurrently published result. Temporary replacements are exclusively created with mode `0600`, flushed and atomically published. These checks are application safeguards, not an OS sandbox against a malicious local process with equivalent filesystem rights; uncooperative external writers cannot participate in an atomic database/filesystem transaction.

Approval waiting uses the separate 15-minute expiry; it does not consume the remaining active execution-time budget. Approval consumes exactly one action and resumes the original run with its saved context, measured request/token totals, remaining deadline and original configuration. The continuation receives the saved function result and no tools or inferred grants. It can finish the remaining answer/review stages, with streaming and RAG citation validation retained. Knowledge sources are revalidated before publication and model requests; document deletion invalidates and clears affected checkpoint context. Completed/failed continuations clear transient checkpoint context while retaining the action audit. Dashboard filters and charts include waiting approvals.

Publication first records `applying`, then stages/fsyncs/replaces the file and records `applied`. This explicitly handles the gap between SQLite and the filesystem. Activity's deadline-gated recovery can return an interrupted pre-generation checkpoint to the paused state. **Resume approved execution** reconciles an interrupted publication only if the expected post-image is present in the bound workspace. Otherwise it reports an uncertain outcome and never repeats the write. A recorded completed replacement is never written again. A crash after a possibly billable continuation request starts is finalized as interrupted by explicit recovery, without automatically repeating the request. Cancellation before publication leaves the file unchanged; cancellation or generation failure after publication retains the explicitly approved change and reports that fact. Abandoned staging files may remain after an abrupt process kill; they are hidden, unselectable and never executed or trusted by recovery.

Automated approval coverage includes exact diff/decision binding, ownership, manifests, malformed arguments, secret/path/link/file-type rejection, expiry, cancellation, stale files, duplicate/concurrent claims, checkpoint tampering, crash windows, no repeat of completed calculator calls, knowledge deletion, and functional Streamlit/Chromium approval/denial/cancellation/reload flows. All automated SDK traffic is mocked and test files/databases live in disposable test directories. Phase 4D is documented below.

## Phase 4D: versioned sequential multi-agent workflows

**Phase 4D** introduced the sequential execution controls that the Phase 4E visual interface below reuses. Configure five existing agents for **Requirements Analyst → Researcher → Architect → Developer → Reviewer**, select bounded global limits, and save an immutable numbered definition version. The coordinator and handoff protocol are versioned `1.0.0`. The backend accepts 1–8 strictly sequential steps; parallel execution, branching, loops, model-chosen agents and dynamic adapters are not implemented. No new dependencies are required.

Each run freezes its definition, agent instructions/settings and limits in a bound snapshot. Each step gets a fresh, separately owned conversation with no personal chat history. Only that step's configured agent instructions, its role/task, the explicit shared user goal and the previous validated artifact are supplied. Step inputs are bounded host-created JSON envelopes; final outputs must be JSON objects with exactly `schema_version: "1.0.0"`, `summary` (1–1000 characters), `deliverable` (1–6000 characters), and `risks` (up to five 1–300 character strings). Unknown fields, duplicate JSON fields, unsupported versions, malformed/oversized output and incomplete checkpoints stop the workflow before the next agent starts. A RAG step that produces the ordinary insufficient-evidence response stops instead of manufacturing a structured handoff. Model output is untrusted data and cannot alter identity, routing, limits, permissions or approval state.

Select tools and project context files independently for **each step** before **Start workflow** or **Continue workflow**. Everything defaults off, including practice proposals. Separate host-owned authorities bind the selected exact tool versions and file manifests to their respective principals. Existing Chat grants, another workflow step's selections and permissions mentioned inside an artifact are never inherited. RAG and knowledge search query only the current agent's assigned collections. These collection assignments are checked at execution time, not granted by a definition or restored from an old run snapshot.

Context isolation also applies to generated material. The coordinator records host-derived document/generation and project-file provenance for artifacts, without copying raw tool results into handoff metadata. Every downstream recipient must already be assigned **all** referenced active document generations; selected project context files must be explicitly authorised for that recipient and still match their recorded sanitized-content hashes. Provenance is cumulative and rechecked before requests, actions and publication of a completed handoff. If Researcher used a collection not assigned to Architect, the workflow stops before Architect receives that artifact. Assign the same relevant collections to intended recipients beforehand, or run a workflow that does not include restricted context. Project context consent may be given without granting file-reading tools. Knowledge provenance is collected through a fail-closed execution hook, separately from best-effort UI observers. Completed artifact/context checkpoints have integrity digests. These are application isolation and corruption checks, not authentication against a local user who can edit the SQLite database.

SQLite schema **5** adds immutable workflow definitions, run snapshots, ordered step/checkpoint records, child execution links and durable progress events. Back up the existing database with the app stopped before upgrading. Existing schema migrations remain transactional, saved chats/RAG records remain available, and future schemas are refused. Workflow child runs use the existing execution, streaming, citation, usage and Activity/dashboard paths; the Workflows screen reports coordinator progress and global counters separately, without adding them to child-generation totals again. Referenced workflow agents/conversations cannot be independently deleted or repurposed. Ordinary Chat cannot launch or resume a workflow-owned child outside its coordinator.

Defaults are 20 total model requests, 100,000 charged/reserved tokens, 300 seconds of active execution and 10 tool calls. Configurable ceilings are 40 requests, 200,000 tokens, 900 active seconds and 20 tool calls; per-agent limits and the existing sequential-tool limits still apply. Query embedding requests count against global request/token budgets too. Requests and conservative token reservations are persisted before network activity, known usage adjusts reservations, and missing/failed-call usage retains the reservation. Tool-call budgets are charged before any returned function call can execute, including proposed practice replacements. No automatic request retries occur. Active deadlines cover tool waits and approval continuations; human decision/continuation waiting does not replenish consumed budgets or consume active time.

Phase 4C file changes remain exact-action approvals. A practice proposal pauses the whole workflow and presents the original path, before/after text, diff, hashes, expiry and decision controls in **Workflows**. Denial, expiry or cancellation prevents subsequent agents from starting. Approve consumes only that original action and resumes only its owning child through the coordinator/global budgets. It never approves the Reviewer or another step. Replayed approval IDs and approvals belonging to another workflow are rejected atomically before a coordinator claim can start another agent. After the child returns a validated handoff, the workflow stops at a durable `ready` boundary; review the remaining per-step selections and explicitly press **Continue workflow**. No tools are reconstructed from approval records on resumption. The same child's previously authorised project context may be rechecked without restoring reading/writing grants. No automatic file modifications, shell access, arbitrary Python or other-project access are added.

**Cancel workflow** persists cancellation and interrupts an active connection/tool wait, including cancellation from another browser session. Completed steps and already approved/published file changes remain recorded. Leaving the Workflows page or losing the owning UI heartbeat cancels its active worker rather than continuing an unattended paid job. Pending human approvals remain durable. Errors and recorded progress are visible on reload after selecting the retained execution. Workflow selection labels remain stable when statuses change.

**Recover interrupted workflow** is explicitly gated by the persisted lease deadline. Recovery adopts an already completed child from its saved, validated result or resumes a pre-request boundary, without repeating completed steps. An uncertain in-flight model request is marked interrupted and cannot be automatically retried. Phase 4C publication recovery retains its post-image reconciliation/no-repeat guarantees; recovery does not turn an ambiguous file write into a new write approval. Continuation after restart requires fresh explicit step selections; saved definitions/checkpoints are not tool grants. A task that needs to repeat uncertain work must be started as a new, explicit workflow execution. The application remains for trusted local use without authentication or encryption at rest; credentials must not enter workflow goals, definitions, practice files or shared artifacts. Redaction and model instruction boundaries are best-effort.

Automated coverage includes five-specialist sequencing, immutable snapshots/versions, separate histories/principals/tools, structured input/output validation, injection-shaped handoffs, source/file sharing restrictions and revocation, global embedding/generation/tool/token/time limits, durable and cross-session cancellation, single-consumer claims, completed/uncertain-child recovery, checkpoint integrity, exact workflow-bound approvals, and Chromium execution/progress/failure/cancel/Approve/Deny/Continue controls. All test SDK responses are mocked and practice files/databases are disposable. The Phase 4E visual interface is documented below.


## Phase 4E: visual workflow studio

### Six-stage software workflow

The builder defaults to **Requirements Analyst → Researcher → Software Architect → Developer
(AI Software Engineer) → QA Testing Engineer → Reviewer (Code Reviewer)**. Select six distinct
existing agents; Architect and QA must be separate agents. The original five-stage template remains
available. Choosing an existing **Definition to version** suggests its next version; saving inserts
a new immutable definition and never rewrites an earlier definition or execution. No schema migration
is required: schema 5 and engine/handoff version `1.0.0` continue to support both templates.

The six-stage defaults are **8 model requests, 30,000 workflow tokens, 0 tool calls and 300 seconds
of active time**. New six-stage executions snapshot each agent's output cap at no more than
**1,200 tokens**, without changing that agent's Chat settings or earlier run snapshots. Keep agent
internal steps at one initially. A budget is a ceiling, not a target spend or a guarantee of completion.

Developer must include proposed code, proposed tests and relevant acceptance criteria in its bounded
deliverable. QA receives that exact validated handoff, and Reviewer receives QA's validated findings.
Each stage receives only the preceding artifact plus the shared goal; QA should preserve relevant
implementation context for Reviewer. Larger proposals may need smaller goals rather than longer handoffs.

**QA performs static review; no test runner or arbitrary code execution is exposed.** Its deliverable
must start with `Test execution: NOT RUN`. Host instructions forbid execution/passing claims, and a
defensive validator rejects a missing status or common unsupported execution claims before publishing
the handoff or starting Reviewer. The inspector independently labels QA as unexecuted. This text check
is not a general natural-language truth verifier and may conservatively reject quoted or hypothetical
passing claims. Proposed tests are not execution evidence. Existing exact-action file approvals,
separate tool grants and cumulative context/provenance checks still apply. With the initial zero tool
allowance, selecting a required tool/proposal stops before sending its model request; increasing tool
allowances requires a new saved version and fresh per-step selections.

Workflow reservations remain conservative: serialized request bytes (including JSON escaping and
tool schemas), a 1,024 allowance and the response output cap must fit alongside already charged usage.
Known actual usage replaces a successful request's reservation. Unknown/uncertain usage retains the
reservation; rejected reservations do not consume requests or tokens. Budget failures now distinguish
request exhaustion from token shortfalls and report the stage, required reservation and remaining
allowance. No rejected request is sent and no automatic retry occurs. Save a new version and explicitly
start a new execution to change limits; completed steps and the failed execution remain in history.

Offline tests cover both templates, six-agent handoffs, QA truthfulness guards, output caps, immutable
version/history compatibility, budget failures, separate QA/Reviewer grants, human approval and UI
inspection. Opt-in Chromium tests cover both stage counts, execution controls and responsive layouts.

**Workflows** now shows a connected agent-node canvas for the selected immutable definition or retained execution. Each numbered node shows its role, configured agent name and a written/icon status: Pending, Running, Completed, Failed, Cancelled, Awaiting approval, or Interrupted. A definition preview never implies that any work has run. Executions display the frozen agent names from their snapshots, even if an agent is later renamed. Connections follow the saved sequential order; they are not new routes, editable permissions or parallel scheduling.

Use **Inspect step N** with a pointer or keyboard to select a node. The inspector shows its responsibility, validated handoff summary and risks, with expandable deliverable/JSON and execution details. It distinguishes a provisional response from a saved handoff and does not infer success for a pending node. Inspection changes only session selection: it makes no model request, creates no grant, consumes no approval and writes no execution checkpoint. Agent names, roles, risks and summaries are native text. Deliverables reuse the existing safe Markdown renderer; untrusted content is never interpolated into HTML, CSS or JavaScript, and remote Markdown images do not load.

The canvas includes overall validated progress, model requests, charged/reserved tokens, tool calls and active time alongside their existing limits. The token counter retains Phase 4D's reservation semantics and is not a billing total or a promise that another request will fit. Active time is a display-only estimate derived from the existing coordinator lease while running; measured checkpoint time is shown when paused or terminal. Human approval waits are excluded. No displayed estimate participates in budget enforcement. Running retained workflows refresh through the existing fragment mechanism, including read-only views in another session.

**Start workflow**, **Continue workflow**, **Cancel workflow**, deadline-gated recovery and Phase 4C decision controls retain their existing backend handlers and authorisation checks. The visual layer cannot run a node independently, change agent identity, skip a checkpoint, approve a file action, restore tools or bypass a reservation. Per-step permissions remain separate and explicit. Assigning agents and editing limits still creates a new immutable definition version through the existing configuration controls. This phase adds neither drag-and-drop execution changes nor a parallel/branching engine, database migration or dependency.

Desktop nodes are connected left-to-right; narrower screens use a single connected column with the same reading order. Native inspection buttons support Tab and Enter, have visible keyboard focus and at least 44-pixel targets. Status meaning is conveyed by text and symbols as well as border colour. Metric labels wrap on narrow screens, and reduced-motion preferences disable visual transitions. Automated tests cover projections, terminal/pending states, frozen names, full completion progress, read-only inspection, disabled unauthorised continuation, malicious node/handoff rendering, native keyboard navigation, enlarged text, and mobile/tablet/desktop layouts. Existing Chromium Start/Continue/Cancel/Approve/Deny scenarios also exercise the visual states. All automated execution remains mocked and uses disposable databases/files. Release cleanup includes shared analytics date filtering, consistent tool failure envelopes and removal of unused code/imports. These changes retain the existing security checks and execution limits.
