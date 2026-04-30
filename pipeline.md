# Pipeline: путь сообщения от клиента до ответа

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ПУТЬ СООБЩЕНИЯ: КЛИЕНТ → ОТВЕТ                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Telegram / WhatsApp / Wazzup
   │  webhook POST
   ▼
┌─────────────────────────────────────────────────────┐
│  enqueue_inbound_message_job()                      │
│  run_after = now + 5 сек  ◄── debounce delay        │
└─────────────────────────────────────────────────────┘
   │  5 сек спустя
   ▼
Worker: fetch_and_lock_jobs()
   │  payload["_job_id"] = job.id
   ▼
╔══════════════════════════════════════════════════════╗
║                 process_message()                    ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  ① дедупликация по message_id           RULE         ║
║  ② rate limit (6 msg / 10 сек)          RULE         ║
║  ③ эскалация 24ч (user_last_escalation) RULE         ║
║  ④ save_message() → DB                              ║
║                                                      ║
║  ⑤ DEBOUNCE                                         ║
║     sleep 2s                                         ║
║     has_newer_queued_job(job_id, user_id)?           ║
║     да → return (сообщение сохранено, ответа нет)    ║
║     нет → продолжаем                                 ║
║                                                      ║
║  ⑥ extract_runtime_slots()              RULE         ║
║     client_type, bank_name, INN,                     ║
║     email, priority, product_type,                   ║
║     client_style (hurried/doubtful/detailed)         ║
║                                                      ║
║  ⑦ detect_state()                       RULE         ║
║     aggressive → escalate + reply                    ║
║     negative / not_interested / later → static reply ║
║                                                      ║
║  ⑧ _detect_objection()                  RULE         ║
║     "дорого" / "подумаю" / "у других дешевле" ...    ║
║     ├─ нет → продолжаем                              ║
║     └─ да  →  🔴 LLM: llm_objection_reply()         ║
║               контекст: банк + тарифы + last_bot_text║
║               fallback: статика из _OBJECTION_REPLIES║
║               → send_bot() → return                  ║
║                                                      ║
║  ⑨ detect_stage_and_action()            LLM-first    ║
║     RULE — только тривиальные случаи:                ║
║       GREETING (чистое приветствие без контента)     ║
║       THANKS (≤5 слов), ACK (≤4 слова), PONDER (хм) ║
║       INTRO (кто вы / бот?), ACTION_SEND (пришлю...) ║
║       HANDOFF (оператор/менеджер/позвоните)          ║
║       READY (мне подходит / оформляем)               ║
║     всё остальное → 🔴 LLM: dialog_analyzer          ║
║          → JSON: stage / action / query_mode /       ║
║                  needs_kb / needs_handoff / conf     ║
║                                                      ║
║  ⑩ LLM slot enrichment                              ║
║     если needs_kb=True                               ║
║        И client_type == None                         ║
║        И len(user_text.split()) > 4                  ║
║     → 🔴 LLM: llm_extract_missing_slots()            ║
║          max_tokens=40, JSON: {client_type, bank}     ║
║          "должник-физик" → ФЛ                        ║
║          fallback: ничего не меняется                 ║
║                                                      ║
║  ⑪ HANDOFF — ранняя эскалация            RULE        ║
║     action=HANDOFF → bridge_text → escalate → return ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║           answer_with_plan()                         ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  ⑫ try_build_followup_plan()            RULE         ║
║     YES_RE (включает "хм", "мм")                     ║
║     _last_expected_followup → docs / pricing?        ║
║     да → followup_plan → render → return             ║
║                                                      ║
║  ⑬ retrieve_facts()                     NO LLM       ║
║     TF-IDF + semantic (hybrid) по KB chunks          ║
║     bank_hints boost ×2.5                            ║
║     client_type / priority фильтрация                ║
║                                                      ║
║  ⑭ build_response_plan()               RULE          ║
║     action: answer / compare /                       ║
║     selection_opening / clarify / handoff ...        ║
║     items, candidates, question_to_ask               ║
║                                                      ║
║  ⑮ ACK intercept                                    ║
║     plan.action=="service" AND intent=="ack"         ║
║     AND _last_bot_text exists?                       ║
║     → 🔴 LLM: llm_ack_reply()                       ║
║          контекст: last_bot_text + bank + client_type║
║          "хм" → "Что смущает — цена или условия?"    ║
║          fallback: None → статика "Понял, продолжаем"║
║                                                      ║
║  ⑯ validate_plan()                     RULE          ║
║     invalid → action=clarify                         ║
║                                                      ║
║  ⑰ render_manager_text()                             ║
║                                                      ║
║  ┌─ STATIC (без LLM) ─────────────────────────────┐  ║
║  │  service / handoff / clarify                    │  ║
║  │  partner_banks                                  │  ║
║  │  selection_opening (1 банк)                     │  ║
║  │  specific_bank с items                          │  ║
║  │  where_answer / timing / contradiction_repair   │  ║
║  └─────────────────────────────────────────────────┘  ║
║                                                      ║
║  ┌─ 🔴 LLM (_render_service_text) ────────────────┐  ║
║  │  service: intent = intro / smalltalk /          │  ║
║  │           no_candidates                         │  ║
║  │  max_tokens=120, перефразирует шаблон            │  ║
║  └─────────────────────────────────────────────────┘  ║
║                                                      ║
║  ┌─ 🔴 LLM (build_render_prompt, max_tokens=180) ──┐  ║
║  │  answer (без items)                             │  ║
║  │  selection_opening (2+ банков)                  │  ║
║  │  compare / pricing_expand / params_explain      │  ║
║  │  Промпт: build_render_prompt()                  │  ║
║  │    данные + жёсткие запреты:                    │  ║
║  │    • только разрешённые банки/числа             │  ║
║  │    • не предлагать ссылки/звонки                │  ║
║  │    • писать данные естественно                  │  ║
║  │  Валидация → fallback если invalid              │  ║
║  │  Near-duplicate → 1 retry                       │  ║
║  └─────────────────────────────────────────────────┘  ║
║                                                      ║
║  ⑱ summarize_dialog()          🔴 LLM (если >12 msg)║
║     max_tokens=120                                   ║
║     сжимает историю в 5-строчный блок               ║
║     подставляется вместо полной истории в промпт    ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  ⑲ send_bot()                                       ║
║     timing delays (3–55 сек, human-like)             ║
║     split → bubbles                                  ║
║     save_message() → DB                              ║
║     OutboundDispatcher.send()                        ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
   │
   ├─ ОСНОВНОЙ ОТВЕТ отправлен клиенту
   │
   └─ asyncio.create_task() → фон (параллельно):
      ┌────────────────────────────────────────────────┐
      │  run_business_analysis()                       │
      │                                                │
      │  re-read свежие slots из DB  ◄─ race condition fix
      │                                                │
      │  🔴 LLM: detect_escalation_signal()            │
      │     → JSON:                                    │
      │     escalate / interest_score / confidence /   │
      │     reason / next_step / client_need           │
      │                                                │
      │  Порог: interest_score ≥ 85, conf ≥ 0.80       │
      │  или 2 из 3 последних ≥ 70 при conf ≥ 0.65     │
      │                                                │
      │  → maybe_escalate() → Bitrix24                 │
      │    re-read slots перед записью                  │
      └────────────────────────────────────────────────┘
```

---

## LLM-вызовы (итого до 7 за одно сообщение)

| # | Где | Когда | max_tokens |
|---|-----|-------|-----------|
| 1 | `dialog_analyzer` | правила не сработали (~80% сообщений) | 300 |
| 2 | `llm_extract_missing_slots` | `needs_kb=True` + нет `client_type` + >4 слов | 40 |
| 3 | `llm_objection_reply` | детектирована одна из 5 форм возражения | 120 |
| 4 | `llm_ack_reply` | `intent=ack` + есть `_last_bot_text` | 80 |
| 5 | `_render_service_text` | `intent` = intro / smalltalk / no_candidates | 120 |
| 6 | `render_manager_text` | ответы без жёстких данных в плане | 180 |
| 7 | `summarize_dialog` | история диалога >12 сообщений | 120 |
| фон | `detect_escalation_signal` | каждое содержательное сообщение | 300 |

### Типичные сценарии

**Тарифы + известный `client_type`:**
→ #1 analyzer (если не тривиальное) + #6 render — **1–2 LLM call**

**Тарифы + `client_type` в сложной форме ("должник-физик"):**
→ #1 analyzer + #2 slots + #6 render — **3 LLM calls**

**Возражение с контекстом банка:**
→ #3 objection (ранний return, render не вызывается) — **1 LLM call**

**"хм" после показа тарифов:**
→ #4 ack reply — **1 LLM call**

**"хм" + pending follow-up (бот предлагал docs/pricing):**
→ followup_plan → статический render — **0 LLM calls**

**Чистое приветствие / ACK / THANKS:**
→ rule → static render — **0 LLM calls**

---

## Провайдеры LLM (`LLM_PROVIDER`)

| Значение | Описание |
|---------|---------|
| `stub` | Заглушка (для разработки без LLM) |
| `ollama` | Локальный Ollama (`OLLAMA_BASE_URL`) |
| `timeweb` | Timeweb Cloud AI (OpenAI-compatible API) |

### Переключение на Timeweb Cloud AI

В `.env`:
```env
LLM_PROVIDER=timeweb
TIMEWEB_AI_TOKEN=<ваш токен>
TIMEWEB_AI_BASE_URL=<base url из личного кабинета>
TIMEWEB_AI_MODEL=deepseek-v3          # модель для render/enrichment
TIMEWEB_AI_ANALYZER_MODEL=deepseek-v3 # модель для analyzer/escalation
```

Маппинг при вызовах: если caller передаёт `model=OLLAMA_ANALYZER_MODEL`,
провайдер автоматически использует `TIMEWEB_AI_ANALYZER_MODEL`.

---

## Слоты (slots) — что сохраняется между сообщениями

| Ключ | Что хранит |
|------|-----------|
| `client_type` | ФЛ / ЮЛ / ИП |
| `bank_name` | выбранный банк |
| `priority_criteria` | price / speed |
| `inn`, `email` | реквизиты клиента |
| `client_style` | hurried / doubtful / detailed |
| `sales_stage` | QUALIFY → SELECT → PRESENT → OBJECTION → READY → HANDOFF |
| `lead_temperature` | info_only → interested → warm → ready |
| `_last_bank` | последний банк в ответе бота |
| `_last_items` | тарифы из последнего ответа |
| `_last_bot_text` | текст последнего ответа |
| `_last_expected_followup` | docs / pricing (для YES/хм follow-up) |
| `_pending_question_type` | client_type / bank_name / priority |
| `_escalation_sent` | флаг: уже эскалировано |
| `_had_consent` | флаг: consent сигнал сохранён при debounce |

---

## Файлы по слоям

```
app/
├── webhooks/telegram.py          — webhook, enqueue
├── jobs/enqueue.py               — 5s delay, _to_payload
├── worker.py                     — fetch jobs, inject _job_id
├── processing/
│   ├── message_processor.py      — main orchestrator
│   ├── slots.py                  — extract_runtime_slots, client_style
│   ├── plan_builder.py           — build_response_plan, followup, objection detection
│   ├── renderer.py               — answer_with_plan, render_manager_text, static renders
│   └── utils.py                  — send_bot, escalation helpers, debounce
├── services/
│   ├── dialog_analyzer.py        — 8 rule shortcuts + LLM fallback (LLM-first)
│   ├── fact_retriever.py         — TF-IDF + hybrid KB search, profile builders
│   ├── escalation_detector.py    — LLM escalation signal
│   ├── llm_enrichment.py         — llm_extract_missing_slots, llm_objection_reply, llm_ack_reply
│   ├── conversation_summary.py   — summarize_dialog (>12 msg)
│   └── sales_policy.py           — sales stage tracker
├── knowledge_base/
│   ├── kb.py                     — KnowledgeBase, TF-IDF + semantic hybrid, search_with_scores
│   └── loader.py                 — singleton loader, settings-based config
└── llm/
    ├── providers/
    │   ├── __init__.py           — ask_llm() — маршрутизатор провайдеров
    │   ├── stub.py               — заглушка (разработка)
    │   ├── ollama.py             — локальный Ollama
    │   └── timeweb.py            — Timeweb Cloud AI (OpenAI-compatible)
    └── prompts/manager/
        └── loader.py             — build_render_prompt, запреты, стиль
```
