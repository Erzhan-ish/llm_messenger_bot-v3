# Pipeline: путь сообщения от клиента до ответа

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ПУТЬ СООБЩЕНИЯ: КЛИЕНТ → ОТВЕТ                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Telegram
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
║  ③ сессия / эскалация 24ч               RULE         ║
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
║  ⑨ detect_stage_and_action()            RULE (80%)   ║
║     20+ regex: ACK / GREETING / PRICING /            ║
║     HANDOFF / DOCS / BANK_SELECTION ...              ║
║     └─ не сработало → 🔴 LLM: dialog_analyzer        ║
║          llama3.1:8b → JSON stage/action/query_mode  ║
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
║     TF-IDF по KB chunks                              ║
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
║  ┌─ 🔴 LLM (llama3.1:8b, max_tokens=180) ─────────┐  ║
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
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  ⑱ send_bot()                                       ║
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
      │     llama3.1:8b → JSON:                        │
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

## LLM-вызовы (итого до 5 за одно сообщение)

| # | Где | Когда | max_tokens |
|---|-----|-------|-----------|
| 1 | `dialog_analyzer` | ~20% сообщений — правила не сработали | 300 |
| 2 | `llm_extract_missing_slots` | `needs_kb=True` + нет `client_type` + >4 слов | 40 |
| 3 | `llm_objection_reply` | детектирована одна из 5 форм возражения | 120 |
| 4 | `llm_ack_reply` | `intent=ack` + есть `_last_bot_text` | 80 |
| 5 | `render_manager_text` | ответы без жёстких данных в плане | 180 |
| фон | `detect_escalation_signal` | каждое содержательное сообщение | 300 |

### Типичные сценарии

**Тарифы + известный `client_type`:**
→ #5 render только — **1 LLM call**

**Тарифы + `client_type` в сложной форме ("должник-физик"):**
→ #2 slots + #5 render — **2 LLM calls**

**Возражение с контекстом банка:**
→ #3 objection (ранний return, render не вызывается) — **1 LLM call**

**"хм" после показа тарифов:**
→ #4 ack reply — **1 LLM call**

**"хм" + pending follow-up (бот предлагал docs/pricing):**
→ followup_plan → статический render — **0 LLM calls**

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
│   ├── dialog_analyzer.py        — rule-based routing + LLM fallback
│   ├── fact_retriever.py         — TF-IDF KB search, profile builders
│   ├── escalation_detector.py    — LLM escalation signal
│   ├── llm_enrichment.py         — llm_extract_missing_slots, llm_objection_reply, llm_ack_reply
│   └── sales_policy.py           — sales stage tracker
├── knowledge_base/
│   ├── kb.py                     — KnowledgeBase, TF-IDF, search_with_scores
│   └── loader.py                 — singleton loader, settings-based config
└── llm/prompts/manager/
    └── loader.py                 — build_render_prompt, запреты, стиль
```
