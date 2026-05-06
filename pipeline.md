1. Общая схема

  Входящее сообщение
         │
         ▼
  [Dedup / Rate limit / /reset]  ← без LLM
         │
         ▼
  [STT Whisper]  ← локальная модель, НЕ API
         │
         ▼
  [Debounce 2s]  ← ждём новых сообщений
         │
         ▼
  [Hard Guard 1: Aggression regex]  ── aggression → send reply + escalate → EXIT
         │
         ▼
  [Hard Guard 2: State detector regex]  ── NEGATIVE/NOT_INTERESTED/LATER → send reply → EXIT
         │
         ▼
  [Hard Guard 3: Consent regex]  ── explicit consent → handoff bridge → escalate → EXIT (0 LLM)
         │
         ▼
  [Hard Guard 4: Identity guard]  ── greeting/identity/confusion → deterministic reply → EXIT (0 LLM)
         │
         ▼
  [Hard Guard 5: Frustration symbols]  ── ???/!!! → confusion reply → EXIT (0 LLM)
         │
         ▼
  [Hard Guard 6: Out-of-domain]  ── льготы/скидки/бонусы → redirect → EXIT (0 LLM)
         │
         ▼  (MAIN PIPELINE)
         ▼
  [1] build_conversation_context()  ← без LLM
         │
         ▼
  [2] retrieve_context_for_brain()  ← TF-IDF + semantic, без LLM API
         │
         ▼
  [3] _maybe_send_pause_phrase()  ← 30% вероятность
         │
         ▼
  [4] ★ LLM #1 — run_conversation_brain()
         │
         ├─ raw="" (Ollama timeout) → _repair_brain_json пропускается → _default_brain_response()
         ├─ JSON parse fail + raw непустой → ★ LLM #2 — _repair_brain_json()
         │
         ▼
  [5] should_force_handoff()  ← regex, без LLM
         │
         ├─ needs_tool = calculate_transfer_fee?
         │       ├─ domain calculator (без LLM)
         │       └─ ★ LLM #2 — run_conversation_brain() с tool_results
         │
         ▼
  [6] action == handoff → escalate → EXIT
         │
         ▼
  [7] validate_reply()  ← детерминировано, без LLM
         │
         └─ invalid → ★ LLM #3 — conversation_brain_repair()
         │
         ▼
  [8] Greeting injection (первый ход)
         │
         ▼
  [9] send_bot()  ← delay + OutboundDispatcher
         │
         ▼
  [10] (опционально, фоново) ★ LLM BG — detect_escalation_signal()

---
2. Количество вызовов LLM на одно сообщение

  ┌─────────────────────────────────────────────────────────┬─────────────┐
  │                        Сценарий                         │ LLM-вызовов │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Обычный вопрос (ответ валиден с первого раза)           │ 1           │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Brain вернул невалидный JSON → JSON-repair              │ 2           │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Brain запросил calculate_transfer_fee → повторный brain │ 2           │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Tool + невалидный JSON во втором вызове                 │ 3           │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Ответ не прошёл валидацию → repair                      │ 2           │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Tool + validation repair                                │ 3           │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ JSON repair + tool + validation repair                  │ 4           │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Ollama timeout → raw="" → JSON-repair пропускается      │ 1 (timeout) │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ + фоновый анализ (ENABLE_BACKGROUND_ANALYSIS=True)      │ +1          │
  ├─────────────────────────────────────────────────────────┼─────────────┤
  │ Максимум (все ветки)                                    │ 4–5         │
  └─────────────────────────────────────────────────────────┴─────────────┘

  По умолчанию ENABLE_BACKGROUND_ANALYSIS=False — фоновый LLM выключен.

  Лимиты токенов (BRAIN budget): 450 для Ollama и Timeweb.
  Escalation budget: 500 (Ollama) / 2000 (Timeweb).
  LLM_TIMEOUT=120 сек (из .env).

  Важно: если raw="" (Ollama network error/timeout), _repair_brain_json пропускается,
  чтобы не тратить второй таймаут. Сразу переходит к _default_brain_response().

---
3. База знаний — механизм поиска

  3.1 Формат хранения

  Источник — текстовый файл с разделителем ---CHUNK---. Каждый чанк начинается с метаданных:

  ---CHUNK---
  [type:pricing][bank:ТКБ][client_type:ФЛ][topic:opening_fee]
  Открытие счёта ФЛ в ТКБ стоит 1500 руб. ...
  ---CHUNK---
  [type:constraint][bank:ALL][topic:realization_card]
  ...

  Каждый чанк парсится в KBChunk dataclass:
  chunk_id, text, source, type, bank, client_type, status, field, topic, value_num, value_text,
  aliases, tags, fact, is_internal, search_text

  3.2 Кэш

  JSON-файл (kb_cache.json), version=5, валидируется SHA256 исходника при старте.
  При несовпадении — перестраивается.

  3.3 TF-IDF поиск (KnowledgeBase.search)

  score = sum(
      tf(term, chunk) * idf(term)
      for term in query_tokens if term in chunk.search_text
  )

  - Токенизация: regex [а-яёa-z0-9]+, lower
  - Нормализация: обрезка русских суффиксов (длина ≥4 — берём первые 4 буквы)
  - IDF с сглаживанием: log((N+1) / (df+1)) + 1

  3.4 Гибридный поиск (search_with_scores)

  Когда доступна библиотека sentence-transformers:

  final_score = 0.40 × tfidf_score + 0.60 × cosine_similarity(query_emb, chunk_emb)

  Эмбеддинги ленивые (lazy-loaded при первом вызове).
  Модель: sentence-transformers/paraphrase-multilingual-mpnet-base-v2

  3.5 retrieve_context_for_brain — стратегия для brain

  Файл: app/services/fact_retriever.py

  # Строит несколько запросов
  queries = [user_text, f"{mentioned_bank} тарифы", domain_keywords]

  # Для каждого запроса — top-5, объединяет все результаты
  # Буст: ×2.5 если chunk.bank == mentioned_bank

  # Сортировка по типу:
  priority = {
      "selection": 0,    # подбор банка
      "availability": 1, # доступность
      "constraint": 2,   # ограничения
      "pricing": 3,      # тарифы
  }

  # Возвращает top 16 фактов
  В run_conversation_brain передаются только первые 8: kb_facts[:8].

---
4. Все промпты

  4.1 BRAIN — системный промпт (BRAIN_SYSTEM_PROMPT)

  Ты — Алексей, менеджер-консультант компании «В плюсе».
  Ты помогаешь арбитражным/финансовым управляющим открывать счета для должников в банкротных процедурах.

  ═══ ПРИВЕТСТВИЕ ═══
  Если memory._introduced=true:
  - не начинай с «Здравствуйте», «Добрый день» или любого приветствия;
  - не представляйся снова;
  - сразу отвечай по сути.
  Только первое сообщение может содержать приветствие и представление.

  ═══ ИДЕНТИЧНОСТЬ ═══
  Держишь роль консультанта Алексея из «В плюсе».
  Если «ты бот?» → не говори «я бот»/«я живой человек» → объясни роль и верни к задаче.

  ═══ БИЗНЕС-ПРАВИЛА ═══
  Ты НЕ открываешь счёт сам. Консультируешь, подбираешь банк, собираешь первичные данные,
  передаёшь старшему менеджеру.

  Явные сигналы готовности → action=handoff, handoff.needed=true:
  «оформляем», «хочу открыть», «давайте открыть», «откроем», «готов начать»,
  «подключите менеджера», «позовите человека», «куда оплатить», «пришлю документы».

  НЕ handoff: «давайте с Уралсибом», «сравним», «а там как?», «с какими сравнить?».

  ═══ СТИЛЬ ═══
  - Пиши коротко: обычно 1–3 предложения.
  - Задавай уточняющий вопрос только если он реально нужен: собрать недостающие данные,
    выбрать банк, уточнить тип клиента, продвинуть к следующему шагу.
  - Если клиент уже дал данные, подтвердил выбор, согласился на оформление, попрощался
    или поблагодарил — вопрос не нужен.
  - Не задавай вопрос ради вопроса.

  Примеры:
    User: "ИНН 1234567890"
    Good: "Принял ИНН, передам менеджеру для проверки."
    Bad:  "Принял ИНН. Что ещё подсказать?"

    User: "Уралсиб подходит, оформляем"
    Good: "Отлично, Уралсиб фиксируем. Подключу старшего менеджера, он доведёт по документам."
    Bad:  "Отлично, Уралсиб фиксируем. Документы подсказать?"

    User: "спасибо, понял"
    Good: "Пожалуйста, обращайтесь."
    Bad:  "Пожалуйста. Что ещё хотите уточнить?"

  ═══ ВОПРОС В КОНЦЕ ═══
  Задавай вопрос только если:
  - не хватает данных для выбора банка или подбора тарифа;
  - нужно уточнить тип клиента (ЮЛ/ФЛ/ИП);
  - нужно продвинуть клиента к следующему шагу.

  Не задавай вопрос, если:
  - клиент дал данные (ИНН, тип, банк);
  - клиент подтвердил выбор банка;
  - клиент готов к оформлению или сказал «оформляем»;
  - клиент попрощался или сказал «спасибо»;
  - action=handoff или stop;
  - answer_contract.question_policy=forbidden.

  Если answer_contract.question_policy=forbidden — заверши ответ утверждением, без вопроса.

  ═══ ПРАВИЛА FACT_PACK ═══
  fact_pack — основной источник истины.
  1. primary fact → используй первым.
  2. _note → следуй инструкции.
  3. _pricing_hint → используй конкретные цифры.
  4. _banks_hint → следуй его инструкции.
  5. answer_contract:
     - must_include: обязательные факты/формулировки.
     - do_not_include: запрещённые слова/фразы.
     - next_question: если задан (не null) и question_policy ≠ forbidden — используй.
     - question_policy: required / optional / forbidden.
  6. scenario_facts → приоритетнее raw_kb_facts.

  ═══ ТАРИФЫ ═══
  Активные банки ЮЛ/ИП: Альфа-Банк (открытие 800, ведение 800), ТКБ (2800/2090), Уралсиб (3500/1600).
  Активные банки ФЛ: ТКБ (открытие 1500, ведение 0), Уралсиб (ведение 0, переводы до 100 тыс. бесплатно).
  На паузе: Т-Банк, МКБ, Росбанк.

  ═══ TOOLS ═══
  - calculate_transfer_fee: точная комиссия. Нужны: bank, amount, recipient. Верни reply=null.
  - search_kb: поиск в базе (редко).

  ═══ ACTION FIELD ═══
  "answer" | "ask_clarification" | "request_data" | "handoff" | "stop"

  ═══ STATE_UPDATE ═══
  Всегда возвращай state_update: active_task, last_bank, last_topic, pending_question,
  last_answer_summary, sales_context (client_type / selected_bank / need / price_priority / ready_to_open).

  ФОРМАТ ВЫВОДА:
  - Верни только JSON по OUTPUT_SCHEMA. Никакого текста вне JSON. Никакого markdown.

  4.2 BRAIN — пользовательское сообщение (_build_user_message)

  <CONTEXT_DO_NOT_COPY>
  {
    "user_message": "<текст клиента>",
    "recent_dialog": [ ...последние 12 реплик... ],
    "memory": {
      "active_task": {...},
      "last_bank": null,
      "last_topic": null,
      "pending_question": null,
      "client_type": null,
      "last_answer_summary": null,
      "sales_context": {...},
      "_introduced": true/false
    },
    "fact_pack": {
      "partner_banks": {"active": [...], "paused": [...], "fl_active": [...], "yul_active": [...]},
      "bank_pricing_yul": [
        {"bank": "Альфа-Банк", "opening_fee": 800, "monthly_fee": 800, "notes": "..."},
        {"bank": "ТКБ", "opening_fee": 2800, "monthly_fee": 2090, "notes": "..."},
        {"bank": "Уралсиб", "opening_fee": 3500, "monthly_fee": 1600, "notes": "..."}
      ],
      "bank_pricing_fl": [
        {"bank": "ТКБ", "opening_fee": 1500, "monthly_fee": 0, "notes": "..."},
        {"bank": "Уралсиб", "opening_fee": null, "monthly_fee": 0, "notes": "..."}
      ],
      "card_rules": {"primary": "...", "secondary": "...", "_note": "..."},
      "advance_opening": {"allowed": true, "fact": "..."},
      "_pricing_hint": "...",   ← только если клиент спросил "подешевле" / тарифы
      "_banks_hint": "...",     ← только если клиент спросил список банков
      "answer_contract": {
        "topic": "debtor_card|partner_banks|bank_selection_fl|identity|account_type_difference|...",
        "must_include": [...],
        "do_not_include": [...],
        "question_policy": "required|optional|forbidden",
        "next_question": "..." | null
      },
      "scenario_facts": "..." | {},   ← плоский текст из KBScenario (после преобразования)
      "scenario_matches": [...]
    },
    "raw_kb_facts": [ ...до 8 чанков из KB... ],
    "tool_results": null | {"calculate_transfer_fee": {"calculated_fee": 150, ...}}
  }
  </CONTEXT_DO_NOT_COPY>

  <OUTPUT_SCHEMA>
  {
    "reply": "текст клиенту или null",
    "action": "answer|ask_clarification|request_data|handoff|stop",
    "needs_tool": {"name": "calculate_transfer_fee|search_kb|none", "args": {}},
    "state_update": {
      "active_task": {}, "last_bank": null, "last_topic": null,
      "pending_question": null, "last_answer_summary": null,
      "sales_context": {"client_type": null, "selected_bank": null,
                        "need": null, "price_priority": null, "ready_to_open": false}
    },
    "handoff": {"needed": false, "reason": null},
    "stop": false,
    "confidence": 0.0
  }
  </OUTPUT_SCHEMA>

  4.3 JSON-repair промпт (_REPAIR_JSON_PROMPT)

  Применяется только если raw непустой (raw != ""). Если raw="" — repair пропускается.

  Системный промпт:
    Ты вернул невалидный JSON или скопировал входной контекст.
    Верни только один JSON-объект по схеме: {reply, action, needs_tool, state_update, handoff, stop, confidence}.
    Не копируй user_message, recent_dialog, memory, kb_facts, fact_pack.
    Никакого markdown, никакого текста вне JSON.

  Пользовательское сообщение:
    Исправь: <первые 3000 символов невалидного raw>

  4.4 Text-repair промпт (REPAIR_PROMPT)

  Системный промпт:
    Ты — менеджер-консультант «В плюсе». Твой предыдущий ответ содержит ошибку.
    Исправь ответ, используя только предоставленные fact_pack и tool_results.
    Не придумывай цифры. Если данных нет — напиши, что уточнишь.

  Правила исправления:
  - promised_action_without_handoff: не пиши "откроем" — скажи "поможем оформить, нужны данные".
  - wrong_topic_fact / repeated_intro: перепиши без запрещённых фраз.
  - missing_primary_fact: добавь нужный факт из fact_pack.answer_contract.must_include.
  - repeated_intro: убери приветствие, начни сразу по сути.
  - did_not_explain_reason: объясни причину простыми словами, не повторяй прошлый ответ.
  - unnecessary_question: question_policy=forbidden — убери вопрос в конце.
  - missing_required_question: question_policy=required — добавь вопрос из next_question.
  - Соблюдай fact_pack.answer_contract.do_not_include.

  Верни только исправленный текст ответа, без JSON и без объяснений.

  Пользовательское сообщение:
  {
    "previous_reply": "<текст предыдущего ответа>",
    "error": "<код ошибки или расширенное описание>",
    "user_message": "<текст клиента>",
    "fact_pack": {...},
    "raw_kb_facts": [...до 6 чанков...],
    "tool_results": null | {...},
    "memory": {...}
  }

  4.5 Escalation detector промпт (SYSTEM_PROMPT)

  Ты — классификатор эскалации. Ты НЕ пишешь клиенту.

  Эскалация НУЖНА (escalate=true, score>=85, confidence>=0.85), если:
  1) Клиент явно готов: "оформляем", "что дальше?", "куда оплатить?", "хочу открыть счет"
  2) Просит человека/звонок: "подключите менеджера", "оператор"
  3) Конфликт/агрессия
  4) Сложный случай, KB не ответила

  ЖЁСТКОЕ ПРАВИЛО: если next_step != "handoff_manager" → escalate=false.

  Верни строго JSON:
  { "escalate": true|false, "reason": "...", "interest_score": 0..100,
    "confidence": 0..1, "next_step": "none|ask_clarify|handoff_manager",
    "client_need": "OPEN_ACCOUNT|CONDITIONS|DOCUMENTS|CONSULTATION|SUPPORT|UNKNOWN",
    "reasons": [...] }

---
5. Детальные шаги пайплайна

  Шаг 0 — Pre-flight (без LLM)

  ┌───────────────────────────┬────────────────────────────┬────────────────────────────┐
  │         Проверка          │            Код             │ Действие при срабатывании  │
  ├───────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Дубль сообщения           │ is_duplicate_message()     │ return                     │
  ├───────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Rate limit (6 сообщ/10с)  │ check_rate_limit()         │ return                     │
  ├───────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Команда /reset            │ == "/reset"                │ сбросить сессию            │
  ├───────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ STT (аудио)               │ transcribe_audio()         │ локальный Whisper          │
  ├───────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Debounce 2s               │ has_newer_queued_job()     │ пропустить, обновить слоты │
  ├───────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ После эскалации (24ч)     │ get_user_last_escalation() │ подавить ответ             │
  ├───────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ _escalation_sent в слотах │ slots.get()                │ подавить ответ             │
  └───────────────────────────┴────────────────────────────┴────────────────────────────┘

  Шаг 1 — Жёсткие охранники (без LLM)

  # Hard Guard 1: Aggression (PROFANITY_RE)
  if _is_aggressive(user_text):
      send("предупреждение")
      maybe_escalate(reason="aggression_profanity")
      return

  # Hard Guard 2: State detector (regex, state_detector.py)
  state = detect_state(user_text)
  # → AGGRESSIVE / NEGATIVE / NOT_INTERESTED / LATER / IN_PROGRESS

  # Hard Guard 3: Explicit consent (CONSENT_HARD_RE) — fast-path без LLM
  if _CONSENT_HARD_RE.search(user_text):
      bridge = _build_handoff_bridge(slots, "ready_to_open")
      send(bridge)
      maybe_escalate(reason="ready_to_open")
      return
  _CONSENT_HARD_RE покрывает: мне подходит, устраивает, договорились, начинаем, оформляем,
  хочу открыть, что дальше, пришлю документы, подключите менеджера и др.

  # Hard Guard 4: Identity guard (identity_guard.py) — без LLM
  identity_response = check_identity_guard(user_text, memory)
  if identity_response:
      send(identity_response["reply"])
      return
  Покрывает: приветствия, «ты бот?», путаницу, «зачем дважды имя?»

  # Hard Guard 5: Frustration symbols (????, !!!)
  if _FRUSTRATION_ONLY_RE.match(user_text):
      send(_CONFUSION_REPLY)
      return

  # Hard Guard 6: Out-of-domain (льготы, скидки, бонусы)
  if _OUT_OF_DOMAIN_RE.search(user_text):
      send(_OUT_OF_DOMAIN_REPLY)
      return

  Шаг 2 — Сборка контекста (без LLM)

  ctx = await build_conversation_context(user_text, session_id, slots)

  Возвращает:
  - recent_dialog — последние 14 реплик из БД
  - memory — слоты: active_task, last_bank, last_topic, pending_question, client_type,
    sales_context, _introduced
  - current_entities — regex: mentioned_bank, mentioned_amount, mentioned_client_type,
    mentioned_recipient
  - fact_pack — детерминированный пакет фактов по теме запроса

  fact_pack формируется на основе ключевых слов в user_text:
  - Всегда: partner_banks, тарифы ЮЛ/ФЛ (по known client_type или оба), card_rules, advance_opening
  - Если "карт": card_rules.primary = правило реализации, question_policy=required, next_question задан
  - Если "наличн": card_rules.primary = правило наличных
  - Если "банк/список/партнёр": _banks_hint, question_policy=required, next_question задан
  - Если "подешевле/дешевле": _pricing_hint = сравнение по открытию+ведению
  - Если "идентификация/путаница": question_policy=forbidden, next_question=null
  - Всегда: answer_contract с topic / must_include / do_not_include / question_policy / next_question

  answer_contract.question_policy:
    required  → бот ДОЛЖЕН задать вопрос (или явный запрос данных)
    optional  → вопрос допустим, но не обязателен (по умолчанию)
    forbidden → бот НЕ ДОЛЖЕН заканчивать ответ вопросом

  ┌──────────────────────────┬──────────────────┬──────────────────────────────────────┐
  │          Тема            │ question_policy  │           next_question              │
  ├──────────────────────────┼──────────────────┼──────────────────────────────────────┤
  │ debtor_card              │ required         │ "Реализация уже введена?"            │
  ├──────────────────────────┼──────────────────┼──────────────────────────────────────┤
  │ partner_banks            │ required         │ "Счёт подбираем для юрлица или       │
  │                          │                  │  физлица?"                           │
  ├──────────────────────────┼──────────────────┼──────────────────────────────────────┤
  │ bank_selection_fl        │ optional         │ null                                 │
  ├──────────────────────────┼──────────────────┼──────────────────────────────────────┤
  │ identity                 │ forbidden        │ null                                 │
  ├──────────────────────────┼──────────────────┼──────────────────────────────────────┤
  │ account_type_difference  │ optional         │ null                                 │
  ├──────────────────────────┼──────────────────┼──────────────────────────────────────┤
  │ (базовый контракт)       │ optional         │ null                                 │
  └──────────────────────────┴──────────────────┴──────────────────────────────────────┘

  Шаг 3 — KB поиск (без LLM API)

  kb_facts = await retrieve_context_for_brain(user_text, memory, current_entities)

  Внутри:
  1. Строит 2–4 поисковых запроса: user_text, f"{bank} тарифы", domain keywords
  2. Для каждого — kb.search_with_scores(query, top_k=5)
  3. Применяет буст ×2.5 для чанков с chunk.bank == mentioned_bank
  4. Дедупликация по chunk_id
  5. Сортировка: selection/availability → constraint → pricing → другие
  6. Возвращает top 16, в brain передаёт [:8]

  Шаги 4–9 — Brain + Tool + Validate (LLM)

  # LLM #1
  brain_result = await run_conversation_brain(user_text, recent_dialog, memory, kb_facts, fact_pack=fact_pack)

  # JSON-repair (только если raw непустой) — LLM #2
  if raw == "":
      result = _default_brain_response()   # пропустить repair, сэкономить 2 мин
  elif json_parse_fail:
      result = await _repair_brain_json(raw) or _default_brain_response()

  # Tool
  if brain_result["needs_tool"]["name"] == "calculate_transfer_fee":
      fee_result = calculate_transfer_fee(bank, amount, recipient)  # domain calc, без LLM
      brain_result = await run_conversation_brain(..., tool_results=fee_result)  # LLM #2

  # Force handoff (code override)
  if should_force_handoff(user_text, brain_result, memory):
      brain_result["action"] = "handoff"

  # Handoff
  if action == "handoff" and consent_confirmed:
      send(bridge_reply)
      maybe_escalate()
      return

  # Validate (детерминировано)
  val = validate_reply(reply, brain_result, entities, slots, tool_results, user_text, answer_contract)
  if not val["is_valid"]:
      repaired = await conversation_brain_repair(...)  # LLM #3
      if repaired valid:
          reply = repaired
      else:
          reply = _build_context_fallback(...)  # детерминировано, с учётом question_policy

  # Greeting injection (первый ход)
  if not slots["_introduced"] and turn_count <= 1:
      reply = "Здравствуйте! Я Алексей, менеджер «В плюсе». " + reply

  Шаг 10 — Отправка

  send_bot() разбивает ответ на "пузыри" (≤300 символов), добавляет задержки:
  - 1-й пузырь: target_delay - elapsed_since_start (min 1.0с)
  - следующие: inter_bubble_delay = part_len / 20, от 3 до 8с

  Шаг 11 — Фоновый анализ (если включён)

  if ENABLE_BACKGROUND_ANALYSIS:
      asyncio.create_task(run_business_analysis(...))
      # внутри: detect_escalation_signal() → LLM BG
      # если escalate=true AND confidence>=0.80 AND score>=85 → maybe_escalate()

---
6. Что делает validate_reply (детерминировано, без LLM)

  Проверки в порядке выполнения:

  ┌──────────────────────────────────┬───────────────────────────────────────────────────────┬──────────────────────────────────────────────┐
  │            Проверка              │                       Условие                         │              Причина отказа                  │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Пустой ответ                     │ reply is None or ""                                   │ empty_reply                                  │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ CJK-символы                      │ regex [一-鿿...] в ответе                             │ non_russian_output                           │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Дубль предыдущего                │ Jaccard ≥ 80% с _last_bot_text                        │ near_duplicate_of_previous                   │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ "Почему?" после дубля            │ _WHY_RE match                                         │ did_not_explain_reason                       │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Повторное представление          │ _introduced и ответ начинается с "Здравствуйте"       │ repeated_intro                               │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Комиссия из tool_result          │ fee не найден в ответе числами                        │ fee_{N}_missing_from_reply                   │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Неверный банк                    │ чужой банк в ответе без упомянутого                   │ wrong_bank_X_instead_of_Y                    │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Handoff без согласия             │ handoff.needed и нет _had_consent                     │ handoff_without_explicit_consent             │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Обещание действия                │ _PROMISE_ACTION_RE (откроем, приступим)               │ promised_action_without_handoff              │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Открытие без handoff             │ _OPEN_ACCOUNT_SOFT_RE в user_text                     │ open_account_without_handoff_or_request_data │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ question_policy=forbidden        │ reply заканчивается "?"                               │ unnecessary_question                         │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ question_policy=required         │ нет "?" И нет явного запроса данных                  │ missing_required_question                    │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ answer_contract: тема карты      │ "наличные" в ответе при topic=debtor_card             │ wrong_topic_fact                             │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ answer_contract: список банков   │ тарифные цифры при topic=partner_banks                │ answered_tariffs_when_asked_bank_list        │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ answer_contract: ФЛ банк         │ нет "ткб" в ответе при topic=bank_selection_fl        │ missing_primary_fact                         │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ do_not_include                   │ запрещённая фраза найдена                             │ forbidden_phrase_...                         │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ scenario_facts: generic          │ "как я могу помочь" и т.п. при наличии фактов         │ generic_reply_despite_scenario_facts         │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ scenario_facts: карта            │ нет "реализация"/"финансовый управляющий"             │ missing_required_card_facts                  │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ scenario_facts: банки            │ нет альфа-банк/ткб/уралсиб при topic=partner_banks    │ missing_required_partner_banks               │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Слишком короткий                 │ специфический вопрос + len(reply) < 40               │ too_short_for_specific_question              │
  └──────────────────────────────────┴───────────────────────────────────────────────────────┴──────────────────────────────────────────────┘

---
7. Контекстный fallback (_build_context_fallback)

  Вызывается когда brain вернул пустой ответ или repair не помог.
  Использует fact_pack, scenario_facts, current_entities, slots.
  Учитывает answer_contract.question_policy: если forbidden → не добавляет вопрос.

  Приоритет:
  A) debtor_card (question_policy=required) → "...Реализация уже введена?"
  B) account_type_difference               → объяснение разницы + уточняющий вопрос
  C) partner_banks (question_policy=required) → список банков + "Счёт для юрлица или физлица?"
  D) bank-specific pricing               → тарифы банка (вопрос добавляется если policy≠forbidden)
  E) generic                              → "Секунду, уточняю информацию. Уточните, пожалуйста, вопрос."

---
8. answer_contract — полная структура

  {
    "must_answer_current_question": true,
    "do_not_repeat_intro": true,
    "do_not_repeat_previous_answer": true,
    "style": "natural_short",
    "question_policy": "required|optional|forbidden",
    "next_question": "текст вопроса" | null,
    "topic": "debtor_card|partner_banks|bank_selection_fl|identity|account_type_difference|...",
    "must_include": [...],
    "should_include": [...],
    "do_not_include": [...]
  }

  question_policy управляет тем, нужен ли вопрос в конце ответа:
  - required:  validator ругается если вопроса нет
  - optional:  вопрос допустим, но не обязателен (по умолчанию)
  - forbidden: validator ругается если вопрос есть

---
9. Acceptance criteria (из plan.txt)

  ✓ Бот не задаёт вопрос после "спасибо", "понял", "ИНН ...", "оформляем"
  ✓ next_question может быть null
  ✓ question_policy управляет наличием вопроса
  ✓ Validator блокирует лишний вопрос при forbidden
  ✓ Validator требует вопрос при required
  ✓ Когда данных реально не хватает — бот задаёт уточняющий вопрос
