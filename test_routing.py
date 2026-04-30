"""Routing and fallback tests for plan.txt acceptance criteria."""
import re
import sys
sys.path.insert(0, ".")

from app.processing.slots import extract_runtime_slots, _detect_bank
from app.processing.plan_builder import build_response_plan
from app.processing.renderer import (
    _render_timing_static,
    _render_timing_docs_static,
    _render_specific_bank_static,
    _render_docs_natural,
    _intent_fallback,
)

# --- Regexes mirrored from message_processor ---
_BANK_SELECTION_REQUEST_RE = re.compile(
    r"(банк\s+(подобрать|выбрать|посоветовать|нужен)"
    r"|нужен\s+банк|надо\s+банк|подобрать\s+банк|какой\s+банк"
    r"|где\s+открыть|открыть\s+счет|открыть\s+счёт"
    r"|банк\s+для\s+(физ|физика|физ\s*лица|юр\s*лица|ооо|ип)"
    r"|нужен\s+счет|нужен\s+счёт|счет\s+нужен|счёт\s+нужен)",
    re.I | re.U,
)
_PRICE_QUERY_RE = re.compile(
    r"\b(тариф|тарифы|сколько\s+стоит|стоимость|цена|ценник|ведение|открытие|платёжк|платежк|комисси)\b",
    re.I | re.U,
)
_DOCS_QUERY_RE = re.compile(
    r"\b(документ\w*|что\s+нужно|что\s+понадобится|сканы|паспорт\w*|инн|решение\s+суда)\b",
    re.I | re.U,
)
_TIMING_QUERY_RE = re.compile(
    r"(срок|сроки|как\s+долго|долго\s+ли|за\s+сколько"
    r"|сколько\s+по\s+времени|когда\s+откро|когда\s+будет\s+открыт"
    r"|как\s+быстро|после\s+подачи\s+документ|сколько\s+ждать"
    r"|откроется|открываться\s+будет|откроют)",
    re.I | re.U,
)
_TIRED_ACK_RE = re.compile(
    r"^(да\s+да|ладно|ну\s+ладно|ок\s+ладно|понял\s+ладно|да\s+ладно)",
    re.I | re.U,
)
_CONSENT_SEARCH_RE = re.compile(
    r"(мне\s+подходит|это\s+подходит|(?:всё|все|вроде|как\s+раз)\s+подходит"
    r"|устраивает(?:\s+меня)?|сойд[её]т"
    r"|договорились|начинаем|оформляем|приступаем|по\s+рукам"
    r"|готов\s+открыть|хочу\s+открыть|давайте\s+оформим|готов\s+к\s+оформлению"
    r"|куда\s+(?:оплатить|перевести)|что\s+дальше|готов\s+начать"
    r"|выставляйте\s+счёт|выставляйте\s+счет|отправлю\s+документы|пришлю\s+документы)",
    re.I | re.U,
)


def _route(user_text: str, slots: dict) -> str:
    extract_runtime_slots(user_text, slots)
    _rule_bank = slots.get("bank_name") or slots.get("_last_bank")
    _bank_in_msg = _detect_bank(user_text.lower())
    has_timing = bool(_TIMING_QUERY_RE.search(user_text))
    has_docs_flag = bool(_DOCS_QUERY_RE.search(user_text))
    explicit_bank_selection = bool(slots.get("client_type") and _BANK_SELECTION_REQUEST_RE.search(user_text))
    explicit_pricing = bool(_rule_bank and _PRICE_QUERY_RE.search(user_text))
    explicit_docs = bool(_rule_bank and _DOCS_QUERY_RE.search(user_text))
    explicit_bank_mention = bool(
        _bank_in_msg and slots.get("client_type")
        and (slots.get("_last_mode") or slots.get("_last_bank") or slots.get("bank_name"))
        and not explicit_pricing and not explicit_docs
    )
    consent = bool(_CONSENT_SEARCH_RE.search(user_text))
    tired_ack = bool(_TIRED_ACK_RE.match(user_text) and slots.get("client_type"))

    if consent:
        return "handoff"
    if has_timing and has_docs_flag:
        return "timing_docs"
    if has_timing:
        return "timing"
    if explicit_docs:
        return "docs"
    if explicit_pricing:
        return "specific_bank"
    if explicit_bank_selection:
        return "bank_selection"
    if explicit_bank_mention:
        return "specific_bank"
    if tired_ack:
        return slots.get("_last_mode", "smalltalk")
    return "llm"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_explicit_bank_selection_fl():
    slots = {}
    route = _route("у меня должник физ лицо, надо банк подобрать", slots)
    assert slots.get("client_type") == "ФЛ", f"client_type={slots.get('client_type')}"
    assert route == "bank_selection", f"route={route}"
    print("OK test_explicit_bank_selection_fl")


def test_timing_detection():
    slots = {"client_type": "ФЛ"}
    route = _route("а по срокам как?", slots)
    assert route == "timing", f"route={route}"
    print("OK test_timing_detection")


def test_timing_not_pricing():
    """Timing fallback must mention срок/неделя/два дня, not 'открытие обойдётся'."""
    plan = {
        "action": "answer",
        "intent": "timing",
        "bank": "ТКБ",
        "client_type": "ФЛ",
        "timing_text": "полное открытие счета обычно занимает около недели с момента заявки; после подписания документов в банке открытие занимает до двух дней",
        "items": [],
    }
    text = _render_timing_static(plan)
    assert "открытие обойдётся" not in text, f"unexpected pricing in: {text}"
    assert any(kw in text.lower() for kw in ("срок", "недел", "до двух дней", "двух дн")), \
        f"no timing keyword in: {text}"
    print("OK test_timing_not_pricing:", text[:80])


def test_timing_docs_detection():
    slots = {"client_type": "ЮЛ"}
    route = _route("как долго откроется и какие документы нужны?", slots)
    assert route == "timing_docs", f"route={route}"
    print("OK test_timing_docs_detection")


def test_docs_only_no_pricing():
    """Docs fallback must not include opening_fee or monthly_fee."""
    plan = {
        "action": "answer",
        "intent": "docs",
        "bank": "ТКБ",
        "client_type": "ФЛ",
        "items": [],  # docs mode — no pricing items
        "docs": ["ИНН должника", "список счетов"],
    }
    text = _render_docs_natural("ТКБ", plan["docs"])
    assert "1500" not in text, f"unexpected pricing in: {text}"
    assert "инн" in text.lower() or "счет" in text.lower(), f"no docs info in: {text}"
    print("OK test_docs_only_no_pricing:", text[:80])


def test_handoff_priority():
    slots = {"client_type": "ФЛ", "bank_name": "ТКБ"}
    route = _route("мне подходит, как оформим?", slots)
    assert route == "handoff", f"route={route}"
    print("OK test_handoff_priority")


def test_timeweb_payload_gpt5():
    """gpt-5-mini must use max_completion_tokens, no temperature/top_p."""
    import asyncio
    import httpx
    from unittest.mock import patch, AsyncMock, MagicMock

    from app.llm.providers.timeweb import TimewebProvider, _is_gpt5_model

    assert _is_gpt5_model("gpt-5-mini"), "gpt-5-mini should be detected as gpt-5"
    assert _is_gpt5_model("gpt-5"), "gpt-5 should be detected as gpt-5"
    assert not _is_gpt5_model("deepseek-v3"), "deepseek should not be gpt-5"

    captured = {}

    async def fake_post(url, *, json=None, headers=None):
        captured["payload"] = json
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]
        }
        return mock_resp

    async def run():
        provider = TimewebProvider.__new__(TimewebProvider)
        provider.base_url = "http://mock"
        provider.model = "gpt-5-mini"
        provider.token = "testtoken"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = fake_post
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await provider.generate([{"role": "user", "content": "test"}], max_tokens=200)

    asyncio.run(run())
    payload = captured.get("payload", {})
    assert "max_completion_tokens" in payload, f"missing max_completion_tokens in {payload}"
    assert "max_tokens" not in payload, f"max_tokens should not be present in {payload}"
    assert "temperature" not in payload, f"temperature should not be present in {payload}"
    assert "top_p" not in payload, f"top_p should not be present in {payload}"
    print("OK test_timeweb_payload_gpt5")


def test_timeweb_empty_content_logging():
    """Provider must return '' and log warning for empty content."""
    import asyncio
    from unittest.mock import patch, AsyncMock, MagicMock

    from app.llm.providers.timeweb import TimewebProvider

    async def run():
        provider = TimewebProvider.__new__(TimewebProvider)
        provider.base_url = "http://mock"
        provider.model = "gpt-5-mini"
        provider.token = "testtoken"

        async def fake_post(url, *, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = lambda: None
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            }
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = fake_post
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await provider.generate([{"role": "user", "content": "test"}])

        assert result == "", f"expected empty string, got: {result!r}"

    asyncio.run(run())
    print("OK test_timeweb_empty_content_logging")


def test_intent_fallback_timing():
    plan = {
        "action": "answer",
        "intent": "timing",
        "bank": "ТКБ",
        "timing_text": "около недели с момента заявки, после подписания — до двух дней",
        "items": [],
        "docs": [],
    }
    text = _intent_fallback(plan, {})
    assert "открытие обойдётся" not in text
    assert "срок" in text.lower() or "недел" in text.lower() or "дней" in text.lower()
    print("OK test_intent_fallback_timing:", text[:80])


if __name__ == "__main__":
    tests = [
        test_explicit_bank_selection_fl,
        test_timing_detection,
        test_timing_not_pricing,
        test_timing_docs_detection,
        test_docs_only_no_pricing,
        test_handoff_priority,
        test_timeweb_payload_gpt5,
        test_timeweb_empty_content_logging,
        test_intent_fallback_timing,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    print()
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")
