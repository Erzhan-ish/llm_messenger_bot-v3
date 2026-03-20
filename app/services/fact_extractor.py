import json
import re
from typing import Dict, Any, List, Optional
from app.llm.providers import ask_llm
from app.config import settings
from app.logging import logger

EXTRACTION_PROMPT = """
Ты — экстрактор данных. Твоя задача — извлечь конкретные факты из предоставленного текста базы знаний (KB) в структурированный JSON.

### ТРЕБОВАНИЯ:
1. Если данных нет в тексте, пиши только null. Не придумывай и не подставляй вероятные значения!
2. Если в тексте есть противоречия между фрагментами, укажи их в поле "source_conflicts".
3. Формат JSON:
{{
  "bank": "Название банка или null",
  "client_type": "ФЛ / ЮЛ / ИП / null",
  "product_type": "название продукта (например, РКО) или null",
  "opening_fee": "стоимость открытия (цифра) или null",
  "monthly_fee": "ежемесячная плата (цифра) или null",
  "transfer_fee": "комиссия за перевод или null",
  "cashout_fee": "комиссия за снятие или null",
  "opening_time": "срок открытия или null",
  "visit_required": true / false / null (нужен ли визит),
  "status": "ACTIVE / null",
  "docs": ["список документов"] или [],
  "constraints": ["ограничения"] или [],
  "source_conflicts": ["описание противоречий"] или []
}}

### ТЕКСТ KB:
{kb_text}
""".strip()

def _normalize_facts(data: dict) -> dict:
    """
    Пост-нормализация данных после LLM (Stage 5).
    """
    if not data: return {}
    
    # 1. Сlient Type
    ct = str(data.get("client_type") or "").upper()
    if "ИП" in ct: data["client_type"] = "ИП"
    elif "ЮЛ" in ct or "ООО" in ct: data["client_type"] = "ЮЛ"
    elif "ФЛ" in ct or "ФИЗ" in ct: data["client_type"] = "ФЛ"
    
    # 2. Bank Normalization (Basic)
    bank = str(data.get("bank") or "").lower()
    if "альфа" in bank: data["bank"] = "Альфа-Банк"
    elif "т-банк" in bank or "тинькофф" in bank: data["bank"] = "Т-Банк"
    elif "ткб" in bank: data["bank"] = "ТКБ"
    elif "уралсиб" in bank: data["bank"] = "Уралсиб"
    
    # 3. Numeric extraction (basic)
    for field in ["opening_fee", "monthly_fee"]:
        val = data.get(field)
        if isinstance(val, str):
            nums = re.findall(r"\d+", val.replace(" ", ""))
            if nums: data[field] = int(nums[0])
            elif "беспл" in val.lower() or " 0" in " " + val: data[field] = 0

    return data

def _extract_json(raw: str) -> Optional[dict]:
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None

async def extract_structured_facts(kb_text: str) -> Dict[str, Any]:
    """
    Извлекает структурированные факты из текста KB (Stage 2 Step B).
    """
    if not kb_text or "Данных нет" in kb_text:
        return {}

    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT.format(kb_text=kb_text)},
        {"role": "user", "content": "Извлеки факты из текста выше."}
    ]

    try:
        # Используем аналитическую модель для структурированного вывода
        raw = await ask_llm(messages, model=settings.OLLAMA_ANALYZER_MODEL)
        data = _extract_json(raw)
        return _normalize_facts(data) if data else {}
    except Exception:
        logger.exception("extract_structured_facts failed")
        return {}
