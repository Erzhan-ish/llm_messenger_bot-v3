"""Deterministic fee calculators. LLM does not compute percentages."""
from __future__ import annotations

LARGE_TRANSFER_THRESHOLD = 30_000_000
LARGE_TRANSFER_RATE = 0.002  # 0.2% при уведомлении за 2 дня + реестр

# (threshold, rate, label)  — None threshold means "else" / last bracket
_ALPHA_FL_SCALE = [
    (150_000, 0.0,   "до 150 000 руб. — без комиссии"),
    (None,    0.005, "свыше 150 000 руб. — 0.5%"),
]

_URALSIB_FL_SCALE = [
    (100_000, 0.0,   "до 100 000 руб. — без комиссии"),
    (None,    0.002, "свыше 100 000 руб. — 0.2%"),
]

_URALSIB_CONTROL_FEE = 150  # за каждую банкротную операцию
_URALSIB_UL_FEE = 35         # за перевод на юрлицо

BANK_FL_SCALES: dict[str, list] = {
    "Альфа-Банк": _ALPHA_FL_SCALE,
    "Уралсиб":    _URALSIB_FL_SCALE,
}

BANK_CONTROL_FEES: dict[str, int] = {
    "Уралсиб": _URALSIB_CONTROL_FEE,
}

BANK_UL_FEES: dict[str, int] = {
    "Уралсиб": _URALSIB_UL_FEE,
}


def _apply_scale(scale: list, amount: int) -> tuple[int, float, str]:
    """Return (fee, rate, rate_label) from a tiered scale."""
    for threshold, rate, label in scale:
        if threshold is None or amount <= threshold:
            return int(amount * rate), rate, label
    return 0, 0.0, ""


def calculate_transfer_fee(
    bank_name: str,
    amount: int | float | None,
    transfer_target: str | None,
    *,
    include_control_fee: bool = True,
) -> dict:
    """Return deterministic fee calculation. Empty dict if not enough data."""
    if not bank_name or amount is None:
        return {}

    amount = int(amount)
    target = transfer_target or ""
    is_large = amount > LARGE_TRANSFER_THRESHOLD

    result: dict = {
        "bank":             bank_name,
        "amount":           amount,
        "transfer_target":  target,
        "calculated_fee":   None,
        "rate":             None,
        "rate_label":       None,
        "control_fee":      None,
        "ul_fee":           None,
        "total_fee":        None,
        "is_large_transfer": is_large,
        "large_transfer_note": (
            "банк нужно предупредить за 2 рабочих дня и предоставить реестр кредиторов "
            "(тогда льготная комиссия 0.2%)"
            if is_large else None
        ),
        "explanation": "",
    }

    if target == "ФЛ":
        scale = BANK_FL_SCALES.get(bank_name)
        if scale:
            fee, rate, label = _apply_scale(scale, amount)
            result["calculated_fee"] = fee
            result["rate"]           = rate
            result["rate_label"]     = label

    elif target == "ЮЛ":
        ul_fee = BANK_UL_FEES.get(bank_name)
        if ul_fee is not None:
            result["ul_fee"]        = ul_fee
            result["calculated_fee"] = ul_fee

    if include_control_fee:
        ctrl = BANK_CONTROL_FEES.get(bank_name)
        if ctrl is not None:
            result["control_fee"] = ctrl

    # total
    base = result["calculated_fee"] or 0
    ctrl = result["control_fee"] or 0
    if base or ctrl:
        result["total_fee"] = base + ctrl

    # explanation string for renderer
    parts: list[str] = []
    if result["rate_label"]:
        parts.append(result["rate_label"])
    if ctrl:
        parts.append(f"плюс {ctrl} руб. контроль банкротной операции")
    result["explanation"] = "; ".join(parts)

    return result


def calculate_extra_fees(bank_name: str) -> dict:
    """Return known extra/additional fees for a bank."""
    result: dict = {
        "bank": bank_name,
        "control_fee": BANK_CONTROL_FEES.get(bank_name),
        "ul_transfer_fee": BANK_UL_FEES.get(bank_name),
        "fl_transfer_free_up_to": None,
    }
    if bank_name == "Уралсиб":
        result["fl_transfer_free_up_to"] = 100_000
    return result
