from datetime import datetime, timedelta

def is_24h_open(last_message_at: datetime) -> bool:
    return datetime.utcnow() - last_message_at < timedelta(hours=24)
