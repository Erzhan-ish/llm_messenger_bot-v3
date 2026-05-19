#!/usr/bin/env python3
"""Import existing Telegram dialogs from personal account into the bot database.

Usage:
    python scripts/import_pyrogram_history.py
    python scripts/import_pyrogram_history.py --messages 50
    python scripts/import_pyrogram_history.py --dialogs 20
    python scripts/import_pyrogram_history.py --dry-run

Options:
    --messages N    Messages to import per dialog (default: 100, 0 = all)
    --dialogs N     Max dialogs to process (default: 0 = all)
    --delay SECS    Sleep between dialogs in seconds (default: 1.0)
    --dry-run       Preview only, do not save to DB
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


async def run(messages_limit: int, dialogs_limit: int, delay: float, dry_run: bool) -> None:
    from pyrogram import Client
    from sqlalchemy import select, func

    from app.config import settings
    from app.storage.db import engine, Base, AsyncSessionLocal
    from app.storage.models import Session, Message
    from app.storage.repositories.users_repo import get_or_create_user
    from app.storage.repositories.sessions_repo import create_new_session

    if not settings.PYROGRAM_API_ID or not settings.PYROGRAM_API_HASH:
        print("ERROR: PYROGRAM_API_ID and PYROGRAM_API_HASH not set in .env")
        sys.exit(1)

    # Ensure DB tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    client = Client(
        name=settings.PYROGRAM_SESSION_NAME,
        api_id=settings.PYROGRAM_API_ID,
        api_hash=settings.PYROGRAM_API_HASH,
    )

    await client.start()
    me = await client.get_me()
    my_id = me.id
    print(f"Logged in as: {me.first_name} (@{me.username or 'no username'}) [id={my_id}]")
    print(f"Mode: {'DRY-RUN' if dry_run else 'IMPORT'} | messages/dialog={messages_limit or 'all'} | delay={delay}s")
    print("=" * 60)

    total_dialogs = 0
    total_new_users = 0
    total_messages = 0

    async for dialog in client.get_dialogs():
        chat = dialog.chat

        # Only private chats
        if chat.type.name != "PRIVATE":
            continue

        # Skip bots
        if getattr(chat, "is_bot", False):
            continue

        total_dialogs += 1
        if dialogs_limit and total_dialogs > dialogs_limit:
            print(f"\nReached dialogs limit ({dialogs_limit}), stopping.")
            break

        name = f"{chat.first_name or ''} {chat.last_name or ''}".strip() or str(chat.id)
        user_id_str = str(chat.id)
        print(f"\n[{total_dialogs}] {name} (id={user_id_str})")

        if dry_run:
            print("  [DRY-RUN] skipped")
            continue

        # Get or create user
        user = await get_or_create_user("telegram_personal", user_id_str)

        # Get or create session
        async with AsyncSessionLocal() as db:
            sess_res = await db.execute(
                select(Session)
                .where(Session.user_id == user.id, Session.status == "active")
                .limit(1)
            )
            sess = sess_res.scalar_one_or_none()

        if not sess:
            sess = await create_new_session(user.id)
            total_new_users += 1
            print(f"  New user → session_id={sess.id}")
        else:
            print(f"  Existing user → session_id={sess.id}")

        # Load already-imported message IDs to avoid duplicates
        async with AsyncSessionLocal() as db:
            existing_ids_res = await db.execute(
                select(Message.external_message_id)
                .where(
                    Message.session_id == sess.id,
                    Message.channel == "telegram_personal",
                )
            )
            existing_ids: set[str] = {r for r, in existing_ids_res.all()}

        # Fetch message history (newest first from Telegram)
        fetch_limit = messages_limit if messages_limit > 0 else None
        to_insert: list[tuple] = []

        async for msg in client.get_chat_history(chat.id, limit=fetch_limit or 10000):
            if not msg.text:
                continue

            ext_id = str(msg.id)
            if ext_id in existing_ids:
                continue

            # Determine role: outgoing = bot reply, incoming = user
            is_outgoing = msg.outgoing or (msg.from_user and msg.from_user.id == my_id)
            role = "bot" if is_outgoing else "user"

            # Strip timezone to match DB column type
            created_at = msg.date.replace(tzinfo=None) if msg.date.tzinfo else msg.date

            to_insert.append((ext_id, role, msg.text, created_at))

        # Reverse to chronological order before inserting
        to_insert.reverse()

        if to_insert:
            async with AsyncSessionLocal() as db:
                for ext_id, role, text, created_at in to_insert:
                    db.add(Message(
                        session_id=sess.id,
                        role=role,
                        text=text,
                        channel="telegram_personal",
                        external_message_id=ext_id,
                        created_at=created_at,
                    ))
                await db.commit()

        imported = len(to_insert)
        total_messages += imported
        print(f"  Imported {imported} messages ({len(existing_ids)} already existed)")

        await asyncio.sleep(delay)

    await client.stop()

    print("\n" + "=" * 60)
    print("Import complete!")
    print(f"  Dialogs processed : {total_dialogs}")
    print(f"  New users created : {total_new_users}")
    print(f"  Messages imported : {total_messages}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Pyrogram history into bot DB")
    parser.add_argument("--messages", type=int, default=100,
                        help="Messages to import per dialog (0 = all, default: 100)")
    parser.add_argument("--dialogs", type=int, default=0,
                        help="Max dialogs to process (0 = all, default: all)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Sleep between dialogs in seconds (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only, do not save to DB")
    args = parser.parse_args()

    asyncio.run(run(
        messages_limit=args.messages,
        dialogs_limit=args.dialogs,
        delay=args.delay,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
