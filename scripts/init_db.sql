-- USERS
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    channel VARCHAR(20) NOT NULL,
    external_user_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (channel, external_user_id)
);

-- SESSIONS
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    last_activity_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_sessions_user_status
ON sessions (user_id, status);

-- MESSAGES
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,
    text TEXT,
    channel VARCHAR(20) NOT NULL,
    external_message_id VARCHAR(128),
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

-- Dedup index
CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_channel_external_id
ON messages (channel, external_message_id)
WHERE external_message_id IS NOT NULL;

-- Optional: ускорение rate-limit
CREATE INDEX IF NOT EXISTS ix_messages_created_at
ON messages (created_at);

CREATE INDEX IF NOT EXISTS ix_messages_role
ON messages (role);
