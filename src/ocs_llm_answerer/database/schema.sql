-- questions: 题目本体
CREATE TABLE IF NOT EXISTS questions (
    question_hash TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    question_type TEXT,
    options_json TEXT,
    question_raw_json TEXT NOT NULL CHECK (
        json_valid(question_raw_json)
    ),
    last_seen_at_ns INTEGER NOT NULL CHECK (
        last_seen_at_ns > 0
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- answer_cache: 题目答案缓存
CREATE TABLE IF NOT EXISTS answer_cache (
    question_hash TEXT PRIMARY KEY,
    call_hash TEXT NOT NULL,
    answer TEXT NOT NULL,
    explanation TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL CHECK (
        confidence >= 0 AND confidence <= 1
    ),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0 CHECK (
        hit_count >= 0
    ),
    last_hit_at_ns INTEGER CHECK (
        last_hit_at_ns IS NULL OR last_hit_at_ns > 0
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (question_hash)
    REFERENCES questions(question_hash)
    ON DELETE CASCADE
);

-- llm_requests: 每次 LLM 调用流水
CREATE TABLE IF NOT EXISTS llm_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_hash TEXT NOT NULL,
    request_started_at_ns INTEGER NOT NULL CHECK (
        request_started_at_ns > 0
    ),
    request_completed_at_ns INTEGER CHECK (
        request_completed_at_ns IS NULL OR request_completed_at_ns >= request_started_at_ns
    ),
    question_hash TEXT NOT NULL,

    question_raw_json TEXT NOT NULL CHECK (
        json_valid(question_raw_json)
    ),

    adapter TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key_env TEXT NOT NULL,
    model TEXT NOT NULL,

    timeout_seconds REAL NOT NULL CHECK (
        timeout_seconds > 0
    ),
    max_retries INTEGER NOT NULL CHECK (
        max_retries >= 0
    ),

    extra_body TEXT CHECK (
        extra_body IS NULL OR json_valid(extra_body)
    ),

    request_status TEXT NOT NULL CHECK (
        request_status IN ('PENDING', 'SUCCESS', 'FAILURE')
    ),

    response_body_raw TEXT CHECK (
        response_body_raw IS NULL OR json_valid(response_body_raw)
    ),

    error_type TEXT,
    error_message TEXT,
    http_status INTEGER,
    latency_ms INTEGER CHECK (
        latency_ms IS NULL OR latency_ms >= 0
    ),

    input_tokens INTEGER CHECK (
        input_tokens IS NULL OR input_tokens >= 0
    ),
    output_tokens INTEGER CHECK (
        output_tokens IS NULL OR output_tokens >= 0
    ),
    total_tokens INTEGER CHECK (
        total_tokens IS NULL OR total_tokens >= 0
    ),
    cached_tokens INTEGER CHECK (
        cached_tokens IS NULL OR cached_tokens >= 0
    ),

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (question_hash)
    REFERENCES questions(question_hash)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_answer_cache_updated_at
ON answer_cache (updated_at);

CREATE INDEX IF NOT EXISTS idx_answer_cache_call_hash
ON answer_cache (call_hash);

CREATE INDEX IF NOT EXISTS idx_llm_requests_question_started
ON llm_requests (question_hash, request_started_at_ns DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_llm_requests_status_started
ON llm_requests (request_status, request_started_at_ns DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_llm_requests_call_hash
ON llm_requests (call_hash);
