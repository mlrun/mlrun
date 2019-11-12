CREATE TABLE IF NOT EXISTS logs (
    uid TEXT,
    project TEXT,
    body TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    name TEXT,
    uid TEXT,
    start_time TIMESTAMP,
    project TEXT,
    state TEXT
);

CREATE TABLE IF NOT EXISTS run_labels (
    uid TEXT, -- FIXME: Check primary key
    label TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    key TEXT,
    project TEXT,
    tag TEXT,
    uid TEXT,

    body  BLOB
);

CREATE TABLE IF NOT EXISTS artifact_labels (
    uid TEXT, -- FIXME: Check primary key
    label TEXT
);

CREATE TABLE IF NOT EXISTS functions(
    name TEXT,
    project TEXT,
    tag TEXT,

    body BLOB
);

CREATE TABLE IF NOT EXISTS function_lables (
    name TEXT, -- FIXME: Check primary key
    project TEXT,
    tag TEXT,
    label TEXT
);
