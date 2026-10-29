-- Migration 005: coverage gaps — questions the knowledge base couldn't answer.
-- Run against existing databases (init.sql already includes this for fresh setups).

CREATE TABLE IF NOT EXISTS coverage_gaps (
    id              BIGSERIAL   PRIMARY KEY,
    tenant_id       UUID        NOT NULL REFERENCES tenants(tenant_id),
    query           TEXT        NOT NULL,
    top_score       REAL        NOT NULL,
    query_embedding REAL[]      NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_coverage_gaps_tenant_created
    ON coverage_gaps (tenant_id, created_at DESC);
