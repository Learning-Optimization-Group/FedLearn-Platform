-- Generative-vs-classification task type for LLM_LORA projects. Null is treated as
-- SEQ_CLASSIFICATION (the pre-existing behavior) by the backend + scripts.
ALTER TABLE projects ADD COLUMN task_type VARCHAR(32);
