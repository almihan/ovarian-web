"""Relation-extraction integration point.

For bring-your-own-key operation, the OpenAI API key should be passed only to
the running job, never written to SQLite or result files, and discarded after
the relation calls complete. A server-owned key can instead be configured with
the Railway ``OPENAI_API_KEY`` variable.
"""
