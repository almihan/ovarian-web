# Relation extraction: completion validation and truncation fixes

## Scope

This update modifies only these existing application files:

- `backend/pipeline/relation_extraction.py`
- `backend/services/relation_executor.py`

It also adds offline regression tests, this change log, a test report, and a
unified diff. No extraction-result cache or cross-run result-reuse mechanism
has been added.

## 1. Reject unfinished or unusable responses before extraction

The complete uploaded project already rejected responses with
`status="incomplete"`. The gateway now requires `status="completed"` before
accessing the SDK's text helper or parsing response JSON.

The check rejects all other statuses, including failed, cancelled, queued,
in_progress, missing, and unknown statuses. It also rejects a response that
claims completion but contains an API error, incomplete details, or an output
item with an explicitly non-completed status. Optional status fields on
reasoning items may be absent or null.

Refusals are rejected even if their explanatory text is absent, or another
output part contains otherwise valid JSON. Missing/whitespace-only text and
malformed JSON are failures, not successful empty extractions. A genuinely
completed, valid `{"triples":[]}` remains an accepted empty extraction.

The errors retain token usage, the incomplete reason, and available response
identifiers/diagnostics. The existing worker records failed attempts and uses
its existing bounded retry policy. A max_output_tokens failure still triggers
the existing increasing-token-budget retry path; limits and retry counts have
not been changed. Refusals and content-filter stops are non-retryable. Exhausted
failures fail the window rather than being written as successful zero-edge
results.

## 2. Validate every returned triple; never silently truncate

`sanitize_triples()` now defaults to `max_triples=None`. It processes the whole
`triples` array, validates entity IDs/directions/predicates, deduplicates, and
uses the existing stable numeric-tag ordering. The old
`raw_triples[:max_triples]` slice has been removed.

An explicitly supplied `max_triples` remains available as a validation guard.
It is checked against the count of valid UNIQUE triples after validation,
deduplication, and sorting. Overflow raises ValueError instead of selecting a
subset. None means unlimited; zero allows only an empty valid set. Negative,
non-integer, and boolean limit values are rejected.

Thus, 60 distinct valid triples remain 60 whether returned in forward or
reverse order. Invalid or duplicate rows at the beginning no longer prevent
valid relations later in the array from being considered. Existing biological
validation and duplicate-removal rules are unchanged.

## Unchanged

The SYSTEM prompt, model selection, reasoning settings, request payload/schema,
token limits, entity processing, relation directions, and application config
are unchanged. Existing prompt-prefix cache routing, normal output artifacts,
and run-only recovery checkpoints/journals are unchanged. No persistent
extraction-result cache was added. New model calls are not made deterministic
by these fixes.

## Apply to an existing installation

Stop the application and back up the two source files above. Replace them with
the updated files at those same paths. The update-only ZIP also includes the
new test file and documentation; it does not contain your data or configuration.
Restart normally from the project root:

```bash
uvicorn backend.main:app --reload
```

Start a new relation extraction run to evaluate the new behavior. Already
exported results are not rewritten. Keep your existing `.env` and data.
The full-project ZIP omits the private `.env`, macOS metadata, and generated
Python/test caches. It retains `.env.example`, the original project source,
and the supplied reference/corpus files. No runtime dependencies were added.

## Tests

Run the targeted tests from the project root in your existing environment
(with pytest installed):

```bash
python -m pytest -q tests/test_relation_rules.py tests/test_relation_response_validation.py
```

The new tests use fake API responses and temporary worker journals; they do
not require an OpenAI API key, construct a real OpenAI client, or call the API.

Results in the editing environment:

| Test selection | Result |
| --- | --- |
| Relation-focused tests, including 56 new cases | 62 passed |
| Original uploaded project, all tests | 33 passed, 7 failed |
| Updated project, all tests | 89 passed, the same 7 failed |

All seven full-suite failures are in the existing
`tests/test_cell_abbreviation_normalization.py`. Their identities match before
and after the patch. Those unrelated annotation-test issues were left
unchanged. See `RELATION_TEST_REPORT.txt` for their names. Python syntax checks
passed for all 67 Python source/test files in the updated project.

No live OpenAI calls or complete interactive web-app runs were performed.
Tests confirm response/error handling and post-processing, not biological
extraction quality or elimination of model-generation variability.

## API reference consulted

Completion/refusal field handling was checked against the OpenAI Responses
reference and Structured Outputs guide:

- https://developers.openai.com/api/reference/python/resources/responses/
- https://developers.openai.com/api/docs/guides/structured-outputs
