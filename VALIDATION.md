# Validation of the updated application

- 42 offline tests passed.
- Python compilation and JavaScript syntax checks passed.
- Modal worker/scheduler definitions import; scheduling remains disabled by default.
- Assembled production-mode app served the landing page and health endpoint.
- Real saved PMIDs 17204523 and 16507264 produced a network with 4 nodes, 2 directed edges, and 2 evidence records without fresh processing credentials.
- Missing PMID handling and selected-result download passed.
- Assembled private updater with mocked processing verified authentication, preview without publication, a one-paper test cap, persistence of one deferred paper, publication into the saved list, and protection against a duplicate trigger.
- The packaged index contains the 2,697 PMIDs from the supplied collections.

Tests did not call PubMed, allocate a Modal GPU, or call OpenAI. Live deployment to Railway/Modal and production model execution were not run.
