from __future__ import annotations

from backend.pipeline.relation_extraction import (
    BASE_PREDICATES,
    PreparedChunk,
    SYSTEM_INSTRUCTIONS,
    effective_prompt_cache_shards,
    prepare_chunk,
    prompt_cache_key_for_request,
    relation_allowed,
    request_body,
    sanitize_triples,
)


def test_exact_relation_direction_matrix() -> None:
    allowed = {
        ("G1", "activation", "C1"),
        ("H1", "activation", "C1"),
        ("H1", "activation", "G1"),
        ("G1", "inhibition", "C1"),
        ("H1", "inhibition", "C1"),
        ("H1", "inhibition", "G1"),
        ("G1", "proliferation", "C1"),
        ("H1", "proliferation", "C1"),
        ("C1", "secreted", "G1"),
        ("C1", "secreted", "H1"),
        ("H1", "binding", "G1"),
        ("H1", "upregulation", "G1"),
        ("H1", "downregulation", "G1"),
    }
    ids = ("C1", "C2", "G1", "G2", "H1", "H2")
    for predicate in BASE_PREDICATES:
        for subject in ids:
            for object_ in ids:
                expected = (subject, predicate, object_) in allowed
                # The matrix is type-based, so equivalent numeric IDs have the
                # same answer as the canonical C1/G1/H1 triples above.
                canonical = (
                    f"{subject[0]}1",
                    predicate,
                    f"{object_[0]}1",
                )
                expected = canonical in allowed and subject != object_
                assert relation_allowed(subject, predicate, object_) is expected

    assert not relation_allowed("G1", "activation", "G2")
    assert not relation_allowed("C1", "activation", "C2")
    assert not relation_allowed("C1", "secreted", "C2")
    assert not relation_allowed("H1", "secreted", "C1")
    assert not relation_allowed("G1", "secreted", "C1")
    assert not relation_allowed("C1", "production", "H1")
    assert not relation_allowed("G1", "binding", "H1")
    assert not relation_allowed("G1", "unsupported", "H1")


def test_cache_shards_scale_down_for_small_and_retry_windows() -> None:
    assert effective_prompt_cache_shards(0, maximum_shards=32) == 1
    assert effective_prompt_cache_shards(15, maximum_shards=32) == 1
    assert effective_prompt_cache_shards(16, maximum_shards=32) == 2
    assert effective_prompt_cache_shards(100, maximum_shards=32) == 7
    assert effective_prompt_cache_shards(500, maximum_shards=32) == 32


def test_prepare_chunk_adds_cell_gene_and_hormone_tags_and_reuses_ids() -> None:
    text = (
        "Estradiol increased FSHR expression in granulosa cells. "
        "Estradiol then stimulated granulosa cells."
    )
    second_hormone = text.index("Estradiol", 1)
    second_cell = text.index("granulosa cells", text.index("granulosa cells") + 1)
    annotation_row = {
        "base": "paper-a",
        "doc_key": "PMID1",
        "canonical_id": "pmid:1",
        "pmid": "1",
        "section_type": "ABSTRACT",
        "chunk_id": 0,
        "annotations": [
            {
                "obj": "hormone",
                "start": 0,
                "end": len("Estradiol"),
                "mention": "Estradiol",
                "concept_id": "MESH:D004958",
                "hormone_id": "D004958",
            },
            {
                "obj": "gene",
                "start": text.index("FSHR"),
                "end": text.index("FSHR") + len("FSHR"),
                "mention": "FSHR",
                "concept_id": "NCBIGene:2492",
                "gene_id": "2492",
                "tax_id": "9606",
                "tax_name": "Homo sapiens",
                "taxonomy_source": "NCBI Gene ESummary",
            },
            {
                "obj": "cell",
                "start": text.index("granulosa cells"),
                "end": text.index("granulosa cells") + len("granulosa cells"),
                "mention": "granulosa cells",
                "concept_id": "CL:0000501",
            },
            {
                "obj": "hormone",
                "start": second_hormone,
                "end": second_hormone + len("Estradiol"),
                "mention": "Estradiol",
                "concept_id": "MESH:D004958",
                "hormone_id": "D004958",
            },
            {
                "obj": "cell",
                "start": second_cell,
                "end": second_cell + len("granulosa cells"),
                "mention": "granulosa cells",
                "concept_id": "CL:0000501",
            },
        ],
    }
    prepared = prepare_chunk(
        row_index=7,
        source_row={**annotation_row, "chunk": text},
        annotation_row=annotation_row,
    )

    assert prepared.custom_id == "r-0000000007"
    assert prepared.eligible is True
    assert set(prepared.entities) == {"C1", "G1", "H1"}
    assert prepared.tagged_text.count("[H1]") == 2
    assert prepared.tagged_text.count("[C1]") == 2
    assert "[G1]FSHR[/G1]" in prepared.tagged_text
    assert prepared.entities["G1"]["tax_id"] == "9606"
    assert prepared.entities["G1"]["taxonomy_source"] == "NCBI Gene ESummary"
    assert prepared.dropped_overlap_count == 0


def test_gene_and_protein_labels_reuse_one_normalized_g_tag() -> None:
    text = "FSHR protein and FSHR gene were detected in granulosa cells."
    first = text.index("FSHR")
    second = text.index("FSHR", first + 1)
    cell = text.index("granulosa cells")
    annotations = [
        {
            "obj": "protein",
            "start": first,
            "end": first + 4,
            "mention": "FSHR",
            "gene_id": "2492",
            "tax_id": "9606",
            "tax_name": "Homo sapiens",
        },
        {
            "obj": "gene",
            "start": second,
            "end": second + 4,
            "mention": "FSHR",
            "gene_id": "2492",
            "tax_id": "9606",
            "tax_name": "Homo sapiens",
        },
        {
            "obj": "cell",
            "start": cell,
            "end": cell + len("granulosa cells"),
            "mention": "granulosa cells",
            "concept_id": "CL:0000501",
        },
    ]
    prepared = prepare_chunk(
        row_index=0,
        source_row={"chunk": text},
        annotation_row={"annotations": annotations},
    )
    assert set(prepared.entities) == {"C1", "G1"}
    assert prepared.tagged_text.count("[G1]") == 2


def test_sanitizer_enforces_matrix_and_cell_context() -> None:
    entities = {
        "C1": {"id": "C1", "obj": "cell"},
        "C2": {"id": "C2", "obj": "cell"},
        "G1": {"id": "G1", "obj": "gene"},
        "G2": {"id": "G2", "obj": "gene"},
        "H1": {"id": "H1", "obj": "hormone"},
    }
    parsed = {
        "triples": [
            {
                "subject": "H1",
                "predicate": "upregulation",
                "object": "G1",
                "cell_context": ["C2", "C2", "G1", "C9"],
            },
            {
                "subject": "G1",
                "predicate": "activation",
                "object": "C1",
                "cell_context": ["C2"],
            },
            {
                "subject": "G1",
                "predicate": "activation",
                "object": "G2",
                "cell_context": [],
            },
            {
                "subject": "H1",
                "predicate": "secreted",
                "object": "C1",
                "cell_context": [],
            },
            {
                "subject": "C1",
                "predicate": "secreted",
                "object": "H1",
                "cell_context": ["C2"],
            },
            {
                "subject": "H1",
                "predicate": "binding",
                "object": "G1",
                "cell_context": ["C1"],
            },
            {
                "subject": "C1",
                "predicate": "production",
                "object": "H1",
                "cell_context": [],
            },
            {
                "subject": "G1",
                "predicate": "binding",
                "object": "H1",
                "cell_context": ["C1"],
            },
        ]
    }
    cleaned = sanitize_triples(
        parsed,
        entities=entities,
        require_hormone_gene_cell_context=False,
    )

    assert {
        (item["subject"], item["predicate"], item["object"])
        for item in cleaned
    } == {
        ("H1", "upregulation", "G1"),
        ("G1", "activation", "C1"),
        ("C1", "secreted", "H1"),
        ("H1", "binding", "G1"),
    }
    upregulation = next(
        item for item in cleaned if item["predicate"] == "upregulation"
    )
    binding = next(item for item in cleaned if item["predicate"] == "binding")
    assert upregulation["cell_context"] == ["C2"]
    assert binding["cell_context"] == ["C1"]
    assert all(
        item["cell_context"] == []
        for item in cleaned
        if item["predicate"] not in {"upregulation", "binding"}
    )


def test_strict_hormone_gene_context_mode_discards_contextless_relation() -> None:
    entities = {
        "C1": {"id": "C1", "obj": "cell"},
        "G1": {"id": "G1", "obj": "gene"},
        "H1": {"id": "H1", "obj": "hormone"},
    }
    parsed = {
        "triples": [
            {
                "subject": "H1",
                "predicate": "binding",
                "object": "G1",
                "cell_context": [],
            }
        ]
    }
    assert sanitize_triples(
        parsed,
        entities=entities,
        require_hormone_gene_cell_context=True,
    ) == []


def test_online_request_is_compact_structured_and_cache_sharded() -> None:
    prepared = PreparedChunk(
        custom_id="r-0000000123",
        identity={"base": "a", "chunk_id": 1},
        tagged_text="[H1]Estradiol[/H1] increased [G1]FSHR[/G1].",
        entities={
            "H1": {"id": "H1", "obj": "hormone"},
            "G1": {"id": "G1", "obj": "gene"},
        },
        eligible=True,
        valid_annotation_count=2,
        dropped_overlap_count=0,
    )
    cache_key = prompt_cache_key_for_request(
        "ovarian-relations-v4",
        custom_id=prepared.custom_id,
        shard_count=32,
    )
    body = request_body(
        tagged_text=prepared.tagged_text,
        model="gpt-5.4-nano",
        max_output_tokens=1200,
        reasoning_effort="none",
        cache_key=cache_key,
    )

    assert body["model"] == "gpt-5.4-nano"
    assert body["store"] is False
    assert "Entity table" not in body["input"]
    assert prepared.tagged_text in body["input"]
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["additionalProperties"] is False
    predicates = body["text"]["format"]["schema"]["properties"]["triples"][
        "items"
    ]["properties"]["predicate"]["enum"]
    assert "production" not in predicates
    assert "binding" in predicates
    assert "binding: H -> G" in SYSTEM_INSTRUCTIONS
    assert "releases, produces, generates, synthesizes" in SYSTEM_INSTRUCTIONS
    assert "undirected" not in SYSTEM_INSTRUCTIONS.casefold()
    assert len(body["prompt_cache_key"]) <= 64
    assert body["prompt_cache_key"] == cache_key
    # Keep the static prefix large enough for prompt caching without retaining
    # repetitive instructions that add cost and distract the extraction model.
    assert 3_500 <= len(SYSTEM_INSTRUCTIONS) <= 6_000


def test_request_body_excludes_unsupported_predicates() -> None:
    body = request_body(
        tagged_text="[G1]CYP19A1[/G1] affects [H1]estradiol[/H1].",
        model="gpt-5.4-nano",
        max_output_tokens=1200,
        reasoning_effort="none",
        cache_key="cache-key",
    )
    predicates = body["text"]["format"]["schema"]["properties"]["triples"][
        "items"
    ]["properties"]["predicate"]["enum"]
    assert "unsupported" not in predicates
