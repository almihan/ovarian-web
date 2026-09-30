"use strict";

const STORAGE_KEY = "ovarianNetworkPipelineStateV2";
const POLL_MS = 1500;
const REQUEST_TIMEOUT_MS = 30000;
const PUBLIC_PRECOMPUTED_ONLY = document.body.dataset.publicPrecomputedOnly === "true";
const STAGES = ["retrieval", "annotation", "relation", "network"];
const STAGE_LABELS = {
    retrieval: "Retrieval",
    annotation: "Entity extraction",
    relation: "Relation extraction",
    network: "Network generation",
};

const state = {
    runId: null,
    run: null,
    pollGeneration: 0,
    activeStage: "retrieval",
    workflowRank: 0,
    navigateWhenNetworkReady: false,
    toastTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const sleep = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

async function requestJson(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(
        () => controller.abort(),
        options.timeoutMs || REQUEST_TIMEOUT_MS,
    );
    const fetchOptions = { ...options };
    delete fetchOptions.timeoutMs;
    try {
        const response = await fetch(url, {
            ...fetchOptions,
            cache: "no-store",
            signal: controller.signal,
            headers: {
                Accept: "application/json",
                ...(fetchOptions.body ? { "Content-Type": "application/json" } : {}),
                ...(fetchOptions.headers || {}),
            },
        });
        let payload = null;
        try {
            payload = await response.json();
        } catch {
            payload = null;
        }
        if (!response.ok) {
            const detail = payload?.detail;
            const message = Array.isArray(detail)
                ? detail.map((item) => item?.msg || String(item)).join(" ")
                : detail || `Request failed with status ${response.status}.`;
            throw new Error(message);
        }
        return payload;
    } catch (error) {
        if (error?.name === "AbortError") {
            throw new Error("The server did not respond in time.");
        }
        throw error;
    } finally {
        window.clearTimeout(timeout);
    }
}

function formatDuration(value) {
    const seconds = Math.max(0, Math.round(Number(value) || 0));
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
    return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}

function formatCount(value) {
    return Number(value || 0).toLocaleString();
}

function showToast(message) {
    const toast = $("#toast");
    const label = $("#toastMessage");
    if (!toast || !label) return;
    label.textContent = message;
    toast.hidden = false;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => {
        toast.hidden = true;
    }, 3800);
}

function normalizeStatus(value) {
    const status = String(value || "ready").trim().toLowerCase();
    const aliases = {
        complete: "completed",
        done: "completed",
        success: "completed",
        succeeded: "completed",
        running: "processing",
        pending: "queued",
    };
    const normalized = aliases[status] || status;
    if (["ready", "locked", "queued", "processing", "completed", "failed"].includes(normalized)) {
        return normalized;
    }
    return "ready";
}

function humanizeStage(value, status) {
    if (!value) {
        if (status === "processing") return "Working";
        if (status === "locked") return "Waiting";
        return status === "completed" ? "Complete" : "Ready";
    }
    return String(value)
        .replace(/^(default|custom|pmid)_/, "")
        .replaceAll("_", " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function stageStateText(status, progress) {
    if (status === "locked") return "Waiting";
    if (status === "queued") return "Queued";
    if (status === "processing") return `Running ${Math.round(progress)}%`;
    if (status === "completed") return "Complete";
    if (status === "failed") return "Attention";
    return "Ready";
}

function readSession() {
    try {
        const raw = window.sessionStorage.getItem(STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : null;
        return parsed && typeof parsed === "object" ? parsed : null;
    } catch {
        return null;
    }
}

function saveSession() {
    if (!state.runId) return;
    const payload = {
        runId: state.runId,
        activeStage: state.activeStage,
        query: $("#queryInput")?.value || state.run?.query || "",
        textMode: $("#textModeSelect")?.value || state.run?.text_mode || "fulltext",
        corpusId: $("#corpusSelect")?.value || (state.run?.input_mode === "precomputed" ? state.run?.corpus_id : "") || "",
    };
    try {
        window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
        // The run remains usable in the current page even when browser storage is blocked.
    }
}

function clearSession() {
    try {
        window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
        // Ignore browser-storage restrictions.
    }
}

function stripResumeParameter() {
    const url = new URL(window.location.href);
    if (!url.searchParams.has("resume")) return;
    url.searchParams.delete("resume");
    const query = url.searchParams.toString();
    window.history.replaceState({}, "", `${url.pathname}${query ? `?${query}` : ""}${url.hash || "#pipeline"}`);
}

function stageIsUnlocked(stage) {
    const status = normalizeStatus(state.run?.stages?.[stage]?.status || (stage === "retrieval" ? "ready" : "locked"));
    return status !== "locked";
}

function activateStage(stage, { scroll = false, focus = false, force = false } = {}) {
    if (!STAGES.includes(stage)) return;
    if (!force && !stageIsUnlocked(stage)) return;
    state.activeStage = stage;
    $$('[data-stage-target]').forEach((tab) => {
        const active = tab.dataset.stageTarget === stage;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
    });
    $$('[data-stage-panel]').forEach((panel) => {
        const active = panel.dataset.stagePanel === stage;
        panel.hidden = !active;
        panel.classList.toggle("is-active", active);
    });
    if (focus) $(`[data-stage-target="${stage}"]`)?.focus({ preventScroll: true });
    if (scroll) $("#stageWorkspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
    saveSession();
}

function initializeNavigation() {
    $("#pipelineStepper")?.addEventListener("click", (event) => {
        const tab = event.target.closest("[data-stage-target]");
        if (!tab || tab.dataset.status === "locked") return;
        activateStage(tab.dataset.stageTarget, { scroll: window.innerWidth <= 820 });
    });
    $("#pipelineStepper")?.addEventListener("keydown", (event) => {
        const tab = event.target.closest("[data-stage-target]");
        if (!tab) return;
        const unlocked = STAGES.filter(stageIsUnlocked);
        const current = unlocked.indexOf(tab.dataset.stageTarget);
        if (current < 0) return;
        let next = current;
        if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (current + 1) % unlocked.length;
        else if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (current - 1 + unlocked.length) % unlocked.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = unlocked.length - 1;
        else return;
        event.preventDefault();
        activateStage(unlocked[next], { focus: true });
    });
}

function setError(stage, message = "") {
    const element = $(`#${stage === "retrieval" ? "formError" : `${stage}Error`}`);
    if (!element) return;
    element.textContent = message;
    element.hidden = !message;
}

function updateReadiness(stage, status, message, error) {
    const wrapper = $(`#${stage}Readiness`);
    const title = $(`#${stage}ReadinessTitle`);
    const text = $(`#${stage}ReadinessText`);
    if (!wrapper || !title || !text) return;
    wrapper.dataset.state = status;

    if (status === "locked") {
        title.textContent = stage === "annotation" ? "Waiting for Stage 1" : stage === "relation" ? "Waiting for Stage 2" : "Waiting for Stage 3";
        text.textContent = stage === "annotation"
            ? "Entity extraction starts automatically after retrieval."
            : stage === "relation"
                ? "Relation extraction starts automatically after entity extraction."
                : "Use View network after relation extraction completes.";
        return;
    }
    if (status === "ready") {
        title.textContent = stage === "retrieval" ? "Ready to retrieve" : `${STAGE_LABELS[stage]} is ready`;
        text.textContent = stage === "retrieval"
            ? (PUBLIC_PRECOMPUTED_ONLY
                ? "Select a saved subject topic, or enter PMIDs with existing results."
                : "Select a subject topic, or enter any PMIDs for a separate analysis.")
            : stage === "network"
                ? "Select View network in Stage 3 to build the graph."
                : `${STAGE_LABELS[stage]} will start automatically.`;
        return;
    }
    if (["queued", "processing"].includes(status)) {
        title.textContent = `${STAGE_LABELS[stage]} is running`;
        text.textContent = message || "This stage is processing.";
        return;
    }
    if (status === "completed") {
        title.textContent = `${STAGE_LABELS[stage]} complete`;
        text.textContent = message || "This stage is complete.";
        return;
    }
    title.textContent = "This stage needs attention";
    text.textContent = error || message || "Review the error and start a new run.";
}

function updateProgress(stage, record) {
    const status = normalizeStatus(record?.status);
    const progress = status === "completed" ? 100 : Math.max(0, Math.min(100, Number(record?.progress) || 0));
    const card = $(`#${stage}ProgressCard`);
    const track = $(`#${stage}ProgressTrack`);
    const bar = $(`#${stage}ProgressBar`);
    const percent = $(`#${stage}ProgressPercent`);
    const label = $(`#${stage}ProgressStage`);
    const message = $(`#${stage}ProgressMessage`);
    if (card) card.className = `progress-card ${status}`;
    if (track) track.setAttribute("aria-valuenow", String(Math.round(progress)));
    if (bar) bar.style.width = `${progress}%`;
    if (percent) percent.textContent = `${Math.round(progress)}%`;
    if (label) label.textContent = humanizeStage(record?.stage, status);
    if (message) message.textContent = record?.message || "Waiting for this stage.";

    const tab = $(`[data-stage-target="${stage}"]`);
    if (tab) {
        tab.dataset.status = status;
        tab.disabled = status === "locked";
        tab.setAttribute("aria-disabled", String(status === "locked"));
    }
    const stepState = $(`#${stage}StepState`);
    if (stepState) stepState.textContent = stageStateText(status, progress);
    const connector = $(`[data-connector-after="${stage}"]`);
    if (connector) {
        connector.dataset.status = status;
        const connectorProgress = status === "completed" ? 100 : ["queued", "processing"].includes(status) ? progress : 0;
        connector.style.setProperty("--connector-progress", `${connectorProgress}%`);
    }
    updateReadiness(stage, status, record?.message, record?.error);
    setError(stage, status === "failed" ? (record?.error || record?.message || "This stage failed.") : "");
}

function renderRetrievalSummary(record) {
    const stats = record.stats || {};
    $("#retrievalSummaryStatus").textContent = stats.precomputed ? "Saved corpus loaded" : "Retrieval complete";
    $("#retrievalSummaryMessage").textContent = record.message || "The paper data is ready.";
    $("#summaryPaperCount").textContent = formatCount(stats.paper_count);
    $("#summaryAbstractCount").textContent = formatCount(stats.abstract_count);
    $("#summaryFulltextCount").textContent = formatCount(stats.fulltext_count ?? stats.fulltexts_downloaded);
    const fulltextLabel = $("#summaryFulltextLabel");
    if (fulltextLabel) fulltextLabel.textContent = stats.text_mode === "abstract" ? "Full texts requested" : "Full texts used";
    $("#summaryElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    $("#retrievalSummary").hidden = false;
}

function renderAnnotationSummary(record) {
    const stats = record.stats || {};
    $("#annotationSummaryStatus").textContent = "Entity extraction complete";
    $("#annotationSummaryMessage").textContent = record.message || "Normalized entities are ready.";
    $("#annotationMentionCount").textContent = formatCount(stats.unique_cell_count ?? stats.cell_count);
    $("#annotationNormalizedCount").textContent = formatCount(stats.unique_gene_count ?? stats.gene_count);
    $("#annotationHormoneCount").textContent = formatCount(stats.unique_hormone_count ?? stats.hormone_count);
    $("#annotationElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    $("#annotationSummary").hidden = false;
}

function renderRelationSummary(record) {
    const stats = record.stats || {};
    $("#relationSummaryStatus").textContent = "Relation extraction complete";
    $("#relationSummaryMessage").textContent = record.message || "The final annotated paper file is ready.";
    $("#relationCount").textContent = formatCount(stats.relation_count);
    $("#relationEdgeCount").textContent = formatCount(stats.global_relation_count);
    $("#relationChunkCount").textContent = formatCount(stats.chunk_count);
    $("#relationElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    $("#relationSummary").hidden = false;
}

function renderNetworkSummary(record) {
    const stats = record.stats || {};
    $("#networkSummaryStatus").textContent = "Network complete";
    $("#networkSummaryMessage").textContent = record.message || "Your interaction network is ready.";
    $("#networkNodeCount").textContent = formatCount(stats.node_count);
    $("#networkEdgeCount").textContent = formatCount(stats.edge_count);
    $("#networkElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    const link = $("#openNetworkLink");
    if (link) link.href = record.open_url || (state.runId ? `/network/${encodeURIComponent(state.runId)}` : "#");
    $("#networkSummary").hidden = false;
}

function renderSummary(stage, record) {
    const summary = $(`#${stage}Summary`);
    if (!summary) return;
    if (normalizeStatus(record?.status) !== "completed") {
        summary.hidden = true;
        return;
    }
    if (stage === "retrieval") renderRetrievalSummary(record);
    else if (stage === "annotation") renderAnnotationSummary(record);
    else if (stage === "relation") renderRelationSummary(record);
    else renderNetworkSummary(record);
}

function workflowRank(run) {
    const stages = run?.stages || {};
    const networkStatus = normalizeStatus(stages.network?.status);
    const relationStatus = normalizeStatus(stages.relation?.status);
    const annotationStatus = normalizeStatus(stages.annotation?.status);
    const retrievalStatus = normalizeStatus(stages.retrieval?.status);
    if (["queued", "processing", "completed", "failed"].includes(networkStatus)) return 3;
    if (retrievalStatus === "completed" && ["ready", "queued", "processing", "completed", "failed"].includes(relationStatus)) return 2;
    if (retrievalStatus === "completed" && ["ready", "queued", "processing", "completed", "failed"].includes(annotationStatus)) return 1;
    return 0;
}

function stageForRank(rank) {
    return ["retrieval", "annotation", "relation", "network"][Math.max(0, Math.min(3, rank))];
}

function runIsBusy(run) {
    return STAGES.some((stage) => ["queued", "processing"].includes(normalizeStatus(run?.stages?.[stage]?.status)));
}

function renderButtons(run) {
    const busy = runIsBusy(run);
    const retrievalStatus = normalizeStatus(run?.stages?.retrieval?.status || "ready");
    const start = $("#startAnalysis");
    const query = $("#queryInput");
    const corpus = $("#corpusSelect");
    const textMode = $("#textModeSelect");
    const hasPmids = Boolean(query?.value.trim());

    if (query) query.disabled = busy;
    if (corpus) corpus.disabled = busy;
    if (textMode) textMode.disabled = busy || !hasPmids;
    if (start) {
        start.disabled = busy;
        start.classList.toggle("loading", busy && ["queued", "processing"].includes(retrievalStatus));
        const label = $("span", start);
        if (label) label.textContent = busy ? "Processing…" : state.runId ? "Start new retrieval" : "Start retrieval";
    }

    const relationStatus = normalizeStatus(run?.stages?.relation?.status);
    const relationComplete = relationStatus === "completed";
    const networkStatus = normalizeStatus(run?.stages?.network?.status);
    const saveButton = $("#saveMetadataAnnotationsButton");
    if (saveButton) {
        saveButton.disabled = !relationComplete;
        saveButton.setAttribute("aria-disabled", String(!relationComplete));
    }

    const viewButton = $("#viewNetworkButton");
    if (viewButton) {
        const building = ["queued", "processing"].includes(networkStatus);
        const disabled = !relationComplete || building;
        viewButton.disabled = disabled;
        viewButton.setAttribute("aria-disabled", String(disabled));
        viewButton.classList.toggle("loading", building);
        const label = $("span", viewButton);
        if (label) label.textContent = building ? "Building network…" : networkStatus === "completed" ? "Open network" : "View network";
    }
}

function renderRun(run, { preserveActiveStage = false } = {}) {
    if (!run?.stages) return;
    state.runId = run.id;
    state.run = run;

    const query = $("#queryInput");
    const corpus = $("#corpusSelect");
    const textMode = $("#textModeSelect");
    if (query && query.value !== String(run.query || "")) query.value = String(run.query || "");
    if (corpus && run.input_mode === "precomputed" && run.corpus_id) corpus.value = run.corpus_id;
    if (textMode && ["abstract", "fulltext"].includes(run.text_mode)) textMode.value = run.text_mode;

    for (const stage of STAGES) {
        updateProgress(stage, run.stages[stage]);
        renderSummary(stage, run.stages[stage]);
    }
    renderButtons(run);
    renderCorpusOverview(run);
    updateInputModeDisplay();

    const rank = workflowRank(run);
    if (!preserveActiveStage && rank > state.workflowRank) {
        state.workflowRank = rank;
        activateStage(stageForRank(rank), { force: true });
    } else {
        state.workflowRank = Math.max(state.workflowRank, rank);
        if (!stageIsUnlocked(state.activeStage)) activateStage(stageForRank(rank), { force: true });
    }
    saveSession();
}

function initialRunShape() {
    return {
        stages: {
            retrieval: { status: "ready", stage: "ready", progress: 0, message: "Select a subject topic or enter PMIDs, then start retrieval.", stats: {} },
            annotation: { status: "locked", stage: "locked", progress: 0, message: "Entity extraction starts automatically after Stage 1.", stats: {} },
            relation: { status: "locked", stage: "locked", progress: 0, message: "Relation extraction starts automatically after Stage 2.", stats: {} },
            network: { status: "locked", stage: "locked", progress: 0, message: "The graph is built after you select View network.", stats: {} },
        },
    };
}

function resetInterface({ clearInputs = false } = {}) {
    state.runId = null;
    state.run = null;
    state.pollGeneration += 1;
    state.activeStage = "retrieval";
    state.workflowRank = 0;
    state.navigateWhenNetworkReady = false;
    if (clearInputs) {
        if ($("#queryInput")) $("#queryInput").value = "";
        if ($("#corpusSelect")) $("#corpusSelect").value = "";
        if ($("#textModeSelect")) $("#textModeSelect").value = "fulltext";
    }
    const initial = initialRunShape();
    for (const stage of STAGES) {
        updateProgress(stage, initial.stages[stage]);
        const summary = $(`#${stage}Summary`);
        if (summary) summary.hidden = true;
        setError(stage, "");
    }
    const corpusPreview = $("#corpusPreview");
    if (corpusPreview) corpusPreview.hidden = true;
    renderButtons(initial);
    activateStage("retrieval", { force: true });
    updateInputModeDisplay();
}

function setCorpusPreviewValues({ label, papers, abstracts, fulltexts, relations, message }) {
    if ($("#corpusPreviewLabel")) $("#corpusPreviewLabel").textContent = label;
    if ($("#corpusPreviewPaperCount")) $("#corpusPreviewPaperCount").textContent = papers;
    if ($("#corpusPreviewAbstractCount")) $("#corpusPreviewAbstractCount").textContent = abstracts;
    if ($("#corpusPreviewFulltextCount")) $("#corpusPreviewFulltextCount").textContent = fulltexts;
    if ($("#corpusPreviewRelationCount")) $("#corpusPreviewRelationCount").textContent = relations;
    if ($("#corpusPreviewMessage")) $("#corpusPreviewMessage").textContent = message;
}

function renderCorpusOverview(run) {
    const preview = $("#corpusPreview");
    if (!preview) return;
    const retrieval = run?.stages?.retrieval || {};
    const status = normalizeStatus(retrieval.status);
    const show = run?.input_mode === "precomputed" && ["queued", "processing", "completed"].includes(status);
    preview.hidden = !show;
    if (!show) return;

    const selectedLabel = $("#corpusSelect")?.selectedOptions?.[0]?.textContent || "Selected subject topic";
    const stats = retrieval.stats || {};
    const label = stats.corpus_label || run?.corpus_label || selectedLabel;
    if (status !== "completed") {
        setCorpusPreviewValues({
            label,
            papers: "—",
            abstracts: "—",
            fulltexts: "—",
            relations: "—",
            message: retrieval.message || "The selected saved corpus is being loaded.",
        });
        return;
    }

    setCorpusPreviewValues({
        label,
        papers: formatCount(stats.paper_count),
        abstracts: formatCount(stats.abstract_count),
        fulltexts: formatCount(stats.fulltext_count ?? stats.fulltexts_downloaded),
        relations: formatCount(stats.relation_count ?? stats.unique_paper_relation_count),
        message: stats.saved_pmid_selection
            ? `Loaded existing results for ${formatCount(stats.found_pmids?.length)} requested papers.${stats.missing_pmids?.length ? ` Unavailable PMIDs skipped: ${stats.missing_pmids.join(", ")}.` : ""}`
            : "The saved corpus is being used without new paper analysis.",
    });
}

function updateInputModeDisplay() {
    const hasPmids = Boolean($("#queryInput")?.value.trim());
    const hasCorpus = Boolean($("#corpusSelect")?.value);
    const busy = runIsBusy(state.run);
    const note = $("#pmidIsolationNote");
    const textMode = $("#textModeSelect");
    if (note) {
        note.textContent = hasPmids
            ? (PUBLIC_PRECOMPUTED_ONLY
                ? "Entered PMIDs are matched against both saved corpora. Available results are loaded; unavailable papers are skipped. Analyze other papers locally."
                : "Only the entered PMIDs will be downloaded and computed using your configured resources. Their results remain separate from the saved corpus files.")
            : hasCorpus
                ? "The selected saved corpus will be loaded without new paper analysis."
                : (PUBLIC_PRECOMPUTED_ONLY
                    ? "Select a subject topic, or enter PMIDs to load existing paper results."
                    : "Select a subject topic, or enter any PMIDs for a separate analysis.");
    }
    if (textMode) textMode.disabled = PUBLIC_PRECOMPUTED_ONLY || busy || !hasPmids;
}

async function pollRun(generation) {
    while (state.runId && generation === state.pollGeneration) {
        try {
            const run = await requestJson(`/api/runs/${encodeURIComponent(state.runId)}`);
            const previous = state.run;
            renderRun(run);

            for (const stage of STAGES) {
                if (normalizeStatus(previous?.stages?.[stage]?.status) !== "completed" && normalizeStatus(run.stages[stage]?.status) === "completed") {
                    showToast(`${STAGE_LABELS[stage]} completed.`);
                }
            }

            const networkStatus = normalizeStatus(run.stages.network?.status);
            if (state.navigateWhenNetworkReady && networkStatus === "completed") {
                state.navigateWhenNetworkReady = false;
                saveSession();
                window.location.assign(run.stages.network.open_url || `/network/${encodeURIComponent(state.runId)}`);
                return;
            }
            if (STAGES.some((stage) => normalizeStatus(run.stages[stage]?.status) === "failed")) return;
            if (normalizeStatus(run.stages.relation?.status) === "completed" && !["queued", "processing"].includes(networkStatus)) return;
            if (networkStatus === "completed") return;
        } catch (error) {
            setError(state.activeStage, error.message);
            return;
        }
        await sleep(POLL_MS);
    }
}

async function submitRetrieval(event) {
    event.preventDefault();
    const queryValue = $("#queryInput")?.value.trim() || "";
    const corpusValue = $("#corpusSelect")?.value || "";
    const textModeValue = $("#textModeSelect")?.value || "fulltext";
    if (!queryValue && !corpusValue) {
        const error = $("#formError");
        if (error) {
            error.textContent = "Select a subject topic or enter at least one PMID.";
            error.hidden = false;
        }
        $("#corpusSelect")?.focus();
        return;
    }

    const requestCorpusValue = corpusValue || "non_neoplastic_inflammatory";
    const selectedLabel = $("#corpusSelect")?.selectedOptions?.[0]?.textContent || "Selected subject topic";
    clearSession();
    resetInterface();
    if ($("#queryInput")) $("#queryInput").value = queryValue;
    if ($("#corpusSelect")) $("#corpusSelect").value = corpusValue;
    if ($("#textModeSelect")) $("#textModeSelect").value = textModeValue;
    updateInputModeDisplay();

    const optimistic = initialRunShape();
    optimistic.input_mode = queryValue && !PUBLIC_PRECOMPUTED_ONLY ? "pmid_only" : "precomputed";
    optimistic.corpus_id = corpusValue;
    optimistic.corpus_label = queryValue && PUBLIC_PRECOMPUTED_ONLY ? "Saved papers selected by PMID" : selectedLabel;
    optimistic.query = queryValue;
    optimistic.text_mode = textModeValue;
    optimistic.stages.retrieval = {
        status: "queued",
        stage: "queued",
        progress: 0,
        message: queryValue
            ? (PUBLIC_PRECOMPUTED_ONLY ? "Loading existing results for the entered PMIDs…" : "Creating a separate PMID run…")
            : "Loading the selected saved corpus…",
        stats: {},
    };
    state.run = optimistic;
    updateProgress("retrieval", optimistic.stages.retrieval);
    renderButtons(optimistic);
    renderCorpusOverview(optimistic);
    try {
        const run = await requestJson("/api/runs", {
            method: "POST",
            body: JSON.stringify({
                query: queryValue,
                corpus_id: requestCorpusValue,
                text_mode: textModeValue,
            }),
        });
        state.workflowRank = 0;
        renderRun(run);
        state.pollGeneration += 1;
        void pollRun(state.pollGeneration);
    } catch (error) {
        const failed = {
            status: "failed",
            stage: "failed",
            progress: 100,
            message: error.message,
            error: error.message,
            stats: {},
        };
        const shape = initialRunShape();
        shape.input_mode = optimistic.input_mode;
        shape.corpus_id = corpusValue;
        shape.corpus_label = selectedLabel;
        shape.stages.retrieval = failed;
        state.run = shape;
        updateProgress("retrieval", failed);
        renderButtons(shape);
        renderCorpusOverview(shape);
        setError("retrieval", error.message);
    }
}

function saveMetadataAndAnnotations() {
    if (!state.runId || normalizeStatus(state.run?.stages?.relation?.status) !== "completed") return;
    const downloadUrl = state.run?.stages?.relation?.download_url
        || `/api/runs/${encodeURIComponent(state.runId)}/download/stage3`;
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = "";
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    link.remove();
}

async function viewNetwork() {
    if (!state.runId || normalizeStatus(state.run?.stages?.relation?.status) !== "completed") return;
    setError("network", "");
    const networkStatus = normalizeStatus(state.run?.stages?.network?.status);
    if (networkStatus === "completed") {
        saveSession();
        window.location.assign(state.run.stages.network.open_url || `/network/${encodeURIComponent(state.runId)}`);
        return;
    }
    try {
        state.navigateWhenNetworkReady = true;
        activateStage("network", { force: true, scroll: true });
        const run = await requestJson(`/api/runs/${encodeURIComponent(state.runId)}/stages/4`, { method: "POST" });
        state.workflowRank = 3;
        renderRun(run);
        state.pollGeneration += 1;
        void pollRun(state.pollGeneration);
    } catch (error) {
        state.navigateWhenNetworkReady = false;
        setError("network", error.message);
    }
}

function initializeForms() {
    $("#analysisForm")?.addEventListener("submit", submitRetrieval);
    $("#saveMetadataAnnotationsButton")?.addEventListener("click", saveMetadataAndAnnotations);
    $("#viewNetworkButton")?.addEventListener("click", viewNetwork);
    $("#queryInput")?.addEventListener("input", updateInputModeDisplay);
    $("#corpusSelect")?.addEventListener("change", () => {
        updateInputModeDisplay();
        if (!state.runId) saveSession();
    });
}

async function restoreRun(saved) {
    if (!saved?.runId) return false;
    if ($("#queryInput")) $("#queryInput").value = saved.query || "";
    if ($("#corpusSelect") && saved.corpusId) $("#corpusSelect").value = saved.corpusId;
    if ($("#textModeSelect") && saved.textMode) $("#textModeSelect").value = saved.textMode;
    updateInputModeDisplay();
    try {
        const run = await requestJson(`/api/runs/${encodeURIComponent(saved.runId)}`);
        state.activeStage = STAGES.includes(saved.activeStage) ? saved.activeStage : stageForRank(workflowRank(run));
        state.workflowRank = workflowRank(run);
        renderRun(run, { preserveActiveStage: true });
        activateStage(stageIsUnlocked(state.activeStage) ? state.activeStage : stageForRank(state.workflowRank), { force: true });
        state.pollGeneration += 1;
        const relationDone = normalizeStatus(run.stages.relation?.status) === "completed";
        const networkDone = normalizeStatus(run.stages.network?.status) === "completed";
        if (!relationDone || ["queued", "processing"].includes(normalizeStatus(run.stages.network?.status))) {
            void pollRun(state.pollGeneration);
        } else if (networkDone) {
            renderRun(run, { preserveActiveStage: true });
        }
        return true;
    } catch {
        clearSession();
        return false;
    }
}

async function initializePage() {
    initializeNavigation();
    initializeForms();

    const navigation = performance.getEntriesByType("navigation")[0];
    const navigationType = navigation?.type || "navigate";
    const params = new URLSearchParams(window.location.search);
    const resumeRequested = params.get("resume") === "1";

    if (navigationType === "reload") {
        clearSession();
        stripResumeParameter();
        resetInterface({ clearInputs: true });
        return;
    }

    if (resumeRequested || navigationType === "back_forward") {
        const restored = await restoreRun(readSession());
        stripResumeParameter();
        if (restored) {
            $("#pipeline")?.scrollIntoView({ block: "start" });
            return;
        }
    }

    clearSession();
    stripResumeParameter();
    resetInterface({ clearInputs: true });
}

document.addEventListener("DOMContentLoaded", () => {
    void initializePage();
});

window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    const saved = readSession();
    if (!saved?.runId) return;
    void restoreRun(saved);
});
