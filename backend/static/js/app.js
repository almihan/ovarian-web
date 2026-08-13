"use strict";

// The current run ID lives only in this JavaScript object. It is never written
// to cookies, localStorage, sessionStorage, the URL, or a backend history list.
const state = {
    runId: null,
    pollGeneration: 0,
    activeStage: "retrieval",
    toastTimer: null,
};

const POLL_MS = 1500;
const REQUEST_TIMEOUT_MS = 30000;
const STAGES = ["retrieval", "annotation", "relation", "network"];
const STAGE_NUMBERS = { annotation: 2, relation: 3, network: 4 };
const STAGE_DEFAULTS = {
    retrieval: "Start Stage 1 to load the shared default corpus and retrieve any papers added by your search.",
    annotation: "Complete Stage 1 to begin entity extraction.",
    relation: "Complete Stage 2 to begin relation extraction.",
    network: "Complete Stage 3 to build the temporary interaction network.",
};
const READY_COPY = {
    retrieval: ["Ready to retrieve", "The built-in default query is always included. Optional terms add papers only to this page run."],
    annotation: ["Stage 1 is ready", "Extract cells with CellExLink and genes/hormones with PubTator3."],
    relation: ["Stage 2 is ready", "Extract evidence-supported relations from the aligned entity artifact."],
    network: ["Stage 3 is ready", "Build a temporary searchable graph for this page run."],
};
const LOCKED_COPY = {
    retrieval: READY_COPY.retrieval,
    annotation: ["Waiting for Stage 1", "Complete retrieval to unlock entity extraction."],
    relation: ["Waiting for Stage 2", "Complete entity extraction to unlock relation extraction."],
    network: ["Waiting for Stage 3", "Complete relation extraction to unlock network generation."],
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const sleep = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

async function requestJson(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs || REQUEST_TIMEOUT_MS);
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
        try { payload = await response.json(); } catch { payload = null; }
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

function formatPercent(value) {
    const numeric = Math.max(0, Math.min(1, Number(value) || 0));
    return `${(numeric * 100).toFixed(1)}%`;
}

function showToast(message) {
    const toast = $("#toast");
    const label = $("#toastMessage");
    if (!toast || !label) return;
    label.textContent = message;
    toast.hidden = false;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => { toast.hidden = true; }, 3800);
}

function normalizeStatus(value) {
    const status = String(value || "ready").toLowerCase();
    if (["ready", "locked", "queued", "processing", "completed", "failed"].includes(status)) {
        return status;
    }
    return status === "running" ? "processing" : "ready";
}

function humanizeStage(value, status) {
    if (!value) return status === "processing" ? "Working" : status === "locked" ? "Waiting" : "Ready";
    return String(value)
        .replace(/^(default|custom)_/, "")
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

function activateStage(stage, { scroll = false, focus = false } = {}) {
    if (!STAGES.includes(stage)) return;
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
}

function initializeNavigation() {
    $("#pipelineStepper")?.addEventListener("click", (event) => {
        const tab = event.target.closest("[data-stage-target]");
        if (tab) activateStage(tab.dataset.stageTarget, { scroll: window.innerWidth <= 820 });
    });
    $("#pipelineStepper")?.addEventListener("keydown", (event) => {
        const tab = event.target.closest("[data-stage-target]");
        if (!tab) return;
        const current = STAGES.indexOf(tab.dataset.stageTarget);
        let next = current;
        if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (current + 1) % STAGES.length;
        else if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (current - 1 + STAGES.length) % STAGES.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = STAGES.length - 1;
        else return;
        event.preventDefault();
        activateStage(STAGES[next], { focus: true });
    });
    $("#stageWorkspace")?.addEventListener("click", (event) => {
        const button = event.target.closest("[data-stage-next]");
        if (button) activateStage(button.dataset.stageNext, { scroll: true, focus: true });
    });
}

function setError(stage, message = "") {
    const id = stage === "retrieval" ? "formError" : `${stage}Error`;
    const element = $(`#${id}`);
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
        [title.textContent, text.textContent] = LOCKED_COPY[stage];
    } else if (status === "ready") {
        [title.textContent, text.textContent] = READY_COPY[stage];
    } else if (["queued", "processing"].includes(status)) {
        title.textContent = `${stage === "retrieval" ? "Retrieval" : stage === "annotation" ? "Entity extraction" : stage === "relation" ? "Relation extraction" : "Network generation"} is running`;
        text.textContent = message || "This stage is processing.";
    } else if (status === "completed") {
        title.textContent = `${stage === "retrieval" ? "Retrieval" : stage === "annotation" ? "Entity extraction" : stage === "relation" ? "Relation extraction" : "Network generation"} complete`;
        text.textContent = message || "The next stage is ready.";
    } else {
        title.textContent = "This stage needs attention";
        text.textContent = error || message || "Review the error and try again.";
    }
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
    if (message) message.textContent = record?.message || STAGE_DEFAULTS[stage];

    const tab = $(`[data-stage-target="${stage}"]`);
    if (tab) tab.dataset.status = status;
    const stepState = $(`#${stage}StepState`);
    if (stepState) stepState.textContent = stageStateText(status, progress);
    const connector = $(`[data-connector-after="${stage}"]`);
    if (connector) {
        connector.dataset.status = status;
        const connectorProgress = status === "completed" ? 100 : ["queued", "processing", "failed"].includes(status) ? progress : 0;
        connector.style.setProperty("--connector-progress", `${connectorProgress}%`);
    }
    updateReadiness(stage, status, record?.message, record?.error);
    setError(stage, status === "failed" ? (record?.error || record?.message || "This stage failed.") : "");
}

function renderRetrievalSummary(record) {
    const stats = record.stats || {};
    $("#retrievalSummaryStatus").textContent = "Retrieval complete";
    $("#retrievalSummaryMessage").textContent = record.message || "The aligned chunk artifact is ready.";
    $("#summaryPaperCount").textContent = Number(stats.paper_count || 0).toLocaleString();
    $("#summaryAbstractCount").textContent = Number(stats.abstract_count || 0).toLocaleString();
    $("#summaryFulltextCount").textContent = Number(stats.fulltext_count ?? stats.fulltexts_downloaded ?? 0).toLocaleString();
    $("#summaryElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    $("#downloadChunksLink").href = record.download_url || "#";
    $("#retrievalSummary").hidden = false;
}

function renderAnnotationSummary(record) {
    const stats = record.stats || {};
    $("#annotationSummaryStatus").textContent = "Entity extraction complete";
    $("#annotationSummaryMessage").textContent = record.message || "Normalized cells, genes, and hormones are ready.";
    $("#annotationMentionCount").textContent = Number(stats.cell_count ?? stats.mention_count ?? 0).toLocaleString();
    $("#annotationNormalizedCount").textContent = Number(stats.gene_count || 0).toLocaleString();
    $("#annotationHormoneCount").textContent = Number(stats.hormone_count || 0).toLocaleString();
    $("#annotationElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    $("#downloadAnnotationsLink").href = record.download_url || "#";
    $("#annotationSummary").hidden = false;
}

function renderRelationSummary(record) {
    const stats = record.stats || {};
    $("#relationSummaryStatus").textContent = "Relation extraction complete";
    $("#relationSummaryMessage").textContent = record.message || "Validated relation rows are ready.";
    $("#relationCount").textContent = Number(stats.relation_count || 0).toLocaleString();
    $("#relationChunkCount").textContent = Number(stats.chunk_count || 0).toLocaleString();
    $("#relationCacheRate").textContent = formatPercent(stats.prompt_cache_rate || 0);
    $("#relationElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    $("#downloadRelationsLink").href = record.download_url || "#";
    $("#relationSummary").hidden = false;
}

function renderNetworkSummary(record) {
    const stats = record.stats || {};
    $("#networkSummaryStatus").textContent = "Network complete";
    $("#networkSummaryMessage").textContent = record.message || "Your temporary interaction network is ready.";
    $("#networkNodeCount").textContent = Number(stats.node_count || 0).toLocaleString();
    $("#networkEdgeCount").textContent = Number(stats.edge_count || 0).toLocaleString();
    $("#networkPaperCount").textContent = Number(stats.paper_count || 0).toLocaleString();
    $("#networkElapsed").textContent = formatDuration(record.elapsed_seconds ?? stats.elapsed_seconds);
    $("#openNetworkLink").href = record.open_url || (state.runId ? `/network/${encodeURIComponent(state.runId)}` : "#");
    $("#networkSummary").hidden = false;
}

function renderSummary(stage, record) {
    const summary = $(`#${stage}Summary`);
    if (!summary) return;
    if (record.status !== "completed") {
        summary.hidden = true;
        return;
    }
    if (stage === "retrieval") renderRetrievalSummary(record);
    else if (stage === "annotation") renderAnnotationSummary(record);
    else if (stage === "relation") renderRelationSummary(record);
    else renderNetworkSummary(record);
}

function renderButtons(run) {
    const retrieval = run?.stages?.retrieval;
    const retrievalBusy = ["queued", "processing"].includes(retrieval?.status);
    const query = $("#queryInput");
    const startRetrieval = $("#startAnalysis");
    if (query) query.disabled = retrievalBusy;
    if (startRetrieval) {
        startRetrieval.disabled = retrievalBusy;
        startRetrieval.classList.toggle("loading", retrievalBusy);
        const label = $("span", startRetrieval);
        if (label) label.textContent = retrievalBusy
            ? "Retrieving…"
            : retrieval?.status === "completed" ? "Start a new retrieval" : "Start retrieval";
    }

    for (const stage of ["annotation", "relation", "network"]) {
        const record = run?.stages?.[stage] || { status: "locked" };
        const button = $(`#start${stage[0].toUpperCase()}${stage.slice(1)}`);
        if (!button) continue;
        const busy = ["queued", "processing"].includes(record.status);
        button.disabled = !["ready", "failed"].includes(record.status);
        button.classList.toggle("loading", busy);
        const label = $("span", button);
        if (!label) continue;
        const idle = stage === "annotation" ? "Start entity extraction" : stage === "relation" ? "Start relation extraction" : "Build and explore network";
        const running = stage === "annotation" ? "Extracting entities…" : stage === "relation" ? "Extracting relations…" : "Building network…";
        label.textContent = busy ? running : record.status === "failed" ? `Retry ${stage === "annotation" ? "entity extraction" : stage === "relation" ? "relation extraction" : "network generation"}` : idle;
    }
}

function renderRun(run) {
    if (!run?.stages) return;
    state.runId = run.id;
    for (const stage of STAGES) {
        const record = run.stages[stage];
        updateProgress(stage, record);
        renderSummary(stage, record);
    }
    renderButtons(run);
}

function resetInterface({ clearQuery = false } = {}) {
    state.runId = null;
    state.pollGeneration += 1;
    if (clearQuery) {
        const query = $("#queryInput");
        if (query) query.value = "";
    }
    const initial = {
        retrieval: { status: "ready", stage: "ready", progress: 0, message: STAGE_DEFAULTS.retrieval, stats: {} },
        annotation: { status: "locked", stage: "locked", progress: 0, message: STAGE_DEFAULTS.annotation, stats: {} },
        relation: { status: "locked", stage: "locked", progress: 0, message: STAGE_DEFAULTS.relation, stats: {} },
        network: { status: "locked", stage: "locked", progress: 0, message: STAGE_DEFAULTS.network, stats: {} },
    };
    for (const stage of STAGES) {
        updateProgress(stage, initial[stage]);
        $(`#${stage}Summary`)?.setAttribute("hidden", "");
        setError(stage, "");
    }
    renderButtons({ stages: initial });
    activateStage("retrieval");
}

async function pollStage(stage, generation) {
    while (state.runId && generation === state.pollGeneration) {
        try {
            const run = await requestJson(`/api/runs/${encodeURIComponent(state.runId)}`);
            renderRun(run);
            const status = run.stages[stage].status;
            if (status === "completed") {
                showToast(`${stage === "retrieval" ? "Retrieval" : stage === "annotation" ? "Entity extraction" : stage === "relation" ? "Relation extraction" : "Network generation"} completed.`);
                return;
            }
            if (status === "failed") return;
        } catch (error) {
            setError(stage, error.message);
            return;
        }
        await sleep(POLL_MS);
    }
}

async function submitRetrieval(event) {
    event.preventDefault();
    resetInterface();
    const optimistic = {
        status: "queued",
        stage: "queued",
        progress: 0,
        message: "Creating a new temporary page run…",
        stats: {},
    };
    updateProgress("retrieval", optimistic);
    renderButtons({ stages: { retrieval: optimistic, annotation: { status: "locked" }, relation: { status: "locked" }, network: { status: "locked" } } });
    try {
        const run = await requestJson("/api/runs", {
            method: "POST",
            body: JSON.stringify({ query: $("#queryInput")?.value?.trim() || "" }),
        });
        renderRun(run);
        state.pollGeneration += 1;
        void pollStage("retrieval", state.pollGeneration);
    } catch (error) {
        const failed = { status: "failed", stage: "failed", progress: 100, message: error.message, error: error.message, stats: {} };
        updateProgress("retrieval", failed);
        renderButtons({ stages: { retrieval: failed, annotation: { status: "locked" }, relation: { status: "locked" }, network: { status: "locked" } } });
    }
}

async function submitStage(stage, event) {
    event.preventDefault();
    if (!state.runId) {
        setError(stage, "Start and complete Stage 1 first.");
        return;
    }
    setError(stage, "");
    const number = STAGE_NUMBERS[stage];
    try {
        const run = await requestJson(`/api/runs/${encodeURIComponent(state.runId)}/stages/${number}`, { method: "POST" });
        renderRun(run);
        state.pollGeneration += 1;
        void pollStage(stage, state.pollGeneration);
    } catch (error) {
        setError(stage, error.message);
    }
}

function initializeForms() {
    $("#analysisForm")?.addEventListener("submit", submitRetrieval);
    $("#annotationForm")?.addEventListener("submit", (event) => submitStage("annotation", event));
    $("#relationForm")?.addEventListener("submit", (event) => submitStage("relation", event));
    $("#networkForm")?.addEventListener("submit", (event) => submitStage("network", event));
}

document.addEventListener("DOMContentLoaded", () => {
    initializeNavigation();
    initializeForms();
    resetInterface({ clearQuery: true });
});

// Browsers can restore a page from their in-memory back/forward cache. Drop
// the previous run before that snapshot can be shown again.
window.addEventListener("pagehide", () => {
    state.runId = null;
    state.pollGeneration += 1;
});

window.addEventListener("pageshow", (event) => {
    if (event.persisted) resetInterface({ clearQuery: true });
});
