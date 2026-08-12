"use strict";

const state = {
    retrieval: { currentJobId: null, generation: 0, busy: false },
    annotation: {
        currentJobId: null,
        generation: 0,
        busy: false,
        executor: "disabled",
        compute: "entity worker",
        configured: false,
        sourceRetrievalId: null,
    },
    relation: {
        currentJobId: null,
        generation: 0,
        busy: false,
        model: "gpt-5.4-nano",
        compute: "OpenAI Responses API",
        configured: false,
        windowSize: 500,
        concurrency: 8,
        sourceAnnotationId: null,
    },
    network: {
        currentJobId: null,
        generation: 0,
        busy: false,
        configured: false,
        compute: "Local/Railway CPU · SQLite · PyVis",
        sourceRelationId: null,
    },
    completedRetrievals: [],
    completedAnnotations: [],
    completedRelations: [],
    completedNetworks: [],
    ui: {
        activeStage: "retrieval",
        stageStatus: {
            retrieval: "ready",
            annotation: "locked",
            relation: "locked",
            network: "locked",
        },
        stageProgress: {
            retrieval: 0,
            annotation: 0,
            relation: 0,
            network: 0,
        },
    },
    toastTimer: null,
};

const POLL_MS = 2000;
const RELATION_POLL_MS = 2000;
const NETWORK_POLL_MS = 2000;
const REQUEST_TIMEOUT_MS = 25000;
const STAGE_ORDER = ["retrieval", "annotation", "relation", "network"];
const STAGE_COPY = {
    retrieval: {
        lockedTitle: "Ready to retrieve",
        lockedText: "Your optional search additions are taken from the bar above.",
        readyTitle: "Ready to retrieve",
        readyText: "Your optional search additions are taken from the bar above.",
        runningTitle: "Retrieval is running",
        runningText: "You can inspect another stage while this job continues.",
        completedTitle: "Retrieval complete",
        completedText: "A reusable chunk artifact is available for Stage 2.",
        failedTitle: "Retrieval needs attention",
        failedText: "Review the message and start the retrieval again.",
    },
    annotation: {
        lockedTitle: "Waiting for Stage 1",
        lockedText: "Complete a retrieval to unlock entity extraction.",
        readyTitle: "Stage 1 result selected",
        readyText: "Cell, gene/protein, and hormone extraction is ready to start.",
        runningTitle: "Entity extraction is running",
        runningText: "CellExLink and PubTator3 are processing the selected retrieval.",
        completedTitle: "Entity extraction complete",
        completedText: "The normalized entity artifact is available for Stage 3.",
        failedTitle: "Entity extraction needs attention",
        failedText: "Review the message and start Stage 2 again.",
    },
    relation: {
        lockedTitle: "Waiting for Stage 2",
        lockedText: "Complete entity extraction to unlock relation extraction.",
        readyTitle: "Stage 2 result selected",
        readyText: "The aligned entity artifact is ready for relation extraction.",
        runningTitle: "Relation extraction is running",
        runningText: "Evidence-supported directed relations are being generated.",
        completedTitle: "Relation extraction complete",
        completedText: "The validated relation artifact is available for Stage 4.",
        failedTitle: "Relation extraction needs attention",
        failedText: "Review the message and start Stage 3 again.",
    },
    network: {
        lockedTitle: "Waiting for Stage 3",
        lockedText: "Complete relation extraction to unlock network generation.",
        readyTitle: "Stage 3 result selected",
        readyText: "The interaction network is ready to build and explore.",
        runningTitle: "Network generation is running",
        runningText: "The entity index and evidence-linked graph are being built.",
        completedTitle: "Network complete",
        completedText: "Open the explorer to search nodes, relations, and evidence.",
        failedTitle: "Network generation needs attention",
        failedText: "Review the message and start Stage 4 again.",
    },
};
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function requestJson(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(
        () => controller.abort(),
        options.timeoutMs || REQUEST_TIMEOUT_MS,
    );
    const fetchOptions = { ...options };
    delete fetchOptions.timeoutMs;
    const headers = {
        Accept: "application/json",
        ...(fetchOptions.body ? { "Content-Type": "application/json" } : {}),
        ...(fetchOptions.headers || {}),
    };
    try {
        const response = await fetch(url, {
            ...fetchOptions,
            headers,
            signal: controller.signal,
            cache: "no-store",
        });
        let payload = null;
        try { payload = await response.json(); } catch { payload = null; }
        if (!response.ok) {
            const detail = payload?.detail;
            const message = Array.isArray(detail)
                ? detail.map((item) => item.msg || String(item)).join(" ")
                : detail || `Request failed with status ${response.status}.`;
            throw new Error(message);
        }
        return payload;
    } catch (error) {
        if (error?.name === "AbortError") {
            throw new Error(
                "The server did not respond in time. The job may still be running; refresh its status shortly.",
            );
        }
        throw error;
    } finally {
        window.clearTimeout(timeout);
    }
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function showToast(message) {
    const toast = $("#toast");
    const label = $("#toastMessage");
    if (!toast || !label) return;
    label.textContent = message;
    toast.hidden = false;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => { toast.hidden = true; }, 4200);
}

function formatDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";
    return new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
    }).format(date);
}

function formatDuration(value) {
    const seconds = Math.max(0, Math.round(Number(value) || 0));
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
    return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}

function formatBytes(value) {
    let amount = Number(value) || 0;
    const units = ["B", "KB", "MB", "GB"];
    let index = 0;
    while (amount >= 1024 && index < units.length - 1) {
        amount /= 1024;
        index += 1;
    }
    return `${amount.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatPercent(value) {
    const numeric = Math.max(0, Math.min(1, Number(value) || 0));
    return `${(numeric * 100).toFixed(1)}%`;
}

function retrievalLabel(job) {
    const query = String(job?.query || "").trim();
    return query || "Built-in ovarian corpus";
}

function annotationLabel(job) {
    return retrievalLabel(job?.source_job || {});
}

function relationLabel(job) {
    const retrieval = job?.source_annotation_job?.retrieval || {};
    return retrievalLabel(retrieval);
}

function networkLabel(job) {
    const sourceId = String(job?.source_relation_job_id || "").trim();
    return sourceId ? `Network from relation result ${sourceId}` : "Interaction network";
}

function humanizeStage(value) {
    return String(value || "")
        .replaceAll("_", " ")
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function stageLabel(stage, status) {
    const labels = {
        queued: "Queued",
        preparing: "Preparing retrieval",
        searching: "Searching PubMed",
        metadata: "Retrieving metadata",
        fulltext: "Downloading full text",
        chunking: "Preparing chunks",
        chunks: "Preparing chunks",
        publishing: "Publishing artifact",
        starting_gpu: "Starting Modal T4",
        preparing_gpu: "Preparing GPU input",
        starting_local: "Starting local worker",
        preparing_local: "Preparing local input",
        running_parallel_branches: "Running entity branches",
        recognition: "Identifying cell types",
        releasing_recognition: "Releasing recognition model",
        normalization: "Normalizing cells",
        merging_entities: "Merging cells, genes, and hormones",
        preparing_sources: "Aligning text and entities",
        preparing_online_requests: "Preparing relation window",
        openai_responses_in_progress: "Extracting relations online",
        checkpointed_relations: "Relation checkpoint saved",
        publishing_relations: "Publishing relation artifact",
        preparing_network_sources: "Preparing network sources",
        building_entity_index: "Building global entity index",
        building_interaction_network: "Building interaction network",
        writing_entity_index: "Writing entity index",
        publishing_network: "Publishing network artifacts",
        completed: "Completed",
        failed: "Failed",
    };
    const key = String(stage || "");
    return labels[key] || (key ? humanizeStage(key) : status === "processing" ? "Working" : "Ready");
}

function normalizeStageStatus(status) {
    const value = String(status || "ready").toLowerCase();
    if (["disabled", "unavailable"].includes(value)) return "locked";
    if (["running", "in_progress"].includes(value)) return "processing";
    if (["ready", "locked", "queued", "processing", "completed", "failed"].includes(value)) {
        return value;
    }
    return "ready";
}

function stageStatusText(status, progress) {
    if (status === "locked") return "Waiting";
    if (status === "queued") return "Queued";
    if (status === "processing") return `Running ${Math.round(progress)}%`;
    if (status === "completed") return "Complete";
    if (status === "failed") return "Attention";
    return "Ready";
}

function updateStageReadiness(kind, status) {
    const copy = STAGE_COPY[kind];
    const readiness = $(`#${kind}Readiness`);
    const title = $(`#${kind}ReadinessTitle`);
    const text = $(`#${kind}ReadinessText`);
    if (!copy || !readiness || !title || !text) return;

    readiness.dataset.state = status;
    if (status === "queued" || status === "processing") {
        title.textContent = copy.runningTitle;
        text.textContent = copy.runningText;
        return;
    }
    if (status === "completed") {
        title.textContent = copy.completedTitle;
        text.textContent = copy.completedText;
        return;
    }
    if (status === "failed") {
        title.textContent = copy.failedTitle;
        text.textContent = copy.failedText;
        return;
    }
    if (status === "locked") {
        title.textContent = copy.lockedTitle;
        text.textContent = copy.lockedText;
        return;
    }
    title.textContent = copy.readyTitle;
    text.textContent = copy.readyText;
}

function setStageStatus(kind, status, progress = 0) {
    if (!STAGE_ORDER.includes(kind)) return;
    const normalized = normalizeStageStatus(status);
    const numericProgress = Math.max(0, Math.min(100, Number(progress) || 0));
    state.ui.stageStatus[kind] = normalized;
    state.ui.stageProgress[kind] = normalized === "completed" ? 100 : numericProgress;

    const tab = $(`[data-stage-target="${kind}"]`);
    const stateLabel = $(`#${kind}StepState`);
    if (tab) {
        tab.dataset.status = normalized;
        tab.setAttribute(
            "aria-label",
            `Stage ${STAGE_ORDER.indexOf(kind) + 1}, ${tab.querySelector("strong")?.textContent || kind}: ${stageStatusText(normalized, numericProgress)}`,
        );
    }
    if (stateLabel) stateLabel.textContent = stageStatusText(normalized, numericProgress);

    const connector = $(`[data-connector-after="${kind}"]`);
    if (connector) {
        connector.dataset.status = normalized;
        const connectorProgress = normalized === "completed"
            ? 100
            : ["queued", "processing", "failed"].includes(normalized)
                ? numericProgress
                : 0;
        connector.style.setProperty("--connector-progress", `${connectorProgress}%`);
    }
    updateStageReadiness(kind, normalized);
}

function setStageAvailability(kind, available) {
    const current = state.ui.stageStatus[kind];
    if (["queued", "processing", "completed", "failed"].includes(current)) return;
    setStageStatus(kind, available ? "ready" : "locked", 0);
}

function activateStage(kind, { scroll = false, focusTab = false } = {}) {
    if (!STAGE_ORDER.includes(kind)) return;
    state.ui.activeStage = kind;

    $$('[data-stage-target]').forEach((tab) => {
        const active = tab.dataset.stageTarget === kind;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
    });
    $$('[data-stage-panel]').forEach((panel) => {
        const active = panel.dataset.stagePanel === kind;
        panel.hidden = !active;
        panel.classList.toggle("is-active", active);
    });

    if (focusTab) $(`[data-stage-target="${kind}"]`)?.focus({ preventScroll: true });
    if (scroll) {
        $("#stageWorkspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function handleStageTabKeydown(event) {
    const tab = event.target.closest("[data-stage-target]");
    if (!tab) return;
    const currentIndex = STAGE_ORDER.indexOf(tab.dataset.stageTarget);
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (currentIndex + 1) % STAGE_ORDER.length;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (currentIndex - 1 + STAGE_ORDER.length) % STAGE_ORDER.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = STAGE_ORDER.length - 1;
    else return;
    event.preventDefault();
    activateStage(STAGE_ORDER[nextIndex], { focusTab: true });
}

function initializeStageNavigation() {
    $("#pipelineStepper")?.addEventListener("click", (event) => {
        const tab = event.target.closest("[data-stage-target]");
        if (!tab) return;
        activateStage(tab.dataset.stageTarget, { scroll: window.innerWidth <= 820 });
    });
    $("#pipelineStepper")?.addEventListener("keydown", handleStageTabKeydown);
    $("#stageWorkspace")?.addEventListener("click", (event) => {
        const button = event.target.closest("[data-stage-next]");
        if (!button) return;
        activateStage(button.dataset.stageNext, { scroll: true, focusTab: true });
    });
    activateStage(state.ui.activeStage);
}

function defaultProgressMessage(kind) {
    if (kind === "annotation") return "Select a completed retrieval to begin.";
    if (kind === "relation") return "Select a completed entity artifact to begin.";
    if (kind === "network") return "Select a completed relation artifact to begin.";
    return "Start Stage 1 to create a reusable chunk artifact.";
}

function updateProgress(kind, job) {
    const status = normalizeStageStatus(job?.status || "ready");
    const reportedProgress = Math.max(0, Math.min(100, Number(job?.progress || 0)));
    const progress = status === "completed" ? 100 : reportedProgress;
    const card = $(`#${kind}ProgressCard`);
    const track = $(`#${kind}ProgressTrack`);
    const bar = $(`#${kind}ProgressBar`);
    const percent = $(`#${kind}ProgressPercent`);
    const stage = $(`#${kind}ProgressStage`);
    const message = $(`#${kind}ProgressMessage`);
    if (!card || !track || !bar || !percent || !stage || !message) return;
    card.className = `progress-card ${status}`;
    track.setAttribute("aria-valuenow", String(Math.round(progress)));
    bar.style.width = `${progress}%`;
    percent.textContent = `${Math.round(progress)}%`;
    stage.textContent = stageLabel(job?.stage, status);
    message.textContent = job?.message || defaultProgressMessage(kind);
    setStageStatus(kind, status, progress);
}

function setRetrievalBusy(busy) {
    state.retrieval.busy = busy;
    const button = $("#startAnalysis");
    const input = $("#queryInput");
    if (!button || !input) return;
    button.disabled = busy;
    input.disabled = busy;
    button.classList.toggle("loading", busy);
    const label = $("span", button);
    if (label) label.textContent = busy ? "Retrieving…" : "Start retrieval";
}

function setAnnotationBusy(busy) {
    state.annotation.busy = busy;
    const button = $("#startAnnotation");
    if (!button) return;
    button.disabled = busy || !state.annotation.sourceRetrievalId || !state.annotation.configured;
    button.classList.toggle("loading", busy);
    const label = $("span", button);
    if (label) {
        const running = state.annotation.executor === "local"
            ? "Running locally…"
            : "Running entity extraction…";
        label.textContent = busy ? running : "Start entity extraction";
    }
}

function setRelationBusy(busy) {
    state.relation.busy = busy;
    const button = $("#startRelation");
    if (!button) return;
    button.disabled = busy || !state.relation.sourceAnnotationId || !state.relation.configured;
    button.classList.toggle("loading", busy);
    const label = $("span", button);
    if (label) label.textContent = busy ? "Running online requests…" : "Start relation extraction";
}

function setNetworkBusy(busy) {
    state.network.busy = busy;
    const button = $("#startNetwork");
    if (!button) return;
    button.disabled = busy || !state.network.sourceRelationId || !state.network.configured;
    button.classList.toggle("loading", busy);
    const label = $("span", button);
    if (label) label.textContent = busy ? "Opening explorer…" : "Build and explore network";
}

function errorElement(kind) {
    if (kind === "annotation") return $("#annotationError");
    if (kind === "relation") return $("#relationError");
    if (kind === "network") return $("#networkError");
    return $("#formError");
}

function showError(kind, message) {
    const element = errorElement(kind);
    if (!element) return;
    element.textContent = message;
    element.hidden = false;
}

function clearError(kind) {
    const element = errorElement(kind);
    if (!element) return;
    element.textContent = "";
    element.hidden = true;
}

function renderRetrievalSummary(summary) {
    const stats = summary?.stats || {};
    const job = summary?.job || {};
    const downloads = summary?.downloads || {};
    $("#retrievalSummaryStatus").textContent = "Retrieval complete";
    $("#retrievalSummaryMessage").textContent = job.message || "The reusable chunk artifact is ready.";
    $("#summaryPaperCount").textContent = Number(stats.paper_count || job.paper_count || 0).toLocaleString();
    $("#summaryAbstractCount").textContent = Number(stats.abstract_count || 0).toLocaleString();
    $("#summaryFulltextCount").textContent = Number(stats.fulltext_available || stats.fulltexts_downloaded || 0).toLocaleString();
    $("#summaryElapsed").textContent = formatDuration(stats.elapsed_seconds ?? job.elapsed_seconds);
    $("#downloadChunksLink").href = downloads.chunks || "#";
    $("#retrievalSummary").hidden = false;
}

function renderAnnotationSummary(summary) {
    const stats = summary?.stats || {};
    const job = summary?.job || {};
    const downloads = summary?.downloads || {};
    const summaryStatus = $("#annotationSummaryStatus");
    if (summaryStatus) summaryStatus.textContent = job.reused
        ? "Existing entity result reused"
        : "Entity extraction complete";
    const summaryMessage = $("#annotationSummaryMessage");
    if (summaryMessage) summaryMessage.textContent = job.message || "Normalized cells, genes, and hormones are ready.";
    const cells = Number(stats.cell_count ?? stats.mention_count ?? job.mention_count ?? 0);
    const genes = Number(stats.gene_count ?? 0);
    const hormones = Number(stats.hormone_count ?? 0);
    $("#annotationMentionCount").textContent = cells.toLocaleString();
    $("#annotationNormalizedCount").textContent = genes.toLocaleString();
    $("#annotationHormoneCount").textContent = hormones.toLocaleString();
    $("#annotationElapsed").textContent = formatDuration(stats.elapsed_seconds ?? job.elapsed_seconds);
    $("#downloadAnnotationsLink").href = downloads.annotations || "#";
    $("#annotationSummary").hidden = false;
}

function renderRelationSummary(summary) {
    const stats = summary?.stats || {};
    const job = summary?.job || {};
    const downloads = summary?.downloads || {};
    const relationCount = Number(stats.relation_count ?? job.relation_count ?? 0);
    const chunks = Number(stats.processed_chunk_count ?? stats.chunk_count ?? job.processed_chunk_count ?? 0);
    const inputTokens = Number(stats.input_tokens || 0);
    const cachedTokens = Number(stats.cached_input_tokens || 0);
    const cacheRate = Number.isFinite(Number(stats.prompt_cache_rate))
        ? Number(stats.prompt_cache_rate)
        : inputTokens ? cachedTokens / inputTokens : 0;

    $("#relationSummaryStatus").textContent = job.reused
        ? "Existing relation result reused"
        : "Relation extraction complete";
    $("#relationSummaryMessage").textContent = job.message || "Validated relation rows are ready.";
    $("#relationCount").textContent = relationCount.toLocaleString();
    $("#relationChunkCount").textContent = chunks.toLocaleString();
    $("#relationCacheRate").textContent = formatPercent(cacheRate);
    $("#relationElapsed").textContent = formatDuration(stats.elapsed_seconds ?? job.elapsed_seconds);

    $("#downloadRelationsLink").href = downloads.relations || "#";
    $("#relationSummary").hidden = false;
}

function renderNetworkSummary(summary) {
    const stats = summary?.stats || {};
    const job = summary?.job || {};
    $("#networkSummaryStatus").textContent = job.reused ? "Existing network reused" : "Network complete";
    $("#networkSummaryMessage").textContent = summary?.message || job.message || "Your interaction network is ready.";
    $("#networkNodeCount").textContent = Number(stats.node_count ?? job.node_count ?? 0).toLocaleString();
    $("#networkEdgeCount").textContent = Number(stats.edge_count ?? job.edge_count ?? 0).toLocaleString();
    $("#networkPaperCount").textContent = Number(stats.paper_count ?? job.paper_count ?? 0).toLocaleString();
    $("#networkElapsed").textContent = formatDuration(stats.elapsed_seconds ?? job.elapsed_seconds);
    $("#openNetworkLink").href = job.explore_url || `/network/${encodeURIComponent(job.id || summary?.job_id || "")}`;
    $("#networkSummary").hidden = false;
}

async function openRetrievalSummary(jobId, scroll = true) {
    const summary = await requestJson(`/api/papers/jobs/${encodeURIComponent(jobId)}`);
    renderRetrievalSummary(summary);
    updateProgress("retrieval", summary.job || {});
    selectRetrieval(jobId, false);
    if (scroll) {
        activateStage("retrieval");
        $("#retrievalSummary")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    return summary;
}

async function openAnnotationSummary(jobId, scroll = true) {
    const summary = await requestJson(`/api/annotations/${encodeURIComponent(jobId)}/summary`);
    renderAnnotationSummary(summary);
    updateProgress("annotation", summary.job || {});
    selectAnnotationForRelations(jobId, false);
    if (scroll) {
        activateStage("annotation");
        $("#annotationSummary")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    return summary;
}

async function openRelationSummary(jobId, scroll = true) {
    const summary = await requestJson(`/api/relations/${encodeURIComponent(jobId)}/summary`);
    renderRelationSummary(summary);
    updateProgress("relation", summary.job || {});
    selectRelationForNetwork(jobId, false);
    if (scroll) {
        activateStage("relation");
        $("#relationSummary")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    return summary;
}

async function openNetworkSummary(jobId, scroll = true) {
    const summary = await requestJson(`/api/networks/${encodeURIComponent(jobId)}/summary`);
    renderNetworkSummary(summary);
    updateProgress("network", summary.job || {});
    if (scroll) {
        activateStage("network");
        $("#networkSummary")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    return summary;
}

async function pollRetrieval(jobId, generation) {
    while (generation === state.retrieval.generation) {
        try {
            const job = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}`);
            updateProgress("retrieval", job);
            if (job.status === "completed") {
                setRetrievalBusy(false);
                await openRetrievalSummary(jobId, false);
                await loadActivity(jobId, null, null);
                showToast("Retrieval finished. Stage 2 is ready when you choose to start it.");
                return;
            }
            if (job.status === "failed") {
                setRetrievalBusy(false);
                showError("retrieval", job.error || job.message || "Retrieval failed.");
                await loadActivity();
                return;
            }
        } catch (error) {
            setRetrievalBusy(false);
            showError("retrieval", error.message);
            return;
        }
        await sleep(POLL_MS);
    }
}

async function pollAnnotation(jobId, generation) {
    while (generation === state.annotation.generation) {
        try {
            const job = await requestJson(`/api/annotations/${encodeURIComponent(jobId)}`);
            updateProgress("annotation", job);
            if (job.status === "completed") {
                setAnnotationBusy(false);
                await openAnnotationSummary(jobId, false);
                await loadActivity(job.source_job_id, jobId, null);
                showToast(job.reused ? "A matching Stage 2 result was reused." : "Entity extraction finished.");
                return;
            }
            if (job.status === "failed") {
                setAnnotationBusy(false);
                showError("annotation", job.error || job.message || "Entity extraction failed.");
                await loadActivity();
                return;
            }
        } catch (error) {
            setAnnotationBusy(false);
            showError("annotation", error.message);
            return;
        }
        await sleep(POLL_MS);
    }
}

async function pollRelation(jobId, generation) {
    while (generation === state.relation.generation) {
        try {
            const job = await requestJson(`/api/relations/${encodeURIComponent(jobId)}`);
            updateProgress("relation", job);
            if (job.status === "completed") {
                setRelationBusy(false);
                await openRelationSummary(jobId, false);
                await loadActivity(null, job.source_annotation_job_id, jobId);
                showToast(job.reused ? "A matching Stage 3 result was reused." : "Relation extraction finished.");
                return;
            }
            if (job.status === "failed") {
                setRelationBusy(false);
                showError("relation", job.error || job.message || "Relation extraction failed.");
                await loadActivity();
                return;
            }
        } catch (error) {
            setRelationBusy(false);
            showError("relation", error.message);
            return;
        }
        await sleep(RELATION_POLL_MS);
    }
}

async function pollNetwork(jobId, generation) {
    while (generation === state.network.generation) {
        try {
            const job = await requestJson(`/api/networks/${encodeURIComponent(jobId)}`);
            updateProgress("network", job);
            if (job.status === "completed") {
                setNetworkBusy(false);
                await openNetworkSummary(jobId, false);
                await loadActivity(null, null, job.source_relation_job_id, jobId);
                showToast("Interaction network is ready.");
                return;
            }
            if (job.status === "failed") {
                setNetworkBusy(false);
                showError("network", job.error || job.message || "Network generation failed.");
                await loadActivity();
                return;
            }
        } catch (error) {
            setNetworkBusy(false);
            showError("network", error.message);
            return;
        }
        await sleep(NETWORK_POLL_MS);
    }
}

async function submitRetrieval(event) {
    event?.preventDefault();
    if (state.retrieval.busy) return;
    clearError("retrieval");
    $("#retrievalSummary").hidden = true;
    setRetrievalBusy(true);
    updateProgress("retrieval", { status: "queued", stage: "queued", progress: 1, message: "Submitting the retrieval job…" });
    try {
        const job = await requestJson("/api/jobs", {
            method: "POST",
            body: JSON.stringify({ input_type: "keywords", query: $("#queryInput").value.trim() }),
        });
        state.retrieval.currentJobId = job.id;
        state.retrieval.generation += 1;
        updateProgress("retrieval", job);
        void pollRetrieval(job.id, state.retrieval.generation);
        await loadActivity();
    } catch (error) {
        setRetrievalBusy(false);
        showError("retrieval", error.message);
        updateProgress("retrieval", { status: "failed", stage: "failed", progress: 100, message: error.message });
    }
}

async function submitAnnotation(event) {
    event?.preventDefault();
    if (state.annotation.busy) return;
    const sourceJobId = state.annotation.sourceRetrievalId;
    if (!sourceJobId) {
        showError("annotation", "Complete a retrieval first.");
        return;
    }
    clearError("annotation");
    $("#annotationSummary").hidden = true;
    setAnnotationBusy(true);
    updateProgress("annotation", {
        status: "queued",
        stage: state.annotation.executor === "local" ? "starting_local" : "starting_gpu",
        progress: 1,
        message: "Starting CellExLink and PubTator3 entity extraction…",
    });
    try {
        const job = await requestJson("/api/annotations", {
            method: "POST",
            body: JSON.stringify({ source_job_id: sourceJobId }),
        });
        state.annotation.currentJobId = job.id;
        updateProgress("annotation", job);
        if (job.status === "completed") {
            setAnnotationBusy(false);
            await openAnnotationSummary(job.id, false);
            await loadActivity(sourceJobId, job.id, null);
            return;
        }
        state.annotation.generation += 1;
        void pollAnnotation(job.id, state.annotation.generation);
        await loadActivity(sourceJobId, job.id, null);
    } catch (error) {
        setAnnotationBusy(false);
        showError("annotation", error.message);
        updateProgress("annotation", { status: "failed", stage: "failed", progress: 100, message: error.message });
    }
}

async function submitRelation(event) {
    event?.preventDefault();
    if (state.relation.busy) return;
    const sourceAnnotationJobId = state.relation.sourceAnnotationId;
    if (!sourceAnnotationJobId) {
        showError("relation", "Complete entity extraction first.");
        return;
    }
    clearError("relation");
    $("#relationSummary").hidden = true;
    setRelationBusy(true);
    updateProgress("relation", {
        status: "queued",
        stage: "preparing_sources",
        progress: 1,
        message: "Starting resumable relation extraction…",
    });
    try {
        const job = await requestJson("/api/relations", {
            method: "POST",
            body: JSON.stringify({ source_annotation_job_id: sourceAnnotationJobId }),
        });
        state.relation.currentJobId = job.id;
        updateProgress("relation", job);
        if (job.status === "completed") {
            setRelationBusy(false);
            await openRelationSummary(job.id, false);
            await loadActivity(null, sourceAnnotationJobId, job.id);
            return;
        }
        state.relation.generation += 1;
        void pollRelation(job.id, state.relation.generation);
        await loadActivity(null, sourceAnnotationJobId, job.id);
    } catch (error) {
        setRelationBusy(false);
        showError("relation", error.message);
        updateProgress("relation", { status: "failed", stage: "failed", progress: 100, message: error.message });
    }
}

async function submitNetwork(event) {
    event?.preventDefault();
    if (state.network.busy) return;
    const sourceRelationJobId = state.network.sourceRelationId;
    if (!sourceRelationJobId) {
        showError("network", "Complete relation extraction first.");
        return;
    }
    clearError("network");
    $("#networkSummary").hidden = true;
    setNetworkBusy(true);
    updateProgress("network", {
        status: "queued",
        stage: "preparing_network_sources",
        progress: 1,
        message: "Creating the global entity index and network job…",
    });
    try {
        const job = await requestJson("/api/networks", {
            method: "POST",
            body: JSON.stringify({ source_relation_job_id: sourceRelationJobId }),
        });
        state.network.currentJobId = job.id;
        updateProgress("network", job);
        if (job.status === "completed") {
            setNetworkBusy(false);
            await openNetworkSummary(job.id, false);
            await loadActivity(null, null, sourceRelationJobId, job.id);
            return;
        }
        state.network.generation += 1;
        void pollNetwork(job.id, state.network.generation);
        await loadActivity(null, null, sourceRelationJobId, job.id);
    } catch (error) {
        setNetworkBusy(false);
        showError("network", error.message);
        updateProgress("network", { status: "failed", stage: "failed", progress: 100, message: error.message });
    }
}

function jobRow({ job, title, detail, kind }) {
    const status = String(job.status || "queued");
    return `
        <button class="job-row" type="button" data-kind="${escapeHtml(kind)}" data-job-id="${escapeHtml(job.id)}" data-status="${escapeHtml(status)}">
            <span class="job-title"><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small></span>
            <span class="job-meta"><span class="job-status ${escapeHtml(status)}">${escapeHtml(status)}</span><small>${escapeHtml(formatDate(job.completed_at || job.updated_at || job.created_at))}</small></span>
        </button>`;
}

function renderRetrievalJobs(jobs) {
    const list = $("#jobsList");
    if (!list) return;
    if (!jobs.length) {
        list.innerHTML = '<div class="empty-state">No retrieval jobs yet.</div>';
        return;
    }
    list.innerHTML = jobs.map((job) => jobRow({
        job,
        title: retrievalLabel(job),
        detail: `${Number(job.stats?.paper_count ?? job.paper_count ?? 0).toLocaleString()} papers · ${Math.round(Number(job.progress || 0))}%`,
        kind: "retrieval",
    })).join("");
}

function renderAnnotationJobs(jobs) {
    const list = $("#annotationJobsList");
    if (!list) return;
    if (!jobs.length) {
        list.innerHTML = '<div class="empty-state">No entity-extraction jobs yet.</div>';
        return;
    }
    list.innerHTML = jobs.map((job) => {
        const entities = Number(job.stats?.entity_count ?? job.mention_count ?? 0);
        return jobRow({
            job,
            title: annotationLabel(job),
            detail: `${entities.toLocaleString()} annotations · ${Math.round(Number(job.progress || 0))}%`,
            kind: "annotation",
        });
    }).join("");
}

function renderRelationJobs(jobs) {
    const list = $("#relationJobsList");
    if (!list) return;
    if (!jobs.length) {
        list.innerHTML = '<div class="empty-state">No relation-extraction jobs yet.</div>';
        return;
    }
    list.innerHTML = jobs.map((job) => jobRow({
        job,
        title: relationLabel(job),
        detail: `${Number(job.relation_count || 0).toLocaleString()} relations · ${Math.round(Number(job.progress || 0))}%`,
        kind: "relation",
    })).join("");
}

function renderNetworkJobs(jobs) {
    const list = $("#networkJobsList");
    if (!list) return;
    if (!jobs.length) {
        list.innerHTML = '<div class="empty-state">No network jobs yet.</div>';
        return;
    }
    list.innerHTML = jobs.map((job) => jobRow({
        job,
        title: networkLabel(job),
        detail: `${Number(job.node_count || 0).toLocaleString()} nodes · ${Number(job.edge_count || 0).toLocaleString()} edges · ${Math.round(Number(job.progress || 0))}%`,
        kind: "network",
    })).join("");
}

function populateRetrievalSources(jobs, preferredId = null) {
    const previous = preferredId || state.annotation.sourceRetrievalId;
    state.completedRetrievals = jobs.filter((job) => job.status === "completed");
    if (!state.completedRetrievals.length) {
        state.annotation.sourceRetrievalId = null;
        setAnnotationBusy(state.annotation.busy);
        setStageAvailability("annotation", false);
        return;
    }
    state.annotation.sourceRetrievalId = state.completedRetrievals.some((job) => job.id === previous)
        ? previous
        : state.completedRetrievals[0].id;
    setAnnotationBusy(state.annotation.busy);
    setStageAvailability("annotation", Boolean(state.annotation.sourceRetrievalId && state.annotation.configured));
}

function populateRelationSources(jobs, preferredId = null) {
    const previous = preferredId || state.relation.sourceAnnotationId;
    state.completedAnnotations = jobs.filter((job) => job.status === "completed");
    if (!state.completedAnnotations.length) {
        state.relation.sourceAnnotationId = null;
        setRelationBusy(state.relation.busy);
        setStageAvailability("relation", false);
        return;
    }
    state.relation.sourceAnnotationId = state.completedAnnotations.some((job) => job.id === previous)
        ? previous
        : state.completedAnnotations[0].id;
    setRelationBusy(state.relation.busy);
    setStageAvailability("relation", Boolean(state.relation.sourceAnnotationId && state.relation.configured));
}

function populateNetworkSources(jobs, preferredId = null) {
    const previous = preferredId || state.network.sourceRelationId;
    state.completedRelations = jobs.filter((job) => job.status === "completed");
    if (!state.completedRelations.length) {
        state.network.sourceRelationId = null;
        setNetworkBusy(state.network.busy);
        setStageAvailability("network", false);
        return;
    }
    state.network.sourceRelationId = state.completedRelations.some((job) => job.id === previous)
        ? previous
        : state.completedRelations[0].id;
    setNetworkBusy(state.network.busy);
    setStageAvailability("network", Boolean(state.network.sourceRelationId && state.network.configured));
}

function selectRetrieval(jobId, scroll = true) {
    if (state.completedRetrievals.some((job) => job.id === jobId)) {
        state.annotation.sourceRetrievalId = jobId;
        setAnnotationBusy(state.annotation.busy);
    }
    if (scroll) activateStage("annotation", { scroll: true, focusTab: true });
}

function selectAnnotationForRelations(jobId, scroll = true) {
    if (state.completedAnnotations.some((job) => job.id === jobId)) {
        state.relation.sourceAnnotationId = jobId;
        setRelationBusy(state.relation.busy);
    }
    if (scroll) activateStage("relation", { scroll: true, focusTab: true });
}

function selectRelationForNetwork(jobId, scroll = true) {
    if (state.completedRelations.some((job) => job.id === jobId)) {
        state.network.sourceRelationId = jobId;
        setNetworkBusy(state.network.busy);
    }
    if (scroll) activateStage("network", { scroll: true, focusTab: true });
}

async function openActivityJob(button) {
    const { kind, jobId, status } = button.dataset;
    try {
        if (kind === "retrieval") {
            if (status === "completed") return await openRetrievalSummary(jobId, true);
            const job = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}`);
            updateProgress("retrieval", job);
            if (["queued", "processing"].includes(job.status)) {
                state.retrieval.currentJobId = job.id;
                state.retrieval.generation += 1;
                setRetrievalBusy(true);
                void pollRetrieval(job.id, state.retrieval.generation);
            } else if (job.status === "failed") showError("retrieval", job.error || job.message);
            activateStage("retrieval", { scroll: true, focusTab: true });
            return;
        }
        if (kind === "annotation") {
            if (status === "completed") return await openAnnotationSummary(jobId, true);
            const job = await requestJson(`/api/annotations/${encodeURIComponent(jobId)}`);
            updateProgress("annotation", job);
            selectRetrieval(job.source_job_id, false);
            if (["queued", "processing"].includes(job.status)) {
                state.annotation.currentJobId = job.id;
                state.annotation.generation += 1;
                setAnnotationBusy(true);
                void pollAnnotation(job.id, state.annotation.generation);
            } else if (job.status === "failed") showError("annotation", job.error || job.message);
            activateStage("annotation", { scroll: true, focusTab: true });
            return;
        }
        if (kind === "relation") {
            if (status === "completed") return await openRelationSummary(jobId, true);
            const job = await requestJson(`/api/relations/${encodeURIComponent(jobId)}`);
            updateProgress("relation", job);
            selectAnnotationForRelations(job.source_annotation_job_id, false);
            if (["queued", "processing"].includes(job.status)) {
                state.relation.currentJobId = job.id;
                state.relation.generation += 1;
                setRelationBusy(true);
                void pollRelation(job.id, state.relation.generation);
            } else if (job.status === "failed") showError("relation", job.error || job.message);
            activateStage("relation", { scroll: true, focusTab: true });
            return;
        }
        if (kind === "network") {
            window.location.assign(`/network/${encodeURIComponent(jobId)}`);
            return;
        }
    } catch (error) {
        showToast(error.message);
    }
}

async function loadActivity(
    preferredRetrievalId = null,
    preferredAnnotationId = null,
    preferredRelationId = null,
    preferredNetworkId = null,
) {
    const [retrievalPayload, annotationPayload, relationPayload, networkPayload] = await Promise.all([
        requestJson("/api/jobs?limit=10"),
        requestJson("/api/annotations?limit=10"),
        requestJson("/api/relations?limit=10"),
        requestJson("/api/networks?limit=10"),
    ]);
    const retrievalJobs = Array.isArray(retrievalPayload?.jobs) ? retrievalPayload.jobs : [];
    const annotationJobs = Array.isArray(annotationPayload?.jobs) ? annotationPayload.jobs : [];
    const relationJobs = Array.isArray(relationPayload?.jobs) ? relationPayload.jobs : [];
    const networkJobs = Array.isArray(networkPayload?.jobs) ? networkPayload.jobs : [];
    state.completedNetworks = networkJobs.filter((job) => job.status === "completed");
    renderRetrievalJobs(retrievalJobs);
    renderAnnotationJobs(annotationJobs);
    renderRelationJobs(relationJobs);
    renderNetworkJobs(networkJobs);
    populateRetrievalSources(retrievalJobs, preferredRetrievalId);
    populateRelationSources(annotationJobs, preferredAnnotationId);
    populateNetworkSources(relationJobs, preferredRelationId);
    if (preferredNetworkId) {
        const preferred = networkJobs.find((job) => job.id === preferredNetworkId);
        if (preferred) updateProgress("network", preferred);
    }
    return { retrievalJobs, annotationJobs, relationJobs, networkJobs };
}

async function initializePipelineStatus() {
    const [annotationResult, relationResult, networkResult] = await Promise.allSettled([
        requestJson("/api/annotations/status"),
        requestJson("/api/relations/status"),
        requestJson("/api/networks/status"),
    ]);

    if (annotationResult.status === "fulfilled") {
        const pipeline = annotationResult.value;
        state.annotation.executor = pipeline?.executor || "disabled";
        state.annotation.compute = pipeline?.compute || "entity worker";
        state.annotation.configured = pipeline?.status === "connected";
    } else {
        state.annotation.configured = false;
        console.error(annotationResult.reason);
    }

    if (relationResult.status === "fulfilled") {
        const pipeline = relationResult.value;
        state.relation.model = pipeline?.model || "gpt-5.4-nano";
        state.relation.compute = pipeline?.compute || "OpenAI Responses API";
        state.relation.windowSize = Number(pipeline?.window_size || 500);
        state.relation.concurrency = Number(pipeline?.concurrency || 8);
        state.relation.configured = pipeline?.status === "connected";
    } else {
        state.relation.configured = false;
        console.error(relationResult.reason);
    }

    if (networkResult.status === "fulfilled") {
        const pipeline = networkResult.value;
        state.network.compute = pipeline?.compute || "Local/Railway CPU · SQLite · PyVis";
        state.network.configured = pipeline?.status === "connected";
    } else {
        state.network.configured = false;
        console.error(networkResult.reason);
    }
    setAnnotationBusy(state.annotation.busy);
    setRelationBusy(state.relation.busy);
    setNetworkBusy(state.network.busy);
}

async function initialize() {
    initializeStageNavigation();
    $("#analysisForm")?.addEventListener("submit", submitRetrieval);
    $("#annotationForm")?.addEventListener("submit", submitAnnotation);
    $("#relationForm")?.addEventListener("submit", submitRelation);
    $("#networkForm")?.addEventListener("submit", submitNetwork);
    $("#refreshJobs")?.addEventListener("click", async () => {
        try { await loadActivity(); showToast("Recent activity refreshed."); }
        catch (error) { showToast(error.message); }
    });
    $("#recent-jobs")?.addEventListener("click", (event) => {
        const button = event.target.closest("[data-kind][data-job-id]");
        if (button) void openActivityJob(button);
    });

    await initializePipelineStatus();

    let activity = { retrievalJobs: [], annotationJobs: [], relationJobs: [], networkJobs: [] };
    try { activity = await loadActivity(); }
    catch (error) { showToast(error.message); }

    const activeRetrieval = activity.retrievalJobs.find((job) => ["queued", "processing"].includes(job.status));
    if (activeRetrieval) {
        state.retrieval.currentJobId = activeRetrieval.id;
        state.retrieval.generation += 1;
        setRetrievalBusy(true);
        updateProgress("retrieval", activeRetrieval);
        void pollRetrieval(activeRetrieval.id, state.retrieval.generation);
    }
    const activeAnnotation = activity.annotationJobs.find((job) => ["queued", "processing"].includes(job.status));
    if (activeAnnotation) {
        state.annotation.currentJobId = activeAnnotation.id;
        state.annotation.generation += 1;
        selectRetrieval(activeAnnotation.source_job_id, false);
        setAnnotationBusy(true);
        updateProgress("annotation", activeAnnotation);
        void pollAnnotation(activeAnnotation.id, state.annotation.generation);
    }
    const activeRelation = activity.relationJobs.find((job) => ["queued", "processing"].includes(job.status));
    if (activeRelation) {
        state.relation.currentJobId = activeRelation.id;
        state.relation.generation += 1;
        selectAnnotationForRelations(activeRelation.source_annotation_job_id, false);
        setRelationBusy(true);
        updateProgress("relation", activeRelation);
        void pollRelation(activeRelation.id, state.relation.generation);
    }
    const activeNetwork = activity.networkJobs.find((job) => ["queued", "processing"].includes(job.status));
    if (activeNetwork) {
        state.network.currentJobId = activeNetwork.id;
        state.network.generation += 1;
        selectRelationForNetwork(activeNetwork.source_relation_job_id, false);
        setNetworkBusy(true);
        updateProgress("network", activeNetwork);
        void pollNetwork(activeNetwork.id, state.network.generation);
    }

    const latestRetrieval = activity.retrievalJobs.find((job) => job.status === "completed");
    if (latestRetrieval && !activeRetrieval) {
        try { await openRetrievalSummary(latestRetrieval.id, false); } catch (error) { console.error(error); }
    }
    const latestAnnotation = activity.annotationJobs.find((job) => job.status === "completed");
    if (latestAnnotation && !activeAnnotation) {
        try { await openAnnotationSummary(latestAnnotation.id, false); } catch (error) { console.error(error); }
    }
    const latestRelation = activity.relationJobs.find((job) => job.status === "completed");
    if (latestRelation && !activeRelation) {
        try { await openRelationSummary(latestRelation.id, false); } catch (error) { console.error(error); }
    }
    const latestNetwork = activity.networkJobs.find((job) => job.status === "completed");
    if (latestNetwork && !activeNetwork) {
        try { await openNetworkSummary(latestNetwork.id, false); } catch (error) { console.error(error); }
    }

    const initialStage = activeNetwork
        ? "network"
        : activeRelation
            ? "relation"
            : activeAnnotation
                ? "annotation"
                : activeRetrieval
                    ? "retrieval"
                    : latestNetwork
                        ? "network"
                        : latestRelation
                            ? "network"
                            : latestAnnotation
                                ? "relation"
                                : latestRetrieval
                                    ? "annotation"
                                    : "retrieval";
    activateStage(initialStage);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void initialize(), { once: true });
} else {
    void initialize();
}
