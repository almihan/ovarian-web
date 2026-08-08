"use strict";

const state = {
    sessionApiKey: null,
    serverApiKeyConfigured: false,
    currentJobId: null,
    pollGeneration: 0,
    resultNetwork: null,
    networkPayload: null,
    nodes: null,
    edges: null,
    toastTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function sleep(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
        ...options,
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
            ? detail.map((item) => item.msg).join(" ")
            : detail || `Request failed with status ${response.status}.`;
        throw new Error(message);
    }
    return payload;
}

function showToast(message) {
    const toast = $("#toast");
    $("#toastMessage").textContent = message;
    toast.hidden = false;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => {
        toast.hidden = true;
    }, 3600);
}

function updateCredentialIndicator() {
    const hasCredential = Boolean(state.sessionApiKey || state.serverApiKeyConfigured);
    const dot = $("#credentialDot");
    dot.classList.toggle("ready", hasCredential);
    dot.setAttribute(
        "aria-label",
        hasCredential ? "OpenAI API key is available" : "No OpenAI API key is available",
    );
}

async function loadSystemStatus(attempt = 0) {
    const modelStatus = $("#modelStatus");
    const serverKeyStatus = $("#serverKeyStatus");

    try {
        const data = await requestJson("/api/system/status");
        const model = data.model || {};
        modelStatus.className = `model-status ${escapeHtml(model.state || "")}`;
        $("strong", modelStatus).textContent = "CellExLink model";
        $("small", modelStatus).textContent = model.message || "Status unavailable.";

        state.serverApiKeyConfigured = Boolean(data.server_openai_key_configured);
        serverKeyStatus.classList.toggle("ready", state.serverApiKeyConfigured);
        $("p", serverKeyStatus).textContent = state.serverApiKeyConfigured
            ? "A project-level OpenAI API key is configured securely on Railway."
            : "No project-level OpenAI API key is configured; users may provide a session key.";
        updateCredentialIndicator();

        if (model.state === "loading" && attempt < 90) {
            window.setTimeout(() => loadSystemStatus(attempt + 1), 4000);
        }
    } catch (error) {
        modelStatus.className = "model-status error";
        $("small", modelStatus).textContent = "Could not read model status.";
        serverKeyStatus.classList.remove("ready");
        $("p", serverKeyStatus).textContent = "Could not read server API-key status.";
        console.error(error);
    }
}

function openApiModal() {
    const modal = $("#apiModal");
    modal.hidden = false;
    document.body.classList.add("modal-open");
    $("#apiKeyError").hidden = true;
    $("#apiKeyInput").value = "";
    $("#apiKeyInput").placeholder = state.sessionApiKey
        ? "A session key is already configured"
        : "sk-...";
    window.setTimeout(() => $("#apiKeyInput").focus(), 50);
}

function closeApiModal() {
    $("#apiModal").hidden = true;
    document.body.classList.remove("modal-open");
    $("#apiKeyInput").value = "";
    $("#apiKeyInput").type = "password";
}

function saveSessionApiKey() {
    const input = $("#apiKeyInput");
    const errorElement = $("#apiKeyError");
    const key = input.value.trim();

    if (key.length < 20 || !key.startsWith("sk-")) {
        errorElement.textContent = "Enter a valid OpenAI API key beginning with “sk-”.";
        errorElement.hidden = false;
        input.focus();
        return;
    }

    state.sessionApiKey = key;
    input.value = ""; // Do not leave the secret in the DOM.
    errorElement.hidden = true;
    updateCredentialIndicator();
    closeApiModal();
    showToast("Session API key is ready for relation extraction.");
}

function removeSessionApiKey() {
    state.sessionApiKey = null;
    $("#apiKeyInput").value = "";
    updateCredentialIndicator();
    closeApiModal();
    showToast("The session API key was removed.");
}

function bindApiModal() {
    $("#openApiSettings").addEventListener("click", openApiModal);
    $("#closeApiSettings").addEventListener("click", closeApiModal);
    $("#saveApiKey").addEventListener("click", saveSessionApiKey);
    $("#removeApiKey").addEventListener("click", removeSessionApiKey);

    $("#toggleApiKey").addEventListener("click", () => {
        const input = $("#apiKeyInput");
        input.type = input.type === "password" ? "text" : "password";
    });

    $("#apiKeyInput").addEventListener("keydown", (event) => {
        if (event.key === "Enter") saveSessionApiKey();
    });

    $("#apiModal").addEventListener("click", (event) => {
        if (event.target === event.currentTarget) closeApiModal();
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !$("#apiModal").hidden) closeApiModal();
    });
}

function updateInputType() {
    const type = $("#inputType").value;
    const input = $("#queryInput");
    const examples = $("#searchExamples");

    const content = {
        keywords: {
            placeholder: "e.g., macrophage AND ovarian cancer",
            examples: "<strong>Examples:</strong> macrophage AND ovarian cancer <span>•</span> IL6 AND ovarian cancer <span>•</span> tumor microenvironment",
        },
        pmid: {
            placeholder: "e.g., 34567890, 35678901",
            examples: "<strong>Examples:</strong> 34567890 <span>•</span> 34567890, 35678901 <span>•</span> one PMID per line",
        },
        pmcid: {
            placeholder: "e.g., PMC1234567, PMC7654321",
            examples: "<strong>Examples:</strong> PMC1234567 <span>•</span> PMC1234567, PMC7654321 <span>•</span> one PMCID per line",
        },
    };

    input.placeholder = content[type].placeholder;
    examples.innerHTML = content[type].examples;
}

function setSubmitState(loading) {
    const button = $("#startAnalysis");
    button.disabled = loading;
    button.classList.toggle("loading", loading);
    $("span", button).textContent = loading ? "Submitting..." : "Start Analysis";
}

function resetProgressSteps() {
    $$(".progress-step").forEach((step) => step.classList.remove("active", "complete"));
}

function updateProgress(job) {
    const panel = $("#progressPanel");
    panel.hidden = false;
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    $("#progressBar").style.width = `${progress}%`;
    $("#progressPercent").textContent = `${progress}%`;
    $("#progressMessage").textContent = job.message || "Processing...";
    $("#progressJobId").textContent = `Job ${job.id}`;

    const order = ["retrieving", "entities", "relations", "network"];
    const activeIndex = order.indexOf(job.stage);
    resetProgressSteps();

    $$(".progress-step").forEach((step) => {
        const index = order.indexOf(step.dataset.stage);
        if (job.status === "completed" || (activeIndex >= 0 && index < activeIndex)) {
            step.classList.add("complete");
        } else if (index === activeIndex) {
            step.classList.add("active");
        }
    });
}

async function submitAnalysis(event) {
    event.preventDefault();
    const queryInput = $("#queryInput");
    const formError = $("#formError");
    const query = queryInput.value.trim();
    const relationExtraction = $("#relationExtraction").checked;

    formError.hidden = true;
    if (!query) {
        formError.textContent = "Enter keywords, PMID values, or PMCID values.";
        formError.hidden = false;
        queryInput.focus();
        return;
    }

    if (relationExtraction && !state.sessionApiKey && !state.serverApiKeyConfigured) {
        showToast("No API key is configured; this starter runs the relation stage with demonstration data.");
    }

    setSubmitState(true);
    state.pollGeneration += 1;

    try {
        const body = {
            input_type: $("#inputType").value,
            query,
            relation_extraction: relationExtraction,
        };
        if (state.sessionApiKey) body.api_key = state.sessionApiKey;

        const job = await requestJson("/api/jobs", {
            method: "POST",
            body: JSON.stringify(body),
        });

        state.currentJobId = job.id;
        updateProgress(job);
        $("#progressPanel").scrollIntoView({ behavior: "smooth", block: "center" });
        await pollJob(job.id, state.pollGeneration);
    } catch (error) {
        formError.textContent = error.message;
        formError.hidden = false;
        setSubmitState(false);
    }
}

async function pollJob(jobId, generation) {
    try {
        while (generation === state.pollGeneration) {
            const job = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}`);
            updateProgress(job);
            await loadRecentJobs();

            if (job.status === "completed") {
                setSubmitState(false);
                showToast("Analysis completed. Your network is ready.");
                await openNetwork(jobId, true);
                return;
            }

            if (job.status === "failed") {
                setSubmitState(false);
                const errorElement = $("#formError");
                errorElement.textContent = job.error || "The analysis failed.";
                errorElement.hidden = false;
                return;
            }
            await sleep(900);
        }
    } catch (error) {
        setSubmitState(false);
        const errorElement = $("#formError");
        errorElement.textContent = error.message;
        errorElement.hidden = false;
    }
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

function renderJobs(jobs) {
    const list = $("#jobsList");
    if (!jobs.length) {
        list.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon"><svg class="icon"><use href="#icon-clock"></use></svg></span>
                <div><strong>No analyses yet</strong><p>Submit a search above and its progress will appear here.</p></div>
            </div>`;
        return;
    }

    list.innerHTML = jobs.map((job) => {
        const completedMetric = job.status === "completed"
            ? `${job.paper_count} papers · ${job.entity_count} entities`
            : escapeHtml(job.message || job.stage);
        const action = job.status === "completed"
            ? `<button class="job-open" type="button" data-open-job="${escapeHtml(job.id)}">Open network</button>`
            : `<div class="job-progress-mini"><span><i style="width:${Number(job.progress || 0)}%"></i></span><small>${Number(job.progress || 0)}%</small></div>`;

        return `
            <article class="job-row">
                <div class="job-query">
                    <strong title="${escapeHtml(job.query)}">${escapeHtml(job.query)}</strong>
                    <small>${escapeHtml(job.input_type)} · Job ${escapeHtml(job.id)}</small>
                </div>
                <span class="status-badge ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>
                <span class="job-metric">${completedMetric}</span>
                <span class="job-date">${escapeHtml(formatDate(job.created_at))}</span>
                ${action}
            </article>`;
    }).join("");
}

async function loadRecentJobs() {
    try {
        const data = await requestJson("/api/jobs?limit=8");
        renderJobs(data.jobs || []);
    } catch (error) {
        console.error("Could not load recent jobs", error);
    }
}

function groupLabel(group) {
    return ({ cell: "Cell type", gene: "Gene", chemical: "Chemical" })[group] || "Entity";
}

function renderEvidencePlaceholder() {
    $("#evidencePanel").innerHTML = `
        <div class="evidence-placeholder">
            <span><svg class="icon"><use href="#icon-info"></use></svg></span>
            <h3>Select an entity or relation</h3>
            <p>Supporting identifiers, confidence, and literature evidence will appear here.</p>
        </div>`;
}

function renderNodeEvidence(nodeId) {
    const payload = state.networkPayload;
    const node = payload?.nodes.find((item) => String(item.id) === String(nodeId));
    if (!node) return renderEvidencePlaceholder();

    const connections = (payload.edges || []).filter(
        (edge) => String(edge.from) === String(node.id) || String(edge.to) === String(node.id),
    );
    const connectedHtml = connections.slice(0, 8).map((edge) => {
        const otherId = String(edge.from) === String(node.id) ? edge.to : edge.from;
        const other = payload.nodes.find((item) => String(item.id) === String(otherId));
        return `<div class="connection-pill"><span>${escapeHtml(edge.label)}</span><em>${escapeHtml(other?.label || otherId)}</em></div>`;
    }).join("");

    $("#evidencePanel").innerHTML = `
        <span class="evidence-type"><i class="legend-dot ${escapeHtml(node.group)}"></i>${escapeHtml(groupLabel(node.group))}</span>
        <h3>${escapeHtml(node.label)}</h3>
        <p class="evidence-id">${escapeHtml(node.id)}</p>
        <p class="evidence-description">${escapeHtml(node.description || "No description is available.")}</p>
        <div class="evidence-meta">
            <div><span>Mentions</span><strong>${Number(node.count || 0)}</strong></div>
            <div><span>Relations</span><strong>${connections.length}</strong></div>
        </div>
        <div class="evidence-connected">
            <strong>Connected relations</strong>
            ${connectedHtml || "<p class='evidence-description'>No relations are available.</p>"}
        </div>`;
}

function renderEdgeEvidence(edgeId) {
    const payload = state.networkPayload;
    const edge = payload?.edges.find((item) => String(item.id) === String(edgeId));
    if (!edge) return renderEvidencePlaceholder();

    const source = payload.nodes.find((item) => String(item.id) === String(edge.from));
    const target = payload.nodes.find((item) => String(item.id) === String(edge.to));
    const confidence = Math.round(Number(edge.confidence || 0) * 100);

    $("#evidencePanel").innerHTML = `
        <span class="evidence-type">Extracted relation</span>
        <h3>${escapeHtml(edge.label)}</h3>
        <p class="evidence-id">${escapeHtml(source?.label || edge.from)} → ${escapeHtml(target?.label || edge.to)}</p>
        <div class="evidence-meta">
            <div><span>Confidence</span><strong>${confidence}%</strong></div>
            <div><span>PMID</span><strong>${escapeHtml(edge.pmid || "—")}</strong></div>
        </div>
        <div class="evidence-quote">“${escapeHtml(edge.evidence || "No supporting sentence is available.")}”</div>`;
}

function buildResultNetwork(payload) {
    if (!window.vis) {
        $("#resultNetwork").innerHTML = "<p>vis-network could not be loaded.</p>";
        return;
    }

    if (state.resultNetwork) {
        state.resultNetwork.destroy();
        state.resultNetwork = null;
    }

    const nodeData = payload.nodes.map((node) => ({
        ...node,
        size: 17 + Math.min(Number(node.count || 0), 22) * 0.55,
        title: `${node.label} (${node.id})`,
    }));
    const edgeData = payload.edges.map((edge) => ({
        ...edge,
        width: 1 + Number(edge.confidence || 0) * 1.7,
        arrows: { to: { enabled: true, scaleFactor: 0.55 } },
    }));

    state.nodes = new vis.DataSet(nodeData);
    state.edges = new vis.DataSet(edgeData);
    const container = $("#resultNetwork");

    state.resultNetwork = new vis.Network(container, {
        nodes: state.nodes,
        edges: state.edges,
    }, {
        autoResize: true,
        layout: { improvedLayout: true },
        physics: {
            enabled: true,
            stabilization: { iterations: 260, updateInterval: 25 },
            barnesHut: {
                gravitationalConstant: -4200,
                centralGravity: 0.18,
                springLength: 150,
                springConstant: 0.035,
                damping: 0.15,
                avoidOverlap: 0.48,
            },
        },
        interaction: {
            hover: true,
            tooltipDelay: 120,
            navigationButtons: true,
            keyboard: { enabled: true, bindToWindow: false },
        },
        nodes: {
            shape: "dot",
            borderWidth: 2,
            borderWidthSelected: 4,
            font: { color: "#302b49", size: 12, face: "Inter", vadjust: 6 },
            shadow: { enabled: true, color: "rgba(39,30,88,.16)", size: 12, x: 0, y: 5 },
        },
        groups: {
            cell: {
                color: {
                    background: "#a276ec",
                    border: "#6742c8",
                    highlight: { background: "#b58ef2", border: "#5532b9" },
                    hover: { background: "#ad84ef", border: "#5d38bd" },
                },
            },
            gene: {
                color: {
                    background: "#67a9ed",
                    border: "#347ac5",
                    highlight: { background: "#7db7f1", border: "#286bac" },
                    hover: { background: "#74b1ef", border: "#2e70b6" },
                },
            },
            chemical: {
                color: {
                    background: "#62c494",
                    border: "#2b9664",
                    highlight: { background: "#79d0a5", border: "#218455" },
                    hover: { background: "#6bc99a", border: "#268d5d" },
                },
            },
        },
        edges: {
            color: { color: "#bbb7ce", highlight: "#6a4bdc", hover: "#8f7cdb", opacity: 0.88 },
            font: { color: "#7b758a", size: 9, face: "Inter", align: "top", strokeWidth: 4, strokeColor: "#ffffff" },
            smooth: { enabled: true, type: "dynamic" },
            selectionWidth: 2.2,
            hoverWidth: 1.5,
        },
    });

    state.resultNetwork.once("stabilizationIterationsDone", () => {
        state.resultNetwork.setOptions({ physics: { enabled: false } });
        state.resultNetwork.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" } });
    });

    state.resultNetwork.on("selectNode", (params) => renderNodeEvidence(params.nodes[0]));
    state.resultNetwork.on("selectEdge", (params) => {
        if (!params.nodes.length && params.edges.length) renderEdgeEvidence(params.edges[0]);
    });
    state.resultNetwork.on("deselectNode", (params) => {
        if (!params.edges.length) renderEvidencePlaceholder();
    });
    state.resultNetwork.on("deselectEdge", (params) => {
        if (!params.nodes.length) renderEvidencePlaceholder();
    });
}

function renderNetwork(payload) {
    state.networkPayload = payload;
    $("#results-title").textContent = "Interaction Network";
    $("#resultQuery").textContent = `Search: ${payload.query}`;
    $("#paperCount").textContent = payload.summary?.papers ?? 0;
    $("#entityCount").textContent = payload.summary?.entities ?? payload.nodes?.length ?? 0;
    $("#relationCount").textContent = payload.summary?.relations ?? payload.edges?.length ?? 0;
    $("#network-results").hidden = false;
    $("#nodeSearch").value = "";
    $$(".filter-chip").forEach((chip) => chip.classList.toggle("active", chip.dataset.group === "all"));
    renderEvidencePlaceholder();
    buildResultNetwork(payload);
}

async function openNetwork(jobId, shouldScroll = true) {
    try {
        const payload = await requestJson(`/api/networks/${encodeURIComponent(jobId)}`);
        renderNetwork(payload);
        if (shouldScroll) {
            await sleep(80);
            $("#network-results").scrollIntoView({ behavior: "smooth", block: "start" });
        }
    } catch (error) {
        showToast(error.message);
    }
}

function applyGroupFilter(group) {
    if (!state.nodes || !state.networkPayload) return;
    const updates = state.networkPayload.nodes.map((node) => ({
        id: node.id,
        hidden: group !== "all" && node.group !== group,
    }));
    state.nodes.update(updates);
    $$(".filter-chip").forEach((chip) => chip.classList.toggle("active", chip.dataset.group === group));
    renderEvidencePlaceholder();
    window.setTimeout(() => state.resultNetwork?.fit({ animation: { duration: 350 } }), 60);
}

function findNode() {
    const query = $("#nodeSearch").value.trim().toLowerCase();
    if (!query || !state.networkPayload || !state.resultNetwork) return;
    const match = state.networkPayload.nodes.find((node) => node.label.toLowerCase().includes(query));
    if (!match) {
        showToast(`No node matching “${query}” was found.`);
        return;
    }
    state.nodes.update({ id: match.id, hidden: false });
    state.resultNetwork.selectNodes([match.id]);
    state.resultNetwork.focus(match.id, { scale: 1.35, animation: { duration: 500, easingFunction: "easeInOutQuad" } });
    renderNodeEvidence(match.id);
}

function bindNetworkControls() {
    $$(".filter-chip").forEach((chip) => {
        chip.addEventListener("click", () => applyGroupFilter(chip.dataset.group));
    });
    $("#nodeSearch").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            findNode();
        }
    });
}

function bindJobControls() {
    $("#analysisForm").addEventListener("submit", submitAnalysis);
    $("#inputType").addEventListener("change", updateInputType);
    $("#refreshJobs").addEventListener("click", async () => {
        await loadRecentJobs();
        showToast("Recent jobs were refreshed.");
    });
    $("#jobsList").addEventListener("click", (event) => {
        const button = event.target.closest("[data-open-job]");
        if (button) openNetwork(button.dataset.openJob, true);
    });
}

function bindNavigation() {
    $$(".main-nav a").forEach((link) => {
        link.addEventListener("click", () => {
            $$(".main-nav a").forEach((item) => item.classList.remove("active"));
            link.classList.add("active");
        });
    });
}

async function init() {
    bindApiModal();
    bindJobControls();
    bindNetworkControls();
    bindNavigation();
    updateInputType();
    updateCredentialIndicator();
    await Promise.all([loadSystemStatus(), loadRecentJobs()]);
}

document.addEventListener("DOMContentLoaded", init);
