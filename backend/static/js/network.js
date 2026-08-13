"use strict";

const body = document.body;
const jobId = String(body.dataset.networkJobId || "").trim();
const defaultInitialNodes = Number(body.dataset.initialNodes || 120);
const maxInitialNodes = Number(body.dataset.maxInitialNodes || 1000);
const defaultRelationSupportMin = Number(body.dataset.relationSupportMin || 1);
const expansionLimit = Number(body.dataset.expansionLimit || 150);
const hierarchyMaxPaths = Number(body.dataset.hierarchyMaxPaths || 3);
const REQUEST_TIMEOUT_MS = 30000;
const JOB_POLL_MS = 1800;
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
    job: null,
    network: null,
    nodes: null,
    edges: null,
    physicsEnabled: true,
    selectedSearchNode: null,
    selection: null,
    evidence: [],
    searchGeneration: 0,
    nodeSearchResults: [],
    toastTimer: null,
    graphLoaded: false,
    hierarchySignature: "",
    relationTypes: [],
    viewMode: "initial",
};

function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function requestJson(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs || REQUEST_TIMEOUT_MS);
    const fetchOptions = { ...options };
    delete fetchOptions.timeoutMs;
    try {
        const response = await fetch(url, {
            ...fetchOptions,
            headers: { Accept: "application/json", ...(fetchOptions.headers || {}) },
            cache: "no-store",
            signal: controller.signal,
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
        if (error?.name === "AbortError") throw new Error("The server did not respond in time.");
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

function formatNumber(value) {
    return Number(value || 0).toLocaleString();
}

function integerInputValue(selector, fallback, minimum = 1, maximum = Number.MAX_SAFE_INTEGER) {
    const input = $(selector);
    const parsed = Math.trunc(Number(input?.value));
    const safe = Number.isFinite(parsed)
        ? Math.max(minimum, Math.min(maximum, parsed))
        : Math.max(minimum, Math.min(maximum, Math.trunc(Number(fallback) || minimum)));
    if (input) input.value = String(safe);
    return safe;
}

function currentRelationSupportMin() {
    return integerInputValue("#relationSupportMin", defaultRelationSupportMin, 0);
}

function currentTopNodeCount() {
    const input = $("#topNodeCount");
    if (!input?.value.trim() || Number(input.value) <= 0) return defaultInitialNodes;
    return integerInputValue("#topNodeCount", defaultInitialNodes, 1, maxInitialNodes);
}

function formatType(value) {
    const type = String(value || "entity").toLowerCase();
    return type === "gene" ? "Gene / protein" : type === "cell" ? "Cell / cell type" : type === "hormone" ? "Hormone" : "Entity";
}

function isUndirectedPredicate(predicate) {
    return String(predicate || "").toLowerCase() === "binding";
}

function relationEndpointText(subject, predicate, object) {
    const separator = isUndirectedPredicate(predicate) ? " — " : " → ";
    return `${subject || ""}${separator}${object || ""}`;
}

function relationEvidenceText(subject, predicate, object) {
    if (isUndirectedPredicate(predicate)) {
        return `${subject || ""} — ${predicate || "binding"} — ${object || ""}`;
    }
    return `${subject || ""} — ${predicate || "relation"} → ${object || ""}`;
}

function cellOntologyId(value) {
    const match = String(value || "").match(/CL[:_](\d{7})/i);
    return match ? `CL:${match[1]}` : "";
}

function setSelectionActions({ evidence = false } = {}) {
    const evidenceButton = $("#showSelectionEvidence");
    if (evidenceButton) evidenceButton.disabled = !evidence;
}

function showToast(message) {
    const toast = $("#networkToast");
    if (!toast) return;
    toast.textContent = String(message || "");
    toast.hidden = false;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => { toast.hidden = true; }, 3800);
}

function activateTab(name) {
    $$(".tab-button").forEach((button) => {
        const active = button.dataset.tab === name;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
    });
    $$(".tab-panel").forEach((panel) => {
        const active = panel.id === `panel${name[0].toUpperCase()}${name.slice(1)}`;
        panel.classList.toggle("active", active);
        panel.hidden = !active;
    });
}

function setBuildProgress(job) {
    const progress = Math.max(0, Math.min(100, Number(job?.progress || 0)));
    $("#buildProgressBar").style.width = `${progress}%`;
    $("#buildProgressTrack").setAttribute("aria-valuenow", String(Math.round(progress)));
    $("#buildPercent").textContent = `${Math.round(progress)}%`;
    $("#buildStage").textContent = String(job?.stage || "Stage 4").replaceAll("_", " ");
    $("#buildMessage").textContent = job?.message || "Preparing the interaction network";
    $("#networkStatusText").textContent = job?.message || "Preparing network explorer…";
}

function setVisibleCounts() {
    $("#viewNodeCount").textContent = formatNumber(state.nodes?.length || 0);
    $("#viewEdgeCount").textContent = formatNumber(state.edges?.length || 0);
}

function removeNetworkTooltips(items) {
    return items.map((item) => {
        const copy = { ...item };
        delete copy.title;
        return copy;
    });
}

function typePill(type, text = null) {
    const safeType = ["cell", "gene", "hormone"].includes(String(type)) ? String(type) : "edge";
    return `<span class="type-pill ${safeType}">${escapeHtml(text || formatType(type))}</span>`;
}

function detailRows(rows) {
    return `<ul class="detail-list">${rows.map(([label, value]) => `<li><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value ?? "—")}</span></li>`).join("")}</ul>`;
}

function renderNodeDetail(node) {
    const aliases = Array.isArray(node.aliases) ? node.aliases : [];
    $("#detailsHeading").textContent = node.label || node.id;
    $("#detailsSubheading").textContent = `${formatType(node.entity_type)} · normalized across the complete Stage 3 artifact`;
    $("#detailsContent").innerHTML = `
        <div class="detail-card">
            ${typePill(node.entity_type)}
            ${detailRows([
                ["Normalized identity", node.normalized_id],
                ["Standard name", node.label || node.id],
                ["Papers", formatNumber(node.paper_count)],
                ["Chunks", formatNumber(node.chunk_count)],
                ["Relations", formatNumber(node.relation_count)],
            ])}
        </div>
        <div class="detail-card">
            <h3>Text mentions</h3>
            ${aliases.length ? `<ul class="detail-list">${aliases.map((item) => `<li><span>${escapeHtml(item.text)}</span><strong>${formatNumber(item.count)} mention${Number(item.count) === 1 ? "" : "s"}</strong></li>`).join("")}</ul>` : '<p class="empty-copy">No text mentions were stored for this node.</p>'}
        </div>`;
}

function renderEdgeDetail(edge) {
    const undirected = edge.directed === false || isUndirectedPredicate(edge.predicate);
    const firstEndpointLabel = undirected ? "Endpoint 1" : "Subject";
    const secondEndpointLabel = undirected ? "Endpoint 2" : "Object";
    $("#detailsHeading").textContent = edge.predicate || "Relation";
    $("#detailsSubheading").textContent = relationEndpointText(
        edge.subject_label || edge.subject_id,
        edge.predicate,
        edge.object_label || edge.object_id,
    );
    $("#detailsContent").innerHTML = `
        <div class="detail-card">
            ${typePill("edge", undirected ? "Undirected relation" : "Directed relation")}
            ${detailRows([
                [firstEndpointLabel, `${edge.subject_label || edge.subject_id} (${formatType(edge.subject_type)})`],
                ["Predicate", edge.predicate],
                [secondEndpointLabel, `${edge.object_label || edge.object_id} (${formatType(edge.object_type)})`],
                ["Papers", formatNumber(edge.paper_count)],
                ["Chunks / evidence", formatNumber(edge.evidence_count)],
            ])}
        </div>`;
}


function renderOntologyNodeDetail(node) {
    const synonyms = Array.isArray(node.synonyms) ? node.synonyms : [];
    const role = String(node.hierarchy_role || "term").replaceAll("_", " ");
    $("#detailsHeading").textContent = node.label || node.cl_id || node.id;
    $("#detailsSubheading").textContent = "Cell Ontology hierarchy term added to the current interaction view";
    $("#detailsContent").innerHTML = `
        <div class="detail-card">
            ${typePill("cell", "Cell Ontology term")}
            ${detailRows([
                ["Cell Ontology ID", node.cl_id || node.normalized_id],
                ["Hierarchy role", role],
                ["Direct parents", formatNumber(node.parent_count)],
                ["Direct children", formatNumber(node.child_count)],
                ["Interaction evidence", node.ontology_only ? "Not present in extracted network" : "Present in extracted network"],
            ])}
            <p class="ontology-note">This hierarchy-only term has no extracted interaction evidence in the current Stage 4 graph. Its Cell Ontology path is shown in the Hierarchy tab.</p>
        </div>
        <div class="detail-card">
            <h3>Definition</h3>
            <p class="empty-copy">${escapeHtml(node.definition || "No definition is available in this Cell Ontology release.")}</p>
        </div>
        <div class="detail-card">
            <h3>Synonyms</h3>
            ${synonyms.length ? `<ul class="detail-list">${synonyms.map((value) => `<li><span>${escapeHtml(value)}</span></li>`).join("")}</ul>` : '<p class="empty-copy">No synonyms are listed.</p>'}
        </div>`;
}


async function selectNode(nodeId, { openDetails = true } = {}) {
    if (!nodeId) return;
    const visibleNode = state.nodes?.get(nodeId) || null;
    if (visibleNode?.ontology_only || visibleNode?.node_kind === "ontology") {
        const clId = cellOntologyId(visibleNode.cl_id || visibleNode.normalized_id);
        state.selection = {
            kind: "ontology-node",
            id: nodeId,
            label: visibleNode.label || clId || nodeId,
            cl_id: clId,
        };
        renderOntologyNodeDetail(visibleNode);
        setSelectionActions({ evidence: false });
        if (openDetails) activateTab("details");
        return;
    }
    try {
        const node = await requestJson(`/api/networks/${encodeURIComponent(jobId)}/nodes/${encodeURIComponent(nodeId)}`);
        const clId = String(node.entity_type || "").toLowerCase() === "cell"
            ? cellOntologyId(node.normalized_id)
            : "";
        state.selection = { kind: "node", id: nodeId, label: node.label || nodeId, cl_id: clId };
        renderNodeDetail(node);
        setSelectionActions({ evidence: true });
        if (openDetails) activateTab("details");
    } catch (error) {
        showToast(error.message);
    }
}

async function selectEdge(edgeId, { openDetails = true } = {}) {
    if (!edgeId) return;
    try {
        const edge = await requestJson(`/api/networks/${encodeURIComponent(jobId)}/edges/${encodeURIComponent(edgeId)}`);
        state.selection = { kind: "edge", id: edgeId, label: `${edge.subject_label} ${edge.predicate} ${edge.object_label}` };
        renderEdgeDetail(edge);
        setSelectionActions({ evidence: true });
        if (openDetails) activateTab("details");
    } catch (error) {
        showToast(error.message);
    }
}

function evidenceSearchText(item) {
    const contexts = Array.isArray(item.cell_contexts) ? item.cell_contexts.map((entry) => entry.label).join(" ") : "";
    const entities = Array.isArray(item.entities)
        ? item.entities.map((entry) => [entry.mention, entry.preferred_label, entry.normalized_id, entry.entity_type].join(" ")).join(" ")
        : "";
    return [
        item.canonical_id, item.doc_key, item.pmid, item.pmcid, item.doi, item.journal,
        item.pub_year, item.section_type, item.chunk_text, item.predicate,
        item.subject_label, item.object_label, contexts, entities,
    ].join(" ").toLowerCase();
}

function highlightEvidenceText(rawText, rawEntities) {
    const text = String(rawText || "");
    if (!text) return escapeHtml("Passage text is unavailable.");
    const order = { cell: 0, gene: 1, hormone: 2 };
    const spans = (Array.isArray(rawEntities) ? rawEntities : [])
        .map((entity, index) => {
            const start = Math.trunc(Number(entity?.start));
            const end = Math.trunc(Number(entity?.end));
            const entityType = String(entity?.entity_type || "").toLowerCase() === "protein"
                ? "gene"
                : String(entity?.entity_type || "").toLowerCase();
            if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
            if (start < 0 || end <= start || end > text.length) return null;
            if (!Object.prototype.hasOwnProperty.call(order, entityType)) return null;
            return { ...entity, _spanId: index, start, end, entity_type: entityType };
        })
        .filter(Boolean);
    if (!spans.length) return escapeHtml(text);

    // Sweep annotation boundaries once. This preserves nested and overlapping
    // entities without rescanning every annotation for every text segment.
    const starts = new Map();
    const ends = new Map();
    const boundaries = new Set([0, text.length]);
    spans.forEach((span) => {
        boundaries.add(span.start);
        boundaries.add(span.end);
        if (!starts.has(span.start)) starts.set(span.start, []);
        if (!ends.has(span.end)) ends.set(span.end, []);
        starts.get(span.start).push(span);
        ends.get(span.end).push(span);
    });

    const sortedBoundaries = Array.from(boundaries).sort((left, right) => left - right);
    const active = new Map();
    const parts = [];
    for (let index = 0; index < sortedBoundaries.length - 1; index += 1) {
        const start = sortedBoundaries[index];
        const end = sortedBoundaries[index + 1];
        (ends.get(start) || []).forEach((span) => active.delete(span._spanId));
        (starts.get(start) || []).forEach((span) => active.set(span._spanId, span));
        if (end <= start) continue;

        const segment = escapeHtml(text.slice(start, end));
        const activeSpans = Array.from(active.values());
        if (!activeSpans.length) {
            parts.push(segment);
            continue;
        }
        const types = Array.from(new Set(activeSpans.map((span) => span.entity_type)))
            .sort((left, right) => order[left] - order[right]);
        const title = Array.from(new Set(activeSpans.map((span) => {
            const label = span.preferred_label || span.mention || text.slice(span.start, span.end);
            const identifier = span.normalized_id ? ` · ${span.normalized_id}` : "";
            return `${formatType(span.entity_type)}: ${label}${identifier}`;
        }))).join("\n");
        parts.push(
            `<mark class="entity-highlight ${types.map(escapeHtml).join(" ")}" title="${escapeHtml(title)}">${segment}</mark>`,
        );
    }
    return parts.join("");
}

function firstMatch(values, pattern, formatter = (match) => match[0]) {
    for (const value of values) {
        const match = String(value || "").trim().match(pattern);
        if (match) return formatter(match);
    }
    return "";
}

function paperDestinations(item) {
    const candidates = [item?.pmcid, item?.pmid, item?.doi, item?.canonical_id, item?.doc_key];
    const pmcid = firstMatch(candidates, /\bPMC\d+\b/i, (match) => match[0].toUpperCase());
    const pmid = firstMatch(
        [item?.pmid, item?.canonical_id, item?.doc_key],
        /(?:^|\bpmid\s*[:_-]?\s*)(\d+)\b/i,
        (match) => match[1],
    ) || firstMatch([item?.pmid], /^\d+$/, (match) => match[0]);
    const doi = String(item?.doi || "").trim()
        || firstMatch([item?.canonical_id], /^doi\s*:\s*(\S+)$/i, (match) => match[1]);
    const destinations = [];
    if (pmcid) {
        destinations.push({
            kind: "PMC",
            identifier: pmcid,
            url: `https://www.ncbi.nlm.nih.gov/pmc/articles/${encodeURIComponent(pmcid)}/`,
        });
    }
    if (pmid) {
        destinations.push({
            kind: "PubMed",
            identifier: pmid,
            url: `https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(pmid)}/`,
        });
    }
    if (doi && !/[\s<>"']/u.test(doi)) {
        const encodedDoi = doi.split("/").map((part) => encodeURIComponent(part)).join("/");
        destinations.push({ kind: "DOI", identifier: doi, url: `https://doi.org/${encodedDoi}` });
    }
    return destinations;
}

function paperDisplayLabel(item, destinations) {
    const primary = destinations[0];
    if (primary?.kind === "PMC") return `PMCID ${primary.identifier}`;
    if (primary?.kind === "PubMed") return `PMID ${primary.identifier}`;
    if (primary?.kind === "DOI") return `DOI ${primary.identifier}`;
    return item?.canonical_id || item?.doc_key || "Source passage";
}

function paperLink(destination, label, className) {
    if (!destination?.url) return `<strong>${escapeHtml(label)}</strong>`;
    return `<a class="${escapeHtml(className)}" href="${escapeHtml(destination.url)}" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer" title="Open this paper in ${escapeHtml(destination.kind)}">${escapeHtml(label)}<span aria-hidden="true"> ↗</span></a>`;
}

function evidenceCard(item) {
    const destinations = paperDestinations(item);
    const paper = paperDisplayLabel(item, destinations);
    const metadata = [item.journal, item.pub_year, item.section_type].filter(Boolean).join(" · ");
    const relation = item.predicate
        ? relationEvidenceText(
            item.subject_label || item.subject_id,
            item.predicate,
            item.object_label || item.object_id,
        )
        : "";
    const contexts = Array.isArray(item.cell_contexts) ? item.cell_contexts : [];
    const destinationLinks = destinations.map((destination) =>
        paperLink(destination, `${destination.kind}: ${destination.identifier}`, "evidence-source-link"),
    ).join("");
    return `<article class="evidence-card" data-evidence-search="${escapeHtml(evidenceSearchText(item))}">
        <div class="evidence-meta">
            <div class="evidence-paper-row">
                ${paperLink(destinations[0], paper, "evidence-paper-title")}
                ${destinationLinks ? `<span class="evidence-source-links">${destinationLinks}</span>` : ""}
            </div>
            <span class="evidence-publication-meta">${escapeHtml(metadata || item.base || "")}</span>
        </div>
        ${relation ? `<p class="evidence-relation">${escapeHtml(relation)}</p>` : ""}
        <p class="evidence-text">${highlightEvidenceText(item.chunk_text, item.entities)}</p>
        ${contexts.length ? `<div class="context-row">${contexts.map((context) => `<span>${escapeHtml(context.label)}</span>`).join("")}</div>` : ""}
    </article>`;
}

function renderEvidence() {
    const filter = String($("#evidenceFilter")?.value || "").trim().toLowerCase();
    const items = state.evidence.filter((item) => !filter || evidenceSearchText(item).includes(filter));
    const list = $("#evidenceList");
    if (!items.length) {
        list.innerHTML = `<p class="empty-copy">${state.evidence.length ? "No loaded evidence matches this filter." : "No supporting evidence was returned."}</p>`;
        return;
    }
    list.innerHTML = items.map(evidenceCard).join("");
}

async function openSelectionEvidence() {
    if (!state.selection || !["node", "edge"].includes(state.selection.kind)) return;
    activateTab("evidence");
    $("#evidenceHeading").textContent = state.selection.label || "Supporting evidence";
    $("#evidenceSubheading").textContent = "Loading source passages…";
    $("#evidenceList").innerHTML = '<p class="empty-copy">Loading evidence…</p>';
    try {
        const path = state.selection.kind === "edge"
            ? `evidence/edges/${encodeURIComponent(state.selection.id)}`
            : `evidence/nodes/${encodeURIComponent(state.selection.id)}`;
        const payload = await requestJson(`/api/networks/${encodeURIComponent(jobId)}/${path}`);
        state.evidence = Array.isArray(payload?.evidence) ? payload.evidence : [];
        $("#evidenceSubheading").textContent = `${formatNumber(state.evidence.length)} supporting passage${state.evidence.length === 1 ? "" : "s"} loaded.`;
        renderEvidence();
    } catch (error) {
        state.evidence = [];
        $("#evidenceSubheading").textContent = error.message;
        renderEvidence();
    }
}

function bindNetworkEvents() {
    if (!state.network) return;
    state.network.on("click", (params) => {
        if (params.nodes?.length) void selectNode(params.nodes[0]);
        else if (params.edges?.length) void selectEdge(params.edges[0]);
    });
    state.network.on("doubleClick", (params) => {
        const nodeId = params.nodes?.[0];
        if (!nodeId || !state.nodes?.get(nodeId)) return;
        const connectedEdges = state.network.getConnectedEdges(nodeId);
        state.edges.remove(connectedEdges);
        state.nodes.remove(nodeId);
        setVisibleCounts();
        showToast("Node removed from this browser view. Use Select node to add it again.");
    });
    state.network.on("stabilizationIterationsDone", () => {
        state.network.setOptions({ physics: { stabilization: false } });
    });
}

function buildNetwork(payload, { replace = true } = {}) {
    if (!window.vis?.DataSet || !window.vis?.Network) {
        throw new Error("The PyVis/vis-network browser library could not be loaded.");
    }
    const incomingNodes = removeNetworkTooltips(Array.isArray(payload?.nodes) ? payload.nodes : []);
    // Preserve the server-provided predicate labels and edge fonts for initial,
    // filtered, and incrementally inserted relation payloads.
    const incomingEdges = removeNetworkTooltips(Array.isArray(payload?.edges) ? payload.edges : []);
    if (!state.network || replace) {
        state.nodes = new window.vis.DataSet(incomingNodes);
        state.edges = new window.vis.DataSet(incomingEdges);
        state.network?.destroy();
        state.network = new window.vis.Network(
            $("#networkCanvas"),
            { nodes: state.nodes, edges: state.edges },
            payload?.options || {},
        );
        state.physicsEnabled = true;
        bindNetworkEvents();
    } else {
        // Incremental insertion is intentional: no clear(), replacement, or reset.
        // DataSet.update preserves all currently displayed nodes and their positions.
        state.nodes.update(incomingNodes);
        state.edges.update(incomingEdges);
        state.network.setOptions({ physics: { enabled: state.physicsEnabled } });
    }
    setVisibleCounts();
    state.graphLoaded = true;
    state.hierarchySignature = "";
    $("#buildOverlay").hidden = true;
    $("#networkStatusText").textContent = `${formatNumber(state.nodes.length)} nodes and ${formatNumber(state.edges.length)} edges are visible.`;
}

function relationDisplayName(predicate) {
    return String(predicate || "relation")
        .replaceAll("_", " ")
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function relationTypeCheckboxes() {
    return $$('#relationTypeOptions input[name="relationType"]');
}

function selectedRelationPredicates() {
    return relationTypeCheckboxes()
        .filter((input) => input.checked && !input.disabled)
        .map((input) => String(input.value || "").trim())
        .filter(Boolean);
}

function updateRelationFilterControls() {
    const selectedCount = selectedRelationPredicates().length;
    const availableCount = relationTypeCheckboxes().filter((input) => !input.disabled).length;
    const applyButton = $("#applyRelationTypes");
    if (applyButton) {
        applyButton.disabled = selectedCount === 0;
        applyButton.textContent = selectedCount
            ? `Show ${formatNumber(selectedCount)} selected relation type${selectedCount === 1 ? "" : "s"}`
            : "Show selected relations";
    }
    const clearButton = $("#clearRelationTypes");
    if (clearButton) clearButton.disabled = selectedCount === 0;
    const selectAllButton = $("#selectAllRelationTypes");
    if (selectAllButton) selectAllButton.disabled = availableCount === 0 || selectedCount === availableCount;
}

function setRelationTypeSelection(predicates) {
    const selected = new Set((Array.isArray(predicates) ? predicates : []).map((value) => String(value)));
    relationTypeCheckboxes().forEach((input) => {
        input.checked = !input.disabled && selected.has(String(input.value));
    });
    updateRelationFilterControls();
}

function renderRelationTypeOptions(relationTypes, { preserveSelection = true } = {}) {
    const previous = preserveSelection ? new Set(selectedRelationPredicates()) : new Set();
    const types = Array.isArray(relationTypes) ? relationTypes : [];
    state.relationTypes = types;
    const container = $("#relationTypeOptions");
    if (!container) return;
    if (!types.length) {
        container.innerHTML = '<p class="empty-copy">No extracted relation types are available.</p>';
        updateRelationFilterControls();
        return;
    }
    container.innerHTML = types.map((item, index) => {
        const predicate = String(item?.predicate || "");
        const supportedEdges = Number(item?.edge_count || 0);
        const totalEdges = Number(item?.total_edge_count || 0);
        const nodeCount = Number(item?.node_count || 0);
        const disabled = supportedEdges <= 0;
        const checked = !disabled && previous.has(predicate);
        const countText = totalEdges === supportedEdges
            ? `${formatNumber(supportedEdges)} edge${supportedEdges === 1 ? "" : "s"} · ${formatNumber(nodeCount)} node${nodeCount === 1 ? "" : "s"}`
            : `${formatNumber(supportedEdges)} of ${formatNumber(totalEdges)} edges · ${formatNumber(nodeCount)} nodes`;
        return `<label class="relation-type-option${disabled ? " unavailable" : ""}" for="relationType${index}">
            <input id="relationType${index}" name="relationType" type="checkbox" value="${escapeHtml(predicate)}" ${checked ? "checked" : ""} ${disabled ? "disabled" : ""}>
            <span><strong>${escapeHtml(relationDisplayName(predicate))}</strong><small>${escapeHtml(countText)}</small></span>
        </label>`;
    }).join("");
    relationTypeCheckboxes().forEach((input) => input.addEventListener("change", updateRelationFilterControls));
    updateRelationFilterControls();
}

async function loadRelationTypes({ preserveSelection = true } = {}) {
    const supportMin = currentRelationSupportMin();
    const payload = await requestJson(
        `/api/networks/${encodeURIComponent(jobId)}/relation-types?relation_support_min=${encodeURIComponent(supportMin)}`,
    );
    renderRelationTypeOptions(payload?.relation_types, { preserveSelection });
    const help = $("#relationTypeHelp");
    if (help) {
        help.textContent = `Counts use relation support ≥ ${formatNumber(supportMin)} paper(s). This view loads every matching edge and endpoint across the complete network; Top nodes does not limit it.`;
    }
}

async function loadRelationTypeGraph() {
    const predicates = selectedRelationPredicates();
    if (!predicates.length) {
        showToast("Select at least one relation type.");
        return;
    }
    const supportMin = currentRelationSupportMin();
    const button = $("#applyRelationTypes");
    if (button) button.disabled = true;
    $("#networkStatusText").textContent = `Loading every ${predicates.map(relationDisplayName).join(", ")} relation with support ≥ ${formatNumber(supportMin)} paper(s)…`;
    try {
        const payload = await requestJson(
            `/api/networks/${encodeURIComponent(jobId)}/graph/relations?predicates=${encodeURIComponent(predicates.join(","))}&relation_support_min=${encodeURIComponent(supportMin)}`,
            { timeoutMs: 120000 },
        );
        buildNetwork(payload, { replace: true });
        state.viewMode = "relation_types";
        const selected = Array.isArray(payload?.selected_predicates) ? payload.selected_predicates : predicates;
        $("#networkStatusText").textContent = `${formatNumber(state.nodes.length)} endpoint nodes and ${formatNumber(state.edges.length)} ${selected.map(relationDisplayName).join(", ")} relations with support ≥ ${formatNumber(supportMin)} paper(s) are visible.`;
        window.setTimeout(() => state.network?.fit({ animation: { duration: 450, easingFunction: "easeInOutQuad" } }), 150);
    } catch (error) {
        showToast(error.message);
    } finally {
        updateRelationFilterControls();
    }
}

async function loadInitialGraph() {
    const count = currentTopNodeCount();
    const supportMin = currentRelationSupportMin();
    $("#networkStatusText").textContent = `Loading the top ${formatNumber(count)} nodes ranked by supporting papers; relation support ≥ ${formatNumber(supportMin)} paper(s)…`;
    const payload = await requestJson(
        `/api/networks/${encodeURIComponent(jobId)}/graph?top_nodes=${encodeURIComponent(count)}&relation_support_min=${encodeURIComponent(supportMin)}`,
        { timeoutMs: 60000 },
    );
    buildNetwork(payload, { replace: true });
    state.viewMode = "initial";
    setRelationTypeSelection([]);
    $("#networkStatusText").textContent = `${formatNumber(state.nodes.length)} nodes ranked by supporting papers and ${formatNumber(state.edges.length)} relations with support ≥ ${formatNumber(supportMin)} paper(s) are visible.`;
    window.setTimeout(() => state.network?.fit({ animation: { duration: 450, easingFunction: "easeInOutQuad" } }), 150);
}

async function loadFullGraph() {
    const supportMin = currentRelationSupportMin();
    const button = $("#resetFullGraph");
    if (button) button.disabled = true;
    $("#networkStatusText").textContent = `Loading the full graph with relation support ≥ ${formatNumber(supportMin)} paper(s)…`;
    try {
        const payload = await requestJson(
            `/api/networks/${encodeURIComponent(jobId)}/graph/full?relation_support_min=${encodeURIComponent(supportMin)}`,
            { timeoutMs: 120000 },
        );
        buildNetwork(payload, { replace: true });
        state.viewMode = "full";
        setRelationTypeSelection([]);
        $("#networkStatusText").textContent = `${formatNumber(state.nodes.length)} nodes and ${formatNumber(state.edges.length)} relations are visible.`;
        window.setTimeout(() => state.network?.fit({ animation: { duration: 450, easingFunction: "easeInOutQuad" } }), 150);
    } catch (error) {
        showToast(error.message);
    } finally {
        if (button) button.disabled = false;
    }
}

function renderNodeSearchOptions(nodes) {
    const options = $("#nodeSearchOptions");
    if (!options) return;
    options.innerHTML = nodes.map((node, index) => `
        <button class="node-search-option" type="button" data-node-index="${index}">
            <strong>${escapeHtml(node.label || node.id)}</strong>
            <small>${escapeHtml(formatType(node.entity_type))} · ${escapeHtml(node.normalized_id || node.id)}</small>
        </button>`).join("");
    options.hidden = nodes.length === 0;
}

function selectSearchNode(node) {
    state.selectedSearchNode = node;
    $("#selectedSearchType").textContent = formatType(node.entity_type);
    $("#selectedSearchType").className = `type-pill ${escapeHtml(node.entity_type)}`;
    $("#selectedSearchLabel").textContent = node.label || node.id;
    $("#selectedSearchId").textContent = node.normalized_id || node.id;
    $("#selectedSearchNode").hidden = false;
    $("#addSelectedNode").disabled = false;
    $("#nodeSearchOptions").hidden = true;
}

async function searchNodes() {
    const query = String($("#nodeSearch").value || "").trim();
    state.searchGeneration += 1;
    const generation = state.searchGeneration;
    if (!query) {
        state.nodeSearchResults = [];
        renderNodeSearchOptions([]);
        $("#searchSpinner").hidden = true;
        return;
    }
    $("#searchSpinner").hidden = false;
    try {
        const payload = await requestJson(`/api/networks/${encodeURIComponent(jobId)}/nodes/search?q=${encodeURIComponent(query)}`);
        if (generation !== state.searchGeneration) return;
        state.nodeSearchResults = Array.isArray(payload?.nodes) ? payload.nodes : [];
        renderNodeSearchOptions(state.nodeSearchResults);
    } catch (error) {
        if (generation === state.searchGeneration) {
            state.nodeSearchResults = [];
            renderNodeSearchOptions([]);
            showToast(error.message);
        }
    } finally {
        if (generation === state.searchGeneration) $("#searchSpinner").hidden = true;
    }
}

async function addSelectedNode() {
    const node = state.selectedSearchNode;
    if (!node || !state.network) return;
    const button = $("#addSelectedNode");
    button.disabled = true;
    button.textContent = "Adding…";
    try {
        const payload = await requestJson(
            `/api/networks/${encodeURIComponent(jobId)}/nodes/${encodeURIComponent(node.id)}/neighborhood?limit=${encodeURIComponent(expansionLimit)}&relation_support_min=${encodeURIComponent(currentRelationSupportMin())}`,
            { timeoutMs: 60000 },
        );
        const beforeNodes = state.nodes.length;
        const beforeEdges = state.edges.length;
        buildNetwork(payload, { replace: false });
        state.network.selectNodes([node.id]);
        state.network.focus(node.id, { scale: 1.15, animation: { duration: 500, easingFunction: "easeInOutQuad" } });
        void selectNode(node.id, { openDetails: false });
        showToast(`Added ${formatNumber(state.nodes.length - beforeNodes)} node(s) and ${formatNumber(state.edges.length - beforeEdges)} edge(s) without replacing the current graph.`);
    } catch (error) {
        showToast(error.message);
    } finally {
        button.disabled = false;
        button.textContent = "Add node + neighbours";
    }
}


function bindHierarchyInteractions() {
    const container = $("#hierarchyContent");
    if (!container) return;
    $$("[data-term-id]", container).forEach((row) => {
        row.addEventListener("click", (event) => {
            event.stopPropagation();
            if (row.dataset.subclassToggle === "1" && row.classList.contains("highlighted")) {
                const panel = row.nextElementSibling;
                if (panel?.classList.contains("subclass-panel")) {
                    panel.hidden = !panel.hidden;
                    row.classList.toggle("expanded", !panel.hidden);
                }
                return;
            }
            void addHierarchyTerm({
                conceptId: row.dataset.termId,
                label: row.dataset.termLabel,
                graphNodeId: row.dataset.nodeId,
                row,
            });
        });
    });
}

async function addHierarchyTerm({ conceptId, label, graphNodeId, row }) {
    const clId = cellOntologyId(conceptId);
    if (!clId || !state.network) return;
    const existingId = String(graphNodeId || "");
    if (existingId && state.nodes?.get(existingId)) {
        state.network.selectNodes([existingId]);
        state.network.focus(existingId, {
            scale: 1.12,
            animation: { duration: 450, easingFunction: "easeInOutQuad" },
        });
        await selectNode(existingId, { openDetails: false });
        row?.classList.add("highlighted");
        showToast(`${label || clId} is already visible.`);
        return;
    }

    row?.classList.add("loading");
    try {
        const payload = await requestJson(
            `/api/networks/${encodeURIComponent(jobId)}/cell-hierarchy/term-neighborhood?concept_id=${encodeURIComponent(clId)}&limit=${encodeURIComponent(expansionLimit)}&relation_support_min=${encodeURIComponent(currentRelationSupportMin())}`,
            { timeoutMs: 60000 },
        );
        const beforeNodes = state.nodes.length;
        const beforeEdges = state.edges.length;
        buildNetwork(payload, { replace: false });
        const rootNodeId = String(payload?.root_node_id || "");
        if (rootNodeId && state.nodes.get(rootNodeId)) {
            if (row) row.dataset.nodeId = rootNodeId;
            state.network.selectNodes([rootNodeId]);
            state.network.focus(rootNodeId, {
                scale: 1.12,
                animation: { duration: 450, easingFunction: "easeInOutQuad" },
            });
            await selectNode(rootNodeId, { openDetails: false });
        }
        row?.classList.add("highlighted");
        const addedNodes = Math.max(0, state.nodes.length - beforeNodes);
        const addedEdges = Math.max(0, state.edges.length - beforeEdges);
        const ontologyOnly = Boolean(payload?.ontology_only);
        showToast(
            ontologyOnly
                ? `${label || clId} was added as a hierarchy-only term; no extracted relation met the current support threshold.`
                : `Added ${formatNumber(addedNodes)} node(s) and ${formatNumber(addedEdges)} supported interaction edge(s) without replacing the current graph.`,
        );
    } catch (error) {
        showToast(error.message);
    } finally {
        row?.classList.remove("loading");
    }
}

function visibleCellTerms() {
    if (!state.nodes) return [];
    const byConceptId = new Map();
    state.nodes.get().forEach((node) => {
        if (String(node?.entity_type || "").toLowerCase() !== "cell") return;
        const conceptId = cellOntologyId(node.cl_id || node.normalized_id);
        if (!conceptId || byConceptId.has(conceptId)) return;
        byConceptId.set(conceptId, {
            conceptId,
            nodeId: String(node.id || ""),
            label: String(node.label || conceptId),
        });
    });
    return Array.from(byConceptId.values()).sort((left, right) => (
        left.label.localeCompare(right.label) || left.conceptId.localeCompare(right.conceptId)
    ));
}

async function showCellHierarchy({ activate = true } = {}) {
    const visibleCells = visibleCellTerms();
    if (activate) activateTab("hierarchy");
    if (!visibleCells.length) {
        $("#hierarchyHeading").textContent = "Visible cell hierarchy";
        $("#hierarchySubheading").textContent = "No visible network node has a Cell Ontology identifier.";
        $("#hierarchyContent").innerHTML = '<p class="hierarchy-empty">Add or display at least one Cell Ontology-normalized cell node first.</p>';
        showToast("No visible cell node has a Cell Ontology identifier.");
        return;
    }

    const button = $("#showCellHierarchy");
    if (button) button.disabled = true;
    const previousText = button?.textContent || "Show Cell Ontology hierarchy";
    if (button) button.textContent = "Loading hierarchy…";
    $("#hierarchyHeading").textContent = "Visible cell hierarchy";
    $("#hierarchySubheading").textContent = `Loading root paths for ${formatNumber(visibleCells.length)} visible cell term(s)…`;
    $("#hierarchyContent").innerHTML = '<p class="empty-copy">Loading hierarchy…</p>';
    try {
        const conceptIds = visibleCells.map((term) => term.conceptId);
        const signature = conceptIds.join("|");
        const payload = await requestJson(`/api/networks/${encodeURIComponent(jobId)}/cell-hierarchy`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_ids: conceptIds, max_paths: hierarchyMaxPaths }),
            timeoutMs: 60000,
        });
        state.hierarchySignature = signature;
        const release = payload?.ontology?.release ? ` · release ${payload.ontology.release}` : "";
        $("#hierarchySubheading").textContent = `${formatNumber(payload?.visible_cell_count)} visible cell term(s) across ${formatNumber(payload?.path_count)} root-to-term path(s)${release}. Highlighted rows are cells in the current graph.`;
        $("#hierarchyContent").innerHTML = payload?.html || '<p class="hierarchy-empty">No Cell Ontology path was returned.</p>';
        bindHierarchyInteractions();
    } catch (error) {
        $("#hierarchySubheading").textContent = error.message;
        $("#hierarchyContent").innerHTML = `<p class="hierarchy-empty">${escapeHtml(error.message)}</p>`;
    } finally {
        if (button) {
            button.textContent = previousText;
            button.disabled = false;
        }
    }
}

async function pollJobUntilReady() {
    if (!jobId) throw new Error("The network job ID is missing.");
    while (true) {
        const job = await requestJson(`/api/networks/${encodeURIComponent(jobId)}`);
        state.job = job;
        setBuildProgress(job);
        if (job.status === "completed") return job;
        if (job.status === "failed") throw new Error(job.error || job.message || "Network generation failed.");
        await sleep(JOB_POLL_MS);
    }
}

function bindControls() {
    $("#returnHome")?.addEventListener("click", (event) => {
        if (window.history.length <= 1) return;
        event.preventDefault();
        window.history.back();
    });
    $$(".tab-button").forEach((button) => button.addEventListener("click", () => {
        const tab = button.dataset.tab;
        activateTab(tab);
        if (tab === "hierarchy") void showCellHierarchy({ activate: false });
    }));
    $("#fitNetwork")?.addEventListener("click", () => state.network?.fit({ animation: { duration: 450, easingFunction: "easeInOutQuad" } }));
    $("#resetNetwork")?.addEventListener("click", async () => {
        try {
            await loadRelationTypes({ preserveSelection: false });
            await loadInitialGraph();
            showToast("The paper-ranked node and relation-support filters were applied.");
        } catch (error) {
            showToast(error.message);
        }
    });
    $("#resetFullGraph")?.addEventListener("click", async () => {
        try {
            await loadRelationTypes({ preserveSelection: false });
            await loadFullGraph();
            showToast("The full graph was restored.");
        } catch (error) {
            showToast(error.message);
        }
    });
    $("#applyRelationTypes")?.addEventListener("click", () => void loadRelationTypeGraph());
    $("#selectAllRelationTypes")?.addEventListener("click", () => {
        relationTypeCheckboxes().forEach((input) => { if (!input.disabled) input.checked = true; });
        updateRelationFilterControls();
    });
    $("#clearRelationTypes")?.addEventListener("click", () => setRelationTypeSelection([]));
    $("#relationSupportMin")?.addEventListener("change", () => {
        void loadRelationTypes({ preserveSelection: true }).catch((error) => showToast(error.message));
    });
    ["#topNodeCount", "#relationSupportMin"].forEach((selector) => {
        $(selector)?.addEventListener("keydown", (event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            $("#resetNetwork")?.click();
        });
    });
    let searchTimer = null;
    $("#nodeSearch")?.addEventListener("input", () => {
        window.clearTimeout(searchTimer);
        searchTimer = window.setTimeout(() => void searchNodes(), 250);
    });
    $("#nodeSearch")?.addEventListener("change", () => {
        const value = String($("#nodeSearch")?.value || "").trim().toLocaleLowerCase();
        const node = state.nodeSearchResults.find((item) =>
            [item.label, item.normalized_id, item.id]
                .some((candidate) => String(candidate || "").toLocaleLowerCase() === value)
        );
        if (node) selectSearchNode(node);
    });
    $("#nodeSearchOptions")?.addEventListener("click", (event) => {
        const option = event.target.closest("[data-node-index]");
        if (!option) return;
        const node = state.nodeSearchResults[Number(option.dataset.nodeIndex)];
        if (!node) return;
        $("#nodeSearch").value = node.label || node.id;
        selectSearchNode(node);
    });
    document.addEventListener("click", (event) => {
        if (event.target.closest(".search-control")) return;
        const options = $("#nodeSearchOptions");
        if (options) options.hidden = true;
    });
    $("#addSelectedNode")?.addEventListener("click", () => void addSelectedNode());
    $("#showCellHierarchy")?.addEventListener("click", () => void showCellHierarchy());
    $("#showSelectionEvidence")?.addEventListener("click", () => void openSelectionEvidence());
    $("#evidenceFilter")?.addEventListener("input", renderEvidence);
}

async function initialize() {
    bindControls();
    integerInputValue("#relationSupportMin", defaultRelationSupportMin, 0);
    try {
        await pollJobUntilReady();
        await loadRelationTypes({ preserveSelection: false });
        await loadInitialGraph();
    } catch (error) {
        $("#buildStage").textContent = "Stage 4 failed";
        $("#buildMessage").textContent = error.message;
        $("#networkStatusText").textContent = error.message;
        $("#buildProgressBar").style.width = "100%";
        $("#buildPercent").textContent = "Failed";
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void initialize(), { once: true });
} else {
    void initialize();
}
