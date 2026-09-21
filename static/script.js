/* =============================================================================
   GridWise LLM — Dashboard Logic
   Handles API calls, validation, rendering, and Chart.js visualizations.

   Defensive design:
     * Every DOM lookup is null-checked via the `el()` helper.
     * Charts only render when their canvas is visible (display !== none).
     * All render paths are wrapped in try/catch so a single bug never breaks
       the entire UI.
   ============================================================================= */

(function () {
    "use strict";

    // =========================================================================
    // State
    // =========================================================================
    const state = {
        presets: {},
        lastResponse: null,    // OptimizeResponse
        lastAnalyze: null,     // AnalyzeResponse (or null)
        charts: {},
        tariffs: [],           // Per-hour tariff (BDT/kWh) for chart math
    };

    // =========================================================================
    // Safe DOM helpers  (THE FIX FOR THE NULL textContent BUG)
    // =========================================================================
    /**
     * Safe `getElementById` — returns the element or `null`.
     * Use `el(id)` to check for existence before mutating.
     */
    function el(id) {
        const node = document.getElementById(id);
        return node; // null if not in DOM
    }

    /**
     * Set textContent only when the element exists. Logs a single warning
     * so silent bugs become visible during development.
     */
    function setText(id, value) {
        const node = el(id);
        if (!node) {
            // Throttled warn to avoid console flooding during render.
            if (!setText._warned) setText._warned = new Set();
            if (!setText._warned.has(id)) {
                console.warn(`[GridWise] Missing element #${id}`);
                setText._warned.add(id);
            }
            return;
        }
        node.textContent = value;
    }

    /**
     * Set innerHTML only when the element exists.
     */
    function setHTML(id, value) {
        const node = el(id);
        if (!node) return;
        node.innerHTML = value;
    }

    /**
     * Toggle `hidden` attribute only when the element exists.
     */
    function setHidden(id, hidden) {
        const node = el(id);
        if (!node) return;
        node.hidden = !!hidden;
    }

    /**
     * Return true if the element is currently visible (display !== none and
     * the element itself + all ancestors are not hidden).  Charts inside a
     * hidden pane cannot be measured by Chart.js and would render at 0 size.
     */
    function isVisible(node) {
        if (!node) return false;
        if (node.hidden) return false;
        const style = window.getComputedStyle(node);
        if (style.display === "none" || style.visibility === "hidden") return false;
        return true;
    }

    const fmtNumber = (n, digits = 2) => {
        const v = Number(n);
        if (!Number.isFinite(v)) return "—";
        return v.toLocaleString(undefined, {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
        });
    };

    const escape = (str) =>
        String(str ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");

    // =========================================================================
    // Tab navigation
    // =========================================================================
    function activateTab(name) {
        document.querySelectorAll(".tab").forEach((t) => {
            t.classList.toggle("active", t.dataset.tab === name);
        });
        document.querySelectorAll(".pane").forEach((p) => {
            p.classList.toggle("active", p.id === `pane-${name}`);
        });
    }

    document.querySelectorAll(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;
            activateTab(target);
            // Re-render so charts (which need a visible canvas) draw correctly.
            if (target === "results" && state.lastResponse) {
                renderResults(state.lastResponse);
            } else if (target === "analytics" && state.lastAnalyze) {
                renderAnalytics(state.lastAnalyze);
            } else if (target === "raw") {
                renderRawJson();
            }
        });
    });

    // =========================================================================
    // Toast notifications
    // =========================================================================
    function toast(message, type = "info") {
        const container = el("toast-container");
        if (!container) return;
        const node = document.createElement("div");
        node.className = `toast ${type}`;
        node.textContent = message;
        container.appendChild(node);
        setTimeout(() => {
            node.style.opacity = "0";
            node.style.transform = "translateX(20px)";
            setTimeout(() => node.remove(), 300);
        }, 3500);
    }

    // =========================================================================
    // Sample data — used when /api/presets is unavailable
    // =========================================================================
    function getSampleScenario() {
        return {
            scenario_id: "SAMPLE-01",
            operator_notes: [
                "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
                "The sports office moved next month's registration deadline.",
            ],
            hours: buildDefaultHours(),
            battery: {
                capacity_kwh: 220,
                initial_energy_kwh: 110,
                minimum_energy_kwh: 40,
                max_charge_kwh_per_hour: 50,
                max_discharge_kwh_per_hour: 50,
            },
        };
    }

    function buildDefaultHours() {
        const profile = [
            [6, 0, 6], [10, 5, 8], [15, 20, 10], [20, 50, 12], [25, 90, 14],
            [30, 130, 16], [35, 160, 16], [40, 180, 15], [42, 170, 14], [38, 140, 13],
            [40, 90, 14], [42, 45, 18], [45, 10, 22], [55, 0, 28], [60, 0, 30],
            [55, 0, 26], [45, 0, 18], [35, 0, 10], [25, 0, 7], [20, 0, 6],
            [15, 0, 5], [12, 0, 5], [10, 0, 5], [8, 0, 6],
        ];
        return profile.map(([demand, solar, tariff], hour) => ({
            hour,
            demand_kwh: demand * 5,
            solar_kwh: solar,
            tariff_bdt_per_kwh: tariff,
        }));
    }

    // =========================================================================
    // Preset loading
    // =========================================================================
    async function loadPresets() {
        const container = el("preset-list");
        try {
            const res = await fetch("/api/presets");
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            state.presets = await res.json();
            renderPresets();
        } catch (err) {
            console.warn("Failed to load presets:", err);
            if (container) container.innerHTML = '<p class="muted">No presets available.</p>';
        }
    }

    function renderPresets() {
        const container = el("preset-list");
        if (!container) return;
        const entries = Object.entries(state.presets);
        if (entries.length === 0) {
            container.innerHTML = '<p class="muted">No presets available.</p>';
            return;
        }
        container.innerHTML = "";
        entries.forEach(([id, data]) => {
            const node = document.createElement("div");
            node.className = "preset-item";
            node.innerHTML = `
                <div class="preset-label">${escape(data.label)}</div>
                <div class="preset-id">${escape(id)} · click to load</div>
            `;
            node.addEventListener("click", () => loadScenario(data.input));
            container.appendChild(node);
        });
    }

    function loadScenario(scenario) {
        if (!scenario) return;
        const sid = el("scenario_id");
        const notes = el("operator_notes");
        const hoursEl = el("hours_json");
        if (sid) sid.value = scenario.scenario_id || "";
        if (notes) notes.value = (scenario.operator_notes || []).join("\n");
        if (hoursEl) hoursEl.value = JSON.stringify(scenario.hours || [], null, 2);
        if (scenario.battery) {
            const map = {
                b_capacity: scenario.battery.capacity_kwh,
                b_initial: scenario.battery.initial_energy_kwh,
                b_min: scenario.battery.minimum_energy_kwh,
                b_charge: scenario.battery.max_charge_kwh_per_hour,
                b_discharge: scenario.battery.max_discharge_kwh_per_hour,
            };
            for (const [id, val] of Object.entries(map)) {
                const node = el(id);
                if (node) node.value = val;
            }
        }
        // Cache tariffs for chart math.
        state.tariffs = (scenario.hours || []).map((h) => h.tariff_bdt_per_kwh);
        window.__gridwiseTariffs = state.tariffs;
        toast(`Loaded scenario "${scenario.scenario_id}".`, "success");
    }

    function safeAddListener(id, handler) {
        const node = el(id);
        if (node) node.addEventListener("click", handler);
    }

    safeAddListener("load-preset", () => {
        const entries = Object.entries(state.presets);
        if (entries.length === 0) return loadScenario(getSampleScenario());
        loadScenario(entries[0][1].input);
    });

    safeAddListener("load-sample-notes", () => {
        const sample = getSampleScenario();
        loadScenario(sample);
    });

    // =========================================================================
    // Build request payload from form
    // =========================================================================
    function buildRequest() {
        const sidEl = el("scenario_id");
        const notesEl = el("operator_notes");
        const hoursEl = el("hours_json");

        const scenario_id = (sidEl ? sidEl.value : "").trim() || "SAMPLE-01";
        const notesRaw = (notesEl ? notesEl.value : "").trim();
        if (!notesRaw) throw new Error("At least one operator note is required.");
        const operator_notes = notesRaw.split("\n").map((s) => s.trim()).filter(Boolean);
        if (operator_notes.length === 0 || operator_notes.length > 3) {
            throw new Error("Operator notes must contain 1–3 non-empty lines.");
        }

        let hours;
        try {
            hours = JSON.parse(hoursEl ? hoursEl.value : "[]");
        } catch (err) {
            throw new Error(`Hours JSON is invalid: ${err.message}`);
        }
        if (!Array.isArray(hours) || hours.length !== 24) {
            throw new Error("Hours array must contain exactly 24 entries.");
        }

        const batteryFields = ["b_capacity", "b_initial", "b_min", "b_charge", "b_discharge"];
        const batteryKeys = [
            "capacity_kwh",
            "initial_energy_kwh",
            "minimum_energy_kwh",
            "max_charge_kwh_per_hour",
            "max_discharge_kwh_per_hour",
        ];
        const battery = {};
        for (let i = 0; i < batteryFields.length; i++) {
            const id = batteryFields[i];
            const key = batteryKeys[i];
            const v = parseFloat(el(id)?.value);
            if (!Number.isFinite(v)) throw new Error(`Battery field "${key}" is invalid.`);
            battery[key] = v;
        }

        // Cache tariffs for chart math.
        state.tariffs = hours.map((h) => Number(h.tariff_bdt_per_kwh) || 0);
        window.__gridwiseTariffs = state.tariffs;

        return { scenario_id, operator_notes, hours, battery };
    }

    // =========================================================================
    // Run optimization
    // =========================================================================
    function bindRunButton() {
        const btn = el("run-optimize");
        if (!btn) return;
        btn.addEventListener("click", async () => {
            const errEl = el("form-error");
            if (errEl) errEl.textContent = "";

            let payload;
            try {
                payload = buildRequest();
            } catch (err) {
                if (errEl) errEl.textContent = err.message;
                toast(err.message, "error");
                return;
            }

            btn.classList.add("is-loading");
            btn.disabled = true;

            try {
                const res = await fetch("/optimize-energy", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (!res.ok) {
                    const detail = await safeError(res);
                    throw new Error(detail);
                }
                const data = await res.json();
                state.lastResponse = data;

                // Fire-and-forget analytics call (non-blocking if it fails).
                fetch("/api/analyze", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                })
                    .then((ar) => (ar.ok ? ar.json() : null))
                    .then((j) => {
                        if (j) state.lastAnalyze = j;
                    })
                    .catch((e) => console.warn("Analytics call failed:", e));

                renderResults(data);
                activateTab("results");
                toast("Optimization complete.", "success");
            } catch (err) {
                console.error("Optimization failed:", err);
                if (errEl) errEl.textContent = err.message;
                toast(err.message, "error");
            } finally {
                btn.classList.remove("is-loading");
                btn.disabled = false;
            }
        });
    }

    async function safeError(res) {
        try {
            const j = await res.json();
            return j.detail || JSON.stringify(j);
        } catch {
            return `HTTP ${res.status}`;
        }
    }

    // =========================================================================
    // Render results  (THE FIX: every mutation is null-safe + try/catch)
    // =========================================================================
    function renderResults(data) {
        if (!data) return;
        try {
            // KPIs — null-safe via setText().
            setText("kpi-cost", fmtNumber(data.total_cost_bdt, 2));
            setText("kpi-grid", fmtNumber(data.total_grid_kwh, 2));
            setText("kpi-peak", fmtNumber(data.peak_grid_kwh, 2));
            const totalSolar = (data.hourly_plan || []).reduce(
                (s, p) => s + (Number(p.solar_used_kwh) || 0), 0
            );
            setText("kpi-solar", fmtNumber(totalSolar, 2));

            // Charts — each wrapped individually so one failure doesn't kill the rest.
            safeRender(() => renderMixChart(data), "mix chart");
            safeRender(() => renderCostGridChart(data), "cost/grid chart");
            safeRender(() => renderSocChart(data), "SoC chart");

            // Directives
            safeRender(() => renderDirectives(data.directive_interpretation || []), "directives");

            // Hourly table
            safeRender(() => renderHourlyTable(data), "hourly table");

            // Warnings
            const warnings = Array.isArray(data.warnings) ? data.warnings : [];
            if (warnings.length > 0) {
                setHidden("warnings-card", false);
                setHTML("warning-list", warnings.map((w) => `<li>⚠ ${escape(w)}</li>`).join(""));
            } else {
                setHidden("warnings-card", true);
            }

            // Analytics tab content (deferred until tab opened)
            if (state.lastAnalyze) safeRender(() => renderAnalytics(state.lastAnalyze), "analytics");

            // Raw JSON tab content
            safeRender(renderRawJson, "raw JSON");
        } catch (err) {
            // Catch-all so the UI never freezes after a render bug.
            console.error("[GridWise] renderResults crashed:", err);
            toast("Failed to render results. See console.", "error");
        }
    }

    function safeRender(fn, label) {
        try { fn(); }
        catch (err) { console.error(`[GridWise] ${label} render failed:`, err); }
    }

    // ---- Charts -----------------------------------------------------------
    function destroyChart(key) {
        if (state.charts[key]) {
            try { state.charts[key].destroy(); } catch (_) { /* noop */ }
            delete state.charts[key];
        }
    }

    const CHART_FONT = "'JetBrains Mono', monospace";
    const chartGridColor = "rgba(255,255,255,0.05)";
    const chartTickColor = "#94a3b8";

    function commonScales(extra = {}) {
        return {
            x: {
                grid: { color: chartGridColor, drawBorder: false },
                ticks: { color: chartTickColor, font: { family: CHART_FONT, size: 10 } },
                title: { display: true, text: "Hour of Day", color: "#94a3b8", font: { family: CHART_FONT, size: 11 } },
                ...(extra.x || {}),
            },
            y: {
                grid: { color: chartGridColor, drawBorder: false },
                ticks: { color: chartTickColor, font: { family: CHART_FONT, size: 10 } },
                beginAtZero: true,
                ...(extra.y || {}),
            },
        };
    }

    /**
     * Charts inside a hidden (`display:none`) pane render at 0 × 0 because
     * Chart.js can't measure them.  We re-create the chart lazily — every
     * time the canvas becomes visible (tab switch) AND we already have data.
     */
    function renderChartIfVisible(canvasId, key, factory) {
        const canvas = el(canvasId);
        if (!canvas) {
            console.warn(`[GridWise] Missing canvas #${canvasId}`);
            return;
        }
        if (!isVisible(canvas)) {
            // Defer: install a one-shot listener that re-renders on first show.
            const parent = canvas.closest(".pane");
            if (parent && !parent.dataset[`pending_${key}`]) {
                parent.dataset[`pending_${key}`] = "1";
                const obs = new MutationObserver(() => {
                    if (isVisible(canvas)) {
                        obs.disconnect();
                        delete parent.dataset[`pending_${key}`];
                        safeRender(() => factory(), `${key} (deferred)`);
                    }
                });
                obs.observe(parent, { attributes: true, attributeFilter: ["class", "hidden", "style"] });
            }
            return;
        }
        factory();
    }

    function renderMixChart(data) {
        renderChartIfVisible("chart-mix", "mix", () => {
            destroyChart("mix");
            const canvas = el("chart-mix");
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const labels = (data.hourly_plan || []).map((p) => p.hour);

            const gradientSolar = ctx.createLinearGradient(0, 0, 0, 280);
            gradientSolar.addColorStop(0, "rgba(251, 191, 36, 0.85)");
            gradientSolar.addColorStop(1, "rgba(251, 191, 36, 0.3)");

            const gradientGrid = ctx.createLinearGradient(0, 0, 0, 280);
            gradientGrid.addColorStop(0, "rgba(56, 189, 248, 0.85)");
            gradientGrid.addColorStop(1, "rgba(56, 189, 248, 0.3)");

            const discharge = (data.hourly_plan || []).map((p) =>
                p.battery_action === "discharge" ? p.battery_kwh : 0
            );
            const charge = (data.hourly_plan || []).map((p) =>
                p.battery_action === "charge" ? p.battery_kwh : 0
            );

            state.charts.mix = new Chart(ctx, {
                type: "bar",
                data: {
                    labels,
                    datasets: [
                        {
                            label: "Solar (kWh)",
                            data: (data.hourly_plan || []).map((p) => p.solar_used_kwh),
                            backgroundColor: gradientSolar,
                            stack: "supply",
                            borderRadius: 3,
                        },
                        {
                            label: "Grid (kWh)",
                            data: (data.hourly_plan || []).map((p) => p.grid_kwh),
                            backgroundColor: gradientGrid,
                            stack: "supply",
                            borderRadius: 3,
                        },
                        {
                            label: "Battery Discharge",
                            data: discharge,
                            type: "line",
                            borderColor: "#c4b5fd",
                            backgroundColor: "rgba(196,181,253,0.15)",
                            tension: 0.35,
                            fill: false,
                            pointRadius: 2,
                            yAxisID: "y",
                        },
                        {
                            label: "Battery Charge",
                            data: charge,
                            type: "line",
                            borderColor: "#34d399",
                            borderDash: [4, 4],
                            backgroundColor: "rgba(52,211,153,0.15)",
                            tension: 0.35,
                            fill: false,
                            pointRadius: 2,
                            yAxisID: "y",
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 600 },
                    interaction: { mode: "index", intersect: false },
                    plugins: {
                        legend: {
                            labels: { color: "#cbd5e1", font: { family: CHART_FONT, size: 11 } },
                        },
                        tooltip: {
                            backgroundColor: "rgba(4,8,16,0.92)",
                            borderColor: "rgba(34,211,238,0.4)",
                            borderWidth: 1,
                            titleColor: "#fff",
                            bodyColor: "#cbd5e1",
                            padding: 10,
                        },
                    },
                    scales: {
                        x: {
                            stacked: true,
                            grid: { color: chartGridColor },
                            ticks: { color: chartTickColor, font: { family: CHART_FONT, size: 10 } },
                        },
                        y: {
                            stacked: true,
                            grid: { color: chartGridColor },
                            ticks: { color: chartTickColor, font: { family: CHART_FONT, size: 10 } },
                            beginAtZero: true,
                            title: { display: true, text: "kWh", color: "#94a3b8", font: { family: CHART_FONT, size: 11 } },
                        },
                    },
                },
            });
        });
    }

    function renderCostGridChart(data) {
        renderChartIfVisible("chart-cost-grid", "costGrid", () => {
            destroyChart("costGrid");
            const canvas = el("chart-cost-grid");
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const labels = (data.hourly_plan || []).map((p) => p.hour);
            const tariffs = state.tariffs || [];

            const gradGrid = ctx.createLinearGradient(0, 0, 0, 280);
            gradGrid.addColorStop(0, "rgba(56, 189, 248, 0.45)");
            gradGrid.addColorStop(1, "rgba(56, 189, 248, 0.05)");

            state.charts.costGrid = new Chart(ctx, {
                data: {
                    labels,
                    datasets: [
                        {
                            type: "bar",
                            label: "Grid (kWh)",
                            data: (data.hourly_plan || []).map((p) => p.grid_kwh),
                            backgroundColor: gradGrid,
                            borderColor: "#38bdf8",
                            borderWidth: 1,
                            yAxisID: "y",
                            order: 2,
                        },
                        {
                            type: "line",
                            label: "Cost (BDT)",
                            data: (data.hourly_plan || []).map((p, i) => p.grid_kwh * (tariffs[i] || 0)),
                            borderColor: "#c4b5fd",
                            backgroundColor: "rgba(196,181,253,0.2)",
                            tension: 0.35,
                            fill: true,
                            pointRadius: 3,
                            yAxisID: "y1",
                            order: 1,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 600 },
                    interaction: { mode: "index", intersect: false },
                    plugins: {
                        legend: {
                            labels: { color: "#cbd5e1", font: { family: CHART_FONT, size: 11 } },
                        },
                        tooltip: {
                            backgroundColor: "rgba(4,8,16,0.92)",
                            borderColor: "rgba(34,211,238,0.4)",
                            borderWidth: 1,
                            titleColor: "#fff",
                            bodyColor: "#cbd5e1",
                            padding: 10,
                        },
                    },
                    scales: {
                        x: {
                            grid: { color: chartGridColor },
                            ticks: { color: chartTickColor, font: { family: CHART_FONT, size: 10 } },
                        },
                        y: {
                            position: "left",
                            grid: { color: chartGridColor },
                            ticks: { color: chartTickColor, font: { family: CHART_FONT, size: 10 } },
                            beginAtZero: true,
                            title: { display: true, text: "Grid (kWh)", color: "#94a3b8", font: { family: CHART_FONT, size: 11 } },
                        },
                        y1: {
                            position: "right",
                            grid: { display: false },
                            ticks: { color: "#c4b5fd", font: { family: CHART_FONT, size: 10 } },
                            beginAtZero: true,
                            title: { display: true, text: "Cost (BDT)", color: "#c4b5fd", font: { family: CHART_FONT, size: 11 } },
                        },
                    },
                },
            });
        });
    }

    function renderSocChart(data) {
        renderChartIfVisible("chart-soc", "soc", () => {
            destroyChart("soc");
            const canvas = el("chart-soc");
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const labels = (data.hourly_plan || []).map((p) => p.hour);

            const gradient = ctx.createLinearGradient(0, 0, 0, 280);
            gradient.addColorStop(0, "rgba(16, 185, 129, 0.45)");
            gradient.addColorStop(1, "rgba(16, 185, 129, 0.05)");

            state.charts.soc = new Chart(ctx, {
                type: "line",
                data: {
                    labels,
                    datasets: [
                        {
                            label: "State of Charge (kWh)",
                            data: (data.hourly_plan || []).map((p) => p.battery_energy_after_kwh),
                            borderColor: "#10b981",
                            backgroundColor: gradient,
                            tension: 0.35,
                            fill: true,
                            pointRadius: 3,
                            pointBackgroundColor: "#10b981",
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 600 },
                    interaction: { mode: "index", intersect: false },
                    plugins: {
                        legend: {
                            labels: { color: "#cbd5e1", font: { family: CHART_FONT, size: 11 } },
                        },
                        tooltip: {
                            backgroundColor: "rgba(4,8,16,0.92)",
                            borderColor: "rgba(16,185,129,0.4)",
                            borderWidth: 1,
                            titleColor: "#fff",
                            bodyColor: "#cbd5e1",
                            padding: 10,
                        },
                    },
                    scales: commonScales({
                        y: { title: { display: true, text: "kWh", color: "#94a3b8", font: { family: CHART_FONT, size: 11 } } },
                    }),
                },
            });
        });
    }

    // ---- Directives -------------------------------------------------------
    function renderDirectives(directives) {
        const container = el("directive-list");
        if (!container) return;
        if (!Array.isArray(directives) || directives.length === 0) {
            container.innerHTML = '<p class="muted">No directives parsed.</p>';
            return;
        }

        const notesEl = el("operator_notes");
        const lines = notesEl ? notesEl.value.split("\n").map((s) => s.trim()).filter(Boolean) : [];

        container.innerHTML = "";
        directives.forEach((d) => {
            const node = document.createElement("div");
            node.className = `directive ${d.applies ? "applies" : "no-op"}`;
            const adj = d.structured_adjustment || {};
            let adjText = "";
            if (Array.isArray(adj.hours) && adj.hours.length) {
                adjText = `hours [${adj.hours.join(", ")}]`;
            }
            if (adj.factor != null) adjText += ` · factor=${adj.factor}`;
            if (adj.minimum_energy_kwh != null) adjText += ` · min=${adj.minimum_energy_kwh} kWh`;
            if (adj.max_grid_kwh != null) adjText += ` · cap=${adj.max_grid_kwh} kWh`;

            const noteText = lines[d.note_index] || "(note)";

            node.innerHTML = `
                <div class="directive-head">
                    <span class="directive-type ${escape(d.directive_type)}">${escape(d.directive_type)}</span>
                    <span class="directive-meta">#${d.note_index} · ${d.applies ? "applies" : "no-op"}</span>
                </div>
                <div class="directive-note">${escape(noteText)}</div>
                <div class="directive-explain">${escape(d.explanation || "")}</div>
                ${adjText ? `<div class="directive-adjustment">${escape(adjText)}</div>` : ""}
            `;
            container.appendChild(node);
        });
    }

    // ---- Hourly table -----------------------------------------------------
    function renderHourlyTable(data) {
        const tbody = document.querySelector("#hourly-table tbody");
        if (!tbody) { console.warn("[GridWise] Missing #hourly-table tbody"); return; }
        const tariffs = state.tariffs || [];
        const plan = data.hourly_plan || [];

        if (plan.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="muted center">No hourly data.</td></tr>';
            return;
        }

        const rows = plan.map((p) => {
            const tariff = tariffs[p.hour] || 0;
            const cost = (Number(p.grid_kwh) || 0) * tariff;
            return `
                <tr>
                    <td>${String(p.hour).padStart(2, "0")}:00</td>
                    <td class="num">${fmtNumber(tariff, 2)}</td>
                    <td class="num">${fmtNumber((Number(p.solar_used_kwh) || 0) + (Number(p.grid_kwh) || 0) + (Number(p.battery_kwh) || 0), 1)}</td>
                    <td class="num">${fmtNumber(p.solar_used_kwh, 1)}</td>
                    <td class="num">${fmtNumber(p.grid_kwh, 1)}</td>
                    <td><span class="action-pill action-${escape(p.battery_action)}">${escape(p.battery_action)}</span></td>
                    <td class="num">${fmtNumber(p.battery_kwh, 1)}</td>
                    <td class="num">${fmtNumber(p.battery_energy_after_kwh, 1)}</td>
                    <td class="num">${fmtNumber(cost, 2)}</td>
                </tr>
            `;
        }).join("");
        tbody.innerHTML = rows;
    }

    // =========================================================================
    // Analytics render
    // =========================================================================
    function renderAnalytics(resp) {
        if (!resp || !resp.analytics) return;
        const a = resp.analytics;

        setText("ana-savings", fmtNumber(a.savings_bdt, 2));
        setText("ana-savings-pct", `${fmtNumber(a.savings_pct, 1)}% saved`);
        setText("ana-peak-reduction", fmtNumber(a.peak_reduction_kwh, 2));
        setText("ana-solar-util", fmtNumber(a.solar_utilization_pct, 1));
        setText("ana-cycles", fmtNumber(a.equivalent_full_cycles, 2));

        const tbody = document.querySelector("#insights-table tbody");
        if (!tbody) return;
        const hourly = Array.isArray(a.hourly) ? a.hourly : [];
        if (hourly.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="muted center">No analytics yet.</td></tr>';
            return;
        }
        const rows = hourly.map((row) => `
            <tr>
                <td>${String(row.hour).padStart(2, "0")}:00</td>
                <td class="num">${fmtNumber(row.tariff_bdt_per_kwh, 1)}</td>
                <td class="num">${fmtNumber(row.demand_kwh, 1)}</td>
                <td class="num">${fmtNumber(row.effective_solar_kwh, 1)}</td>
                <td class="num">${fmtNumber(row.min_reserve_kwh, 1)}</td>
                <td class="num">${row.max_grid_kwh == null ? "—" : fmtNumber(row.max_grid_kwh, 1)}</td>
                <td class="num">${row.charge_allowed ? '<span class="flag-yes">✓</span>' : '<span class="flag-no">✕</span>'}</td>
                <td class="num">${row.discharge_allowed ? '<span class="flag-yes">✓</span>' : '<span class="flag-no">✕</span>'}</td>
                <td class="num">${fmtNumber(row.baseline_grid_kwh, 1)}</td>
                <td class="num">${fmtNumber(row.cost_bdt, 2)}</td>
            </tr>
        `).join("");
        tbody.innerHTML = rows;
    }

    // =========================================================================
    // Raw JSON
    // =========================================================================
    function renderRawJson() {
        const node = el("raw-json");
        if (!node) return;
        if (state.lastAnalyze) {
            node.textContent = JSON.stringify(state.lastAnalyze, null, 2);
        } else if (state.lastResponse) {
            node.textContent = JSON.stringify(state.lastResponse, null, 2);
        } else {
            node.textContent = "Run an optimization to view the response here.";
        }
    }

    safeAddListener("copy-json", async () => {
        const text = el("raw-json")?.textContent || "";
        try {
            await navigator.clipboard.writeText(text);
            toast("JSON copied to clipboard.", "success");
        } catch {
            toast("Clipboard not available.", "error");
        }
    });

    safeAddListener("export-csv", async () => {
        let payload;
        try { payload = buildRequest(); }
        catch (err) { toast(err.message, "error"); return; }

        try {
            const res = await fetch("/api/export-csv", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error(await safeError(res));
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${payload.scenario_id || "scenario"}_schedule.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            toast("CSV downloaded.", "success");
        } catch (err) {
            toast(err.message, "error");
        }
    });

    // =========================================================================
    // LLM status / health probe
    // =========================================================================
    async function checkStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            const pill = el("llm-status");
            if (pill) {
                if (data.llm_configured) {
                    pill.textContent = `LLM: ${data.llm_models?.[0] || "gemini"}`;
                } else {
                    pill.textContent = "LLM: offline (fallback)";
                    pill.parentElement?.classList.add("offline");
                }
            }
            // expose tariffs for chart
            state.tariffs = (getSampleScenario().hours || []).map((h) => h.tariff_bdt_per_kwh);
            window.__gridwiseTariffs = state.tariffs;
        } catch (err) {
            console.warn("Status check failed:", err);
        }
    }

    // =========================================================================
    // Init
    // =========================================================================
    function init() {
        // Load sample scenario into form on first paint.
        const sample = getSampleScenario();
        const sidEl = el("scenario_id");
        const notesEl = el("operator_notes");
        const hoursEl = el("hours_json");
        if (sidEl) sidEl.value = sample.scenario_id;
        if (notesEl) notesEl.value = sample.operator_notes.join("\n");
        if (hoursEl) hoursEl.value = JSON.stringify(sample.hours, null, 2);

        const map = {
            b_capacity: sample.battery.capacity_kwh,
            b_initial: sample.battery.initial_energy_kwh,
            b_min: sample.battery.minimum_energy_kwh,
            b_charge: sample.battery.max_charge_kwh_per_hour,
            b_discharge: sample.battery.max_discharge_kwh_per_hour,
        };
        for (const [id, val] of Object.entries(map)) {
            const node = el(id);
            if (node) node.value = val;
        }

        state.tariffs = sample.hours.map((h) => h.tariff_bdt_per_kwh);
        window.__gridwiseTariffs = state.tariffs;

        bindRunButton();
        loadPresets();
        checkStatus();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        // DOM is already ready (script tag is at end of body).
        init();
    }
})();
