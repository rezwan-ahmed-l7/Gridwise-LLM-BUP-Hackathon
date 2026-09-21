/* =============================================================================
   GridWise LLM — Dashboard Logic
   Handles API calls, validation, rendering, and Chart.js visualizations.
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
    };

    // =========================================================================
    // DOM helpers
    // =========================================================================
    const $ = (id) => document.getElementById(id);
    const fmtNumber = (n, digits = 2) =>
        Number(n || 0).toLocaleString(undefined, {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
        });

    // =========================================================================
    // Tab navigation
    // =========================================================================
    document.querySelectorAll(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;
            document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
            document.querySelectorAll(".pane").forEach((p) =>
                p.classList.toggle("active", p.id === `pane-${target}`)
            );
            if (target === "results" && state.lastResponse) {
                renderResults(state.lastResponse);
            } else if (target === "analytics" && state.lastAnalyze) {
                renderAnalytics(state.lastAnalyze);
            } else if (target === "raw" && state.lastResponse) {
                renderRawJson();
            }
        });
    });

    // =========================================================================
    // Toast notifications
    // =========================================================================
    function toast(message, type = "info") {
        const container = $("toast-container");
        const el = document.createElement("div");
        el.className = `toast ${type}`;
        el.textContent = message;
        container.appendChild(el);
        setTimeout(() => {
            el.style.opacity = "0";
            el.style.transform = "translateX(20px)";
            setTimeout(() => el.remove(), 300);
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
        const container = $("preset-list");
        try {
            const res = await fetch("/api/presets");
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            state.presets = await res.json();
            renderPresets();
        } catch (err) {
            console.warn("Failed to load presets:", err);
            container.innerHTML = '<p class="muted">No presets available.</p>';
        }
    }

    function renderPresets() {
        const container = $("preset-list");
        const entries = Object.entries(state.presets);
        if (entries.length === 0) {
            container.innerHTML = '<p class="muted">No presets available.</p>';
            return;
        }
        container.innerHTML = "";
        entries.forEach(([id, data]) => {
            const el = document.createElement("div");
            el.className = "preset-item";
            el.innerHTML = `
                <div class="preset-label">${escape(data.label)}</div>
                <div class="preset-id">${escape(id)} · click to load</div>
            `;
            el.addEventListener("click", () => loadScenario(data.input));
            container.appendChild(el);
        });
    }

    function loadScenario(scenario) {
        if (!scenario) return;
        $("scenario_id").value = scenario.scenario_id || "";
        $("operator_notes").value = (scenario.operator_notes || []).join("\n");
        $("hours_json").value = JSON.stringify(scenario.hours || [], null, 2);
        if (scenario.battery) {
            $("b_capacity").value = scenario.battery.capacity_kwh;
            $("b_initial").value = scenario.battery.initial_energy_kwh;
            $("b_min").value = scenario.battery.minimum_energy_kwh;
            $("b_charge").value = scenario.battery.max_charge_kwh_per_hour;
            $("b_discharge").value = scenario.battery.max_discharge_kwh_per_hour;
        }
        toast(`Loaded scenario "${scenario.scenario_id}".`, "success");
    }

    $("load-preset").addEventListener("click", () => {
        const entries = Object.entries(state.presets);
        if (entries.length === 0) return loadScenario(getSampleScenario());
        const [, first] = entries[0];
        loadScenario(first.input);
    });

    $("load-sample-notes").addEventListener("click", () => {
        const sample = getSampleScenario();
        $("scenario_id").value = sample.scenario_id;
        $("operator_notes").value = sample.operator_notes.join("\n");
        $("hours_json").value = JSON.stringify(sample.hours, null, 2);
        if (!state.presets || Object.keys(state.presets).length === 0) {
            $("b_capacity").value = sample.battery.capacity_kwh;
            $("b_initial").value = sample.battery.initial_energy_kwh;
            $("b_min").value = sample.battery.minimum_energy_kwh;
            $("b_charge").value = sample.battery.max_charge_kwh_per_hour;
            $("b_discharge").value = sample.battery.max_discharge_kwh_per_hour;
        }
    });

    // =========================================================================
    // Build request payload from form
    // =========================================================================
    function buildRequest() {
        const scenario_id = $("scenario_id").value.trim() || "SAMPLE-01";
        const notesRaw = $("operator_notes").value.trim();
        if (!notesRaw) throw new Error("At least one operator note is required.");
        const operator_notes = notesRaw.split("\n").map((s) => s.trim()).filter(Boolean);
        if (operator_notes.length === 0 || operator_notes.length > 3) {
            throw new Error("Operator notes must contain 1–3 non-empty lines.");
        }

        let hours;
        try {
            hours = JSON.parse($("hours_json").value);
        } catch (err) {
            throw new Error(`Hours JSON is invalid: ${err.message}`);
        }
        if (!Array.isArray(hours) || hours.length !== 24) {
            throw new Error("Hours array must contain exactly 24 entries.");
        }

        const battery = {
            capacity_kwh: parseFloat($("b_capacity").value),
            initial_energy_kwh: parseFloat($("b_initial").value),
            minimum_energy_kwh: parseFloat($("b_min").value),
            max_charge_kwh_per_hour: parseFloat($("b_charge").value),
            max_discharge_kwh_per_hour: parseFloat($("b_discharge").value),
        };
        for (const [k, v] of Object.entries(battery)) {
            if (!Number.isFinite(v)) throw new Error(`Battery field "${k}" is invalid.`);
        }

        return { scenario_id, operator_notes, hours, battery };
    }

    // =========================================================================
    // Run optimization
    // =========================================================================
    $("run-optimize").addEventListener("click", async () => {
        const btn = $("run-optimize");
        const errEl = $("form-error");
        errEl.textContent = "";

        let payload;
        try {
            payload = buildRequest();
        } catch (err) {
            errEl.textContent = err.message;
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

            // Also try to fetch analytics (non-blocking if it fails)
            try {
                const ar = await fetch("/api/analyze", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (ar.ok) state.lastAnalyze = await ar.json();
            } catch (e) {
                console.warn("Analytics call failed:", e);
            }

            renderResults(data);
            switchTab("results");
            toast("Optimization complete.", "success");
        } catch (err) {
            errEl.textContent = err.message;
            toast(err.message, "error");
        } finally {
            btn.classList.remove("is-loading");
            btn.disabled = false;
        }
    });

    async function safeError(res) {
        try {
            const j = await res.json();
            return j.detail || JSON.stringify(j);
        } catch {
            return `HTTP ${res.status}`;
        }
    }

    function switchTab(name) {
        document.querySelectorAll(".tab").forEach((t) =>
            t.classList.toggle("active", t.dataset.tab === name)
        );
        document.querySelectorAll(".pane").forEach((p) =>
            p.classList.toggle("active", p.id === `pane-${name}`)
        );
    }

    // =========================================================================
    // Render results
    // =========================================================================
    function renderResults(data) {
        // KPIs
        $("kpi-cost").textContent = fmtNumber(data.total_cost_bdt, 2);
        $("kpi-grid").textContent = fmtNumber(data.total_grid_kwh, 2);
        $("kpi-peak").textContent = fmtNumber(data.peak_grid_kwh, 2);
        const totalSolar = data.hourly_plan.reduce((s, p) => s + p.solar_used_kwh, 0);
        $("kpi-solar").textContent = fmtNumber(totalSolar, 2);

        // Charts
        renderMixChart(data);
        renderCostGridChart(data);
        renderSocChart(data);

        // Directives
        renderDirectives(data.directive_interpretation);

        // Hourly table
        renderHourlyTable(data);

        // Warnings
        if (data.warnings && data.warnings.length) {
            $("warnings-card").hidden = false;
            $("warning-list").innerHTML = data.warnings
                .map((w) => `<li>⚠ ${escape(w)}</li>`)
                .join("");
        } else {
            $("warnings-card").hidden = true;
        }

        // Analytics tab content
        if (state.lastAnalyze) renderAnalytics(state.lastAnalyze);

        // Raw JSON
        renderRawJson();
    }

    // ---- Charts -----------------------------------------------------------
    function destroyChart(key) {
        if (state.charts[key]) {
            state.charts[key].destroy();
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

    function renderMixChart(data) {
        destroyChart("mix");
        const labels = data.hourly_plan.map((p) => p.hour);
        const ctx = document.getElementById("chart-mix").getContext("2d");

        const gradientSolar = ctx.createLinearGradient(0, 0, 0, 280);
        gradientSolar.addColorStop(0, "rgba(251, 191, 36, 0.85)");
        gradientSolar.addColorStop(1, "rgba(251, 191, 36, 0.3)");

        const gradientGrid = ctx.createLinearGradient(0, 0, 0, 280);
        gradientGrid.addColorStop(0, "rgba(56, 189, 248, 0.85)");
        gradientGrid.addColorStop(1, "rgba(56, 189, 248, 0.3)");

        const discharge = data.hourly_plan.map((p) =>
            p.battery_action === "discharge" ? p.battery_kwh : 0
        );
        const charge = data.hourly_plan.map((p) =>
            p.battery_action === "charge" ? p.battery_kwh : 0
        );

        state.charts.mix = new Chart(ctx, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    {
                        label: "Solar (kWh)",
                        data: data.hourly_plan.map((p) => p.solar_used_kwh),
                        backgroundColor: gradientSolar,
                        stack: "supply",
                        borderRadius: 3,
                    },
                    {
                        label: "Grid (kWh)",
                        data: data.hourly_plan.map((p) => p.grid_kwh),
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
    }

    function renderCostGridChart(data) {
        destroyChart("costGrid");
        const labels = data.hourly_plan.map((p) => p.hour);
        const ctx = document.getElementById("chart-cost-grid").getContext("2d");

        const ctx1 = ctx;
        const gradGrid = ctx1.createLinearGradient(0, 0, 0, 280);
        gradGrid.addColorStop(0, "rgba(56, 189, 248, 0.45)");
        gradGrid.addColorStop(1, "rgba(56, 189, 248, 0.05)");

        state.charts.costGrid = new Chart(ctx1, {
            data: {
                labels,
                datasets: [
                    {
                        type: "bar",
                        label: "Grid (kWh)",
                        data: data.hourly_plan.map((p) => p.grid_kwh),
                        backgroundColor: gradGrid,
                        borderColor: "#38bdf8",
                        borderWidth: 1,
                        yAxisID: "y",
                        order: 2,
                    },
                    {
                        type: "line",
                        label: "Cost (BDT)",
                        data: data.hourly_plan.map((p, i) => {
                            const tariff = window.__gridwiseTariffs ? window.__gridwiseTariffs[i] : 0;
                            return p.grid_kwh * tariff;
                        }),
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
    }

    function renderSocChart(data) {
        destroyChart("soc");
        const labels = data.hourly_plan.map((p) => p.hour);
        const ctx = document.getElementById("chart-soc").getContext("2d");

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
                        data: data.hourly_plan.map((p) => p.battery_energy_after_kwh),
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
    }

    // ---- Directives -------------------------------------------------------
    function renderDirectives(directives) {
        const container = $("directive-list");
        if (!directives || directives.length === 0) {
            container.innerHTML = '<p class="muted">No directives parsed.</p>';
            return;
        }
        container.innerHTML = "";
        directives.forEach((d) => {
            const el = document.createElement("div");
            el.className = `directive ${d.applies ? "applies" : "no-op"}`;
            const adj = d.structured_adjustment || {};
            let adjText = "";
            if (adj.hours && adj.hours.length) {
                adjText = `hours [${adj.hours.join(", ")}]`;
            }
            if (adj.factor != null) adjText += ` · factor=${adj.factor}`;
            if (adj.minimum_energy_kwh != null) adjText += ` · min=${adj.minimum_energy_kwh} kWh`;
            if (adj.max_grid_kwh != null) adjText += ` · cap=${adj.max_grid_kwh} kWh`;

            const noteText = (state.lastResponse && state.lastResponse._rawNotes && state.lastResponse._rawNotes[d.note_index])
                || (() => {
                    // Pull from form as a fallback
                    const lines = $("operator_notes").value.split("\n").map((s) => s.trim()).filter(Boolean);
                    return lines[d.note_index] || "(note)";
                })();

            el.innerHTML = `
                <div class="directive-head">
                    <span class="directive-type ${d.directive_type}">${escape(d.directive_type)}</span>
                    <span class="directive-meta">#${d.note_index} · ${d.applies ? "applies" : "no-op"}</span>
                </div>
                <div class="directive-note">${escape(noteText)}</div>
                <div class="directive-explain">${escape(d.explanation || "")}</div>
                ${adjText ? `<div class="directive-adjustment">${escape(adjText)}</div>` : ""}
            `;
            container.appendChild(el);
        });
    }

    // ---- Hourly table -----------------------------------------------------
    function renderHourlyTable(data) {
        const tbody = document.querySelector("#hourly-table tbody");
        const tariffs = window.__gridwiseTariffs || [];
        let rows = "";
        data.hourly_plan.forEach((p) => {
            const tariff = tariffs[p.hour] || 0;
            const cost = p.grid_kwh * tariff;
            rows += `
                <tr>
                    <td>${String(p.hour).padStart(2, "0")}:00</td>
                    <td class="num">${fmtNumber(tariff, 2)}</td>
                    <td class="num">${fmtNumber(p.solar_used_kwh + p.grid_kwh + p.battery_kwh, 1)}</td>
                    <td class="num">${fmtNumber(p.solar_used_kwh, 1)}</td>
                    <td class="num">${fmtNumber(p.grid_kwh, 1)}</td>
                    <td><span class="action-pill action-${p.battery_action}">${p.battery_action}</span></td>
                    <td class="num">${fmtNumber(p.battery_kwh, 1)}</td>
                    <td class="num">${fmtNumber(p.battery_energy_after_kwh, 1)}</td>
                    <td class="num">${fmtNumber(cost, 2)}</td>
                </tr>
            `;
        });
        tbody.innerHTML = rows;
    }

    // =========================================================================
    // Analytics render
    // =========================================================================
    function renderAnalytics(resp) {
        const a = resp.analytics;

        $("ana-savings").textContent = fmtNumber(a.savings_bdt, 2);
        $("ana-savings-pct").textContent = `${fmtNumber(a.savings_pct, 1)}% saved`;
        $("ana-peak-reduction").textContent = fmtNumber(a.peak_reduction_kwh, 2);
        $("ana-solar-util").textContent = fmtNumber(a.solar_utilization_pct, 1);
        $("ana-cycles").textContent = fmtNumber(a.equivalent_full_cycles, 2);

        const tbody = document.querySelector("#insights-table tbody");
        let rows = "";
        a.hourly.forEach((row) => {
            rows += `
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
            `;
        });
        tbody.innerHTML = rows;
    }

    // =========================================================================
    // Raw JSON
    // =========================================================================
    function renderRawJson() {
        const el = $("raw-json");
        if (state.lastAnalyze) {
            el.textContent = JSON.stringify(state.lastAnalyze, null, 2);
        } else if (state.lastResponse) {
            el.textContent = JSON.stringify(state.lastResponse, null, 2);
        }
    }

    $("copy-json").addEventListener("click", async () => {
        const text = $("raw-json").textContent;
        try {
            await navigator.clipboard.writeText(text);
            toast("JSON copied to clipboard.", "success");
        } catch {
            toast("Clipboard not available.", "error");
        }
    });

    $("export-csv").addEventListener("click", async () => {
        let payload;
        try {
            payload = buildRequest();
        } catch (err) {
            toast(err.message, "error");
            return;
        }
        try {
            const res = await fetch("/api/export-csv", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const detail = await safeError(res);
                throw new Error(detail);
            }
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
            const pill = $("llm-status");
            if (data.llm_configured) {
                pill.textContent = `LLM: ${data.llm_models[0] || "gemini"}`;
            } else {
                pill.textContent = "LLM: offline (fallback)";
                pill.parentElement.classList.add("offline");
            }
            // expose tariffs for chart
            window.__gridwiseTariffs = (getSampleScenario().hours || []).map(
                (h) => h.tariff_bdt_per_kwh
            );
        } catch (err) {
            console.warn("Status check failed:", err);
        }
    }

    // =========================================================================
    // Utilities
    // =========================================================================
    function escape(str) {
        return String(str ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // =========================================================================
    // Init
    // =========================================================================
    function init() {
        // Load sample scenario into form on first paint
        const sample = getSampleScenario();
        $("scenario_id").value = sample.scenario_id;
        $("operator_notes").value = sample.operator_notes.join("\n");
        $("hours_json").value = JSON.stringify(sample.hours, null, 2);
        $("b_capacity").value = sample.battery.capacity_kwh;
        $("b_initial").value = sample.battery.initial_energy_kwh;
        $("b_min").value = sample.battery.minimum_energy_kwh;
        $("b_charge").value = sample.battery.max_charge_kwh_per_hour;
        $("b_discharge").value = sample.battery.max_discharge_kwh_per_hour;

        window.__gridwiseTariffs = sample.hours.map((h) => h.tariff_bdt_per_kwh);

        loadPresets();
        checkStatus();
    }

    document.addEventListener("DOMContentLoaded", init);
})();
