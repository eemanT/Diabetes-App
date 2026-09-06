/* =========================================================================
   Diabetes Prediction — Frontend app.js
   Wires the page to the Flask API:
     GET  /api/results               → metrics, dataset, rules, importance
     GET  /api/sample-patient/<idx>  → real patient data (no dummy values)
     POST /api/predict               → ANN + ANFIS predictions + SHAP + top rules
   ========================================================================= */

const FEATURES = [
  "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
];

const FEATURE_META = {
  Pregnancies:              { unit: "count",   step: 1,    desc: "Number of times pregnant" },
  Glucose:                  { unit: "mg/dL",   step: 1,    desc: "Plasma glucose at 2h, oral glucose tolerance test" },
  BloodPressure:            { unit: "mm Hg",   step: 1,    desc: "Diastolic blood pressure" },
  SkinThickness:            { unit: "mm",      step: 1,    desc: "Triceps skinfold thickness" },
  Insulin:                  { unit: "µU/mL",   step: 1,    desc: "2-hour serum insulin" },
  BMI:                      { unit: "kg/m²",   step: 0.1,  desc: "Body mass index" },
  DiabetesPedigreeFunction: { unit: "score",   step: 0.001,desc: "Family history risk score" },
  Age:                      { unit: "years",   step: 1,    desc: "Age" },
};

// Chart palette — keep consistent with CSS variables
const C = {
  ann:        "#0f766e",  // teal
  annSoft:    "rgba(15, 118, 110, 0.18)",
  anfis:      "#e11d48",  // rose
  anfisSoft:  "rgba(225, 29, 72, 0.18)",
  good:       "#16a34a",  // green
  bad:        "#dc2626",  // red
  line:       "#e2e8f0",
  muted:      "#64748b",
  ink:        "#0f172a",
};

// Holds Chart.js instances so we can destroy/recreate them
const charts = {};

let RESULTS = null;  // populated by /api/results

// ===== Professional Chart.js defaults (applied once, before any chart) =====
if (window.Chart) {
  Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = C.muted;
  Chart.defaults.borderColor = C.line;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.legend.labels.boxHeight = 12;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.padding = 12;
  Chart.defaults.plugins.tooltip.backgroundColor = "rgba(15, 23, 42, 0.92)";
  Chart.defaults.plugins.tooltip.titleColor = "#fff";
  Chart.defaults.plugins.tooltip.bodyColor = "#fff";
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
  Chart.defaults.plugins.tooltip.titleFont = { weight: "600", size: 12 };
  Chart.defaults.plugins.tooltip.bodyFont = { size: 12 };
  Chart.defaults.elements.bar.borderRadius = 4;
  Chart.defaults.elements.bar.borderSkipped = false;
  Chart.defaults.layout.padding = 4;
}

// ===== Bootstrap =====
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupButtons();
  try {
    const res = await fetch("/api/results");
    if (!res.ok) throw new Error(`API error ${res.status}`);
    RESULTS = await res.json();
    renderEverything();
  } catch (err) {
    document.querySelector("main").insertAdjacentHTML(
      "afterbegin",
      `<div class="card error">Failed to load model results: ${err.message}.
       Make sure the backend is running on the same host.</div>`
    );
    console.error(err);
  }
});

// ===== Tabs =====
function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.toggle("active", t === tab));
      panels.forEach(p => p.classList.toggle("active", p.id === target));
    });
  });
}

// ===== Buttons =====
function setupButtons() {
  document.getElementById("predictBtn").addEventListener("click", runPrediction);
  document.getElementById("clearBtn").addEventListener("click", clearForm);
  document.getElementById("loadSample").addEventListener("click", loadSamplePatient);
}

// ===== Render driver =====
function renderEverything() {
  renderForm();
  renderMetricsTab();
  renderDatasetTab();
  renderRulesTab();
}

// ============================================================
// PREDICT TAB
// ============================================================
function renderForm() {
  const stats = RESULTS.dataset.feature_stats;
  const form = document.getElementById("predictForm");

  // Format a number to match the field's step precision
  const fmt = (v, step) => {
    if (step >= 1) return String(Math.round(v));
    if (step >= 0.1) return v.toFixed(1);
    return v.toFixed(3);
  };

  form.innerHTML = FEATURES.map(f => {
    const s = stats[f];
    const m = FEATURE_META[f];
    const defaultVal = fmt(s.median, m.step);
    // Hard limits come from the actual training-data range. Anything outside
    // this is out-of-distribution for both models. We keep min=0 so the API's
    // "0 means missing -> impute median" behavior still works for clinical
    // fields, but cap max at the dataset's observed maximum.
    const minVal = 0;
    const maxVal = fmt(s.max, m.step);
    const rangeMin = fmt(s.min, m.step);
    const rangeMax = fmt(s.max, m.step);
    return `
      <div class="field">
        <label for="in-${f}">
          ${f}
          <span class="unit">(${m.unit})</span>
        </label>
        <input type="number"
               id="in-${f}"
               name="${f}"
               step="${m.step}"
               min="${minVal}"
               max="${maxVal}"
               value="${defaultVal}"
               title="${m.desc} | Training range: ${rangeMin} to ${rangeMax}" />
        <div class="field-hint">Typical: ${rangeMin} – ${rangeMax}</div>
      </div>
    `;
  }).join("");

  // Live clamp: if the user types a value outside [min, max], snap it back
  // and flash the field briefly so they know it was adjusted.
  for (const f of FEATURES) {
    const el = document.getElementById(`in-${f}`);
    el.addEventListener("blur", () => clampInput(el));
  }
}

function clampInput(el) {
  const v = parseFloat(el.value);
  if (Number.isNaN(v)) return;
  const min = parseFloat(el.min);
  const max = parseFloat(el.max);
  let clamped = v;
  if (v < min) clamped = min;
  else if (v > max) clamped = max;
  if (clamped !== v) {
    el.value = clamped;
    el.classList.add("clamped");
    setTimeout(() => el.classList.remove("clamped"), 1200);
  }
}

function readFormValues() {
  const out = {};
  for (const f of FEATURES) {
    const el = document.getElementById(`in-${f}`);
    const v = parseFloat(el.value);
    if (Number.isNaN(v)) return { error: `Invalid value for ${f}` };
    out[f] = v;
  }
  return { values: out };
}

function clearForm() {
  for (const f of FEATURES) {
    document.getElementById(`in-${f}`).value = "";
  }
  document.getElementById("predictionResults").innerHTML =
    `<p class="muted center">Enter features and click <strong>Predict</strong> to see results from both models.</p>`;
  document.getElementById("explainCard").style.display = "none";
}

async function loadSamplePatient() {
  const idxInput = document.getElementById("sampleIdx");
  const idx = parseInt(idxInput.value || "0", 10);
  if (Number.isNaN(idx) || idx < 0 || idx > 767) {
    alert("Index must be between 0 and 767");
    return;
  }
  const btn = document.getElementById("loadSample");
  btn.disabled = true;
  const original = btn.textContent;
  btn.innerHTML = `<span class="loading"></span>Loading...`;
  try {
    const res = await fetch(`/api/sample-patient/${idx}`);
    if (!res.ok) throw new Error(`API error ${res.status}`);
    const data = await res.json();
    for (const f of FEATURES) {
      document.getElementById(`in-${f}`).value = data.features[f];
    }
    // Show ground-truth label below the load button
    showSampleLabel(idx, data.actual_outcome);
  } catch (err) {
    alert(`Failed to load patient: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

function showSampleLabel(idx, outcome) {
  let el = document.getElementById("sampleLabel");
  if (!el) {
    el = document.createElement("div");
    el.id = "sampleLabel";
    el.style.cssText = "font-size:12px;margin-top:6px;color:var(--muted);";
    document.querySelector(".row-actions").after(el);
  }
  const label = outcome === 1 ? "Diabetic" : "Non-diabetic";
  const color = outcome === 1 ? "var(--accent)" : "var(--good)";
  el.innerHTML = `Loaded patient #${idx} — actual diagnosis in dataset:
    <strong style="color:${color}">${label}</strong>`;
}

async function runPrediction() {
  const { values, error } = readFormValues();
  if (error) { alert(error); return; }

  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  const original = btn.textContent;
  btn.innerHTML = `<span class="loading"></span>Predicting...`;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.error || `API error ${res.status}`);
    }
    const data = await res.json();
    renderPredictionResults(data);
    renderExplanation(data);
  } catch (err) {
    document.getElementById("predictionResults").innerHTML =
      `<div class="error">Prediction failed: ${err.message}</div>`;
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

function renderPredictionResults(data) {
  const { ann, anfis } = data;
  const agree = ann.prediction === anfis.prediction;

  const card = (name, color, pred) => {
    const labelClass = pred.prediction === 1 ? "diabetic" : "healthy";
    const fillClass  = pred.prediction === 1 ? "diabetic" : "healthy";
    const pct = (pred.probability * 100).toFixed(1);
    return `
      <div class="pred-card" style="border-left:3px solid ${color}">
        <h4>${name}</h4>
        <div class="pred-label ${labelClass}">${pred.label}</div>
        <div class="pred-prob">Probability of diabetes: <strong>${pct}%</strong></div>
        <div class="probability-bar"><div class="probability-fill ${fillClass}" style="width:${pct}%"></div></div>
      </div>
    `;
  };

  const agreementHtml = agree
    ? `<div class="agreement agree">✓ Both models agree — high confidence.</div>`
    : `<div class="agreement disagree">⚠ Models disagree — borderline case, consider clinical review.</div>`;

  document.getElementById("predictionResults").innerHTML = `
    <div class="pred-grid">
      ${card("ANN", C.ann, ann)}
      ${card("ANFIS", C.anfis, anfis)}
    </div>
    ${agreementHtml}
  `;
}

function renderExplanation(data) {
  document.getElementById("explainCard").style.display = "block";
  renderShapChart(data.shap_explanation.features);
  renderTopRules(data.top_fuzzy_rules);
  // Smooth-scroll to the explanation card
  document.getElementById("explainCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderShapChart(features) {
  // Already sorted by |shap| desc from API
  const ctx = document.getElementById("shapChart");
  if (charts.shap) charts.shap.destroy();
  const labels = features.map(f => f.feature);
  const values = features.map(f => f.shap);
  const colors = values.map(v => v >= 0 ? C.bad : C.good);

  charts.shap = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "SHAP value",
        data: values,
        backgroundColor: colors,
        borderWidth: 0,
        barThickness: 16,
        maxBarThickness: 20,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const f = features[ctx.dataIndex];
              const dir = ctx.raw >= 0 ? "↑ diabetic" : "↓ non-diabetic";
              return `SHAP: ${ctx.raw.toFixed(4)}  (${dir})  · value: ${f.value}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: C.line, drawBorder: false },
          ticks: { font: { size: 11 }, callback: (v) => v.toFixed(2) },
          title: { display: true, text: "SHAP value (impact on prediction)", font: { size: 11, weight: "600" }, color: C.muted },
        },
        y: {
          grid: { display: false, drawBorder: false },
          ticks: { font: { size: 11 }, color: C.ink },
        },
      },
    },
  });
}

function renderTopRules(rules) {
  const container = document.getElementById("topRules");
  container.innerHTML = rules.map(r => ruleCardHtml(r, true)).join("");
}

// ============================================================
// METRICS TAB
// ============================================================
function renderMetricsTab() {
  renderMetricsGrid();
  renderMetricsChart();
  renderRocChart();
  renderConfusion("cmAnn", RESULTS.ann.confusion_matrix);
  renderConfusion("cmAnfis", RESULTS.anfis.confusion_matrix);
  renderLossChart("annLossChart", RESULTS.ann.history, C.ann, C.annSoft);
  renderLossChart("anfisLossChart", RESULTS.anfis.history, C.anfis, C.anfisSoft);
  renderGlobalImportance();
}

function renderMetricsGrid() {
  const m = (which) => RESULTS[which];
  const tile = (cls, name, value, sub) => `
    <div class="metric-tile ${cls}">
      <div class="label">${name}</div>
      <div class="value">${value}</div>
      <div class="sublabel">${sub}</div>
    </div>`;
  const fmt = v => (v * 100).toFixed(2) + "%";
  const fmtRaw = v => v.toFixed(4);

  const annM = m("ann"), anfisM = m("anfis");
  document.getElementById("metricsGrid").innerHTML = `
    ${tile("ann",   "ANN Accuracy",  fmt(annM.accuracy),    `${annM.history.total_epochs} epochs trained`)}
    ${tile("ann",   "ANN F1",        fmtRaw(annM.f1),       `precision ${fmtRaw(annM.precision)} · recall ${fmtRaw(annM.recall)}`)}
    ${tile("ann",   "ANN AUC-ROC",   fmtRaw(annM.auc_roc),  "discrimination across thresholds")}
    ${tile("anfis", "ANFIS Accuracy",fmt(anfisM.accuracy),  `${anfisM.history.total_epochs} epochs · ${RESULTS.n_rules} rules`)}
    ${tile("anfis", "ANFIS F1",      fmtRaw(anfisM.f1),     `precision ${fmtRaw(anfisM.precision)} · recall ${fmtRaw(anfisM.recall)}`)}
    ${tile("anfis", "ANFIS AUC-ROC", fmtRaw(anfisM.auc_roc),"discrimination across thresholds")}
  `;
}

function renderMetricsChart() {
  const ctx = document.getElementById("metricsChart");
  if (charts.metrics) charts.metrics.destroy();
  const labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"];
  const annData   = [RESULTS.ann.accuracy,   RESULTS.ann.precision,   RESULTS.ann.recall,   RESULTS.ann.f1,   RESULTS.ann.auc_roc];
  const anfisData = [RESULTS.anfis.accuracy, RESULTS.anfis.precision, RESULTS.anfis.recall, RESULTS.anfis.f1, RESULTS.anfis.auc_roc];
  charts.metrics = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "ANN",   data: annData,   backgroundColor: C.ann,   maxBarThickness: 28 },
        { label: "ANFIS", data: anfisData, backgroundColor: C.anfis, maxBarThickness: 28 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", align: "end" },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(4)}` } },
      },
      scales: {
        y: {
          beginAtZero: true, max: 1,
          grid: { color: C.line, drawBorder: false },
          ticks: { font: { size: 11 }, stepSize: 0.2, callback: (v) => v.toFixed(1) },
        },
        x: {
          grid: { display: false, drawBorder: false },
          ticks: { font: { size: 11 }, color: C.ink },
        },
      },
    },
  });
}

function renderRocChart() {
  const ctx = document.getElementById("rocChart");
  if (charts.roc) charts.roc.destroy();
  const annRoc = RESULTS.ann.roc;
  const anfisRoc = RESULTS.anfis.roc;
  const diag = [{x: 0, y: 0}, {x: 1, y: 1}];

  charts.roc = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: `ANN (AUC = ${RESULTS.ann.auc_roc.toFixed(3)})`,
          data: annRoc.fpr.map((f, i) => ({ x: f, y: annRoc.tpr[i] })),
          borderColor: C.ann, backgroundColor: C.annSoft,
          tension: 0.1, pointRadius: 0, borderWidth: 2,
        },
        {
          label: `ANFIS (AUC = ${RESULTS.anfis.auc_roc.toFixed(3)})`,
          data: anfisRoc.fpr.map((f, i) => ({ x: f, y: anfisRoc.tpr[i] })),
          borderColor: C.anfis, backgroundColor: C.anfisSoft,
          tension: 0.1, pointRadius: 0, borderWidth: 2,
        },
        {
          label: "Random classifier",
          data: diag, borderColor: C.muted,
          borderDash: [4, 4], pointRadius: 0, borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", align: "center" } },
      scales: {
        x: {
          type: "linear", min: 0, max: 1,
          title: { display: true, text: "False positive rate", font: { size: 11, weight: "600" }, color: C.muted },
          grid: { color: C.line, drawBorder: false },
          ticks: { font: { size: 11 }, stepSize: 0.2, callback: (v) => v.toFixed(1) },
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: "True positive rate", font: { size: 11, weight: "600" }, color: C.muted },
          grid: { color: C.line, drawBorder: false },
          ticks: { font: { size: 11 }, stepSize: 0.2, callback: (v) => v.toFixed(1) },
        },
      },
    },
  });
}

function renderConfusion(elemId, cm) {
  // cm = [[TN, FP], [FN, TP]] from sklearn
  const [[tn, fp], [fn, tp]] = cm;
  document.getElementById(elemId).innerHTML = `
    <div></div>
    <div class="axis-label">Predicted 0</div>
    <div class="axis-label">Predicted 1</div>
    <div class="axis-label">Actual 0</div>
    <div class="cell tn">${tn}<span class="lab">True negative</span></div>
    <div class="cell fp">${fp}<span class="lab">False positive</span></div>
    <div class="axis-label">Actual 1</div>
    <div class="cell fn">${fn}<span class="lab">False negative</span></div>
    <div class="cell tp">${tp}<span class="lab">True positive</span></div>
  `;
}

function renderLossChart(elemId, history, color, colorSoft) {
  const ctx = document.getElementById(elemId);
  if (charts[elemId]) charts[elemId].destroy();
  charts[elemId] = new Chart(ctx, {
    type: "line",
    data: {
      labels: history.epochs,
      datasets: [
        { label: "Train loss",      data: history.loss,     borderColor: color,    backgroundColor: colorSoft, tension: 0.2, pointRadius: 0, borderWidth: 2 },
        { label: "Validation loss", data: history.val_loss, borderColor: C.muted,  backgroundColor: "transparent", tension: 0.2, pointRadius: 0, borderWidth: 2, borderDash: [4, 4] },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "top", align: "end" } },
      scales: {
        x: {
          title: { display: true, text: "Epoch", font: { size: 11, weight: "600" }, color: C.muted },
          grid: { color: C.line, drawBorder: false },
          ticks: { font: { size: 11 }, maxTicksLimit: 8 },
        },
        y: {
          title: { display: true, text: "Binary cross-entropy", font: { size: 11, weight: "600" }, color: C.muted },
          grid: { color: C.line, drawBorder: false },
          ticks: { font: { size: 11 } },
        },
      },
    },
  });
}

function renderGlobalImportance() {
  const ctx = document.getElementById("globalImportanceChart");
  if (charts.globalImportance) charts.globalImportance.destroy();
  const sorted = [...RESULTS.global_importance].sort((a, b) => b.importance - a.importance);
  charts.globalImportance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(d => d.feature),
      datasets: [{
        label: "Mean |SHAP|",
        data: sorted.map(d => d.importance),
        backgroundColor: C.ann,
        borderWidth: 0,
        barThickness: 16,
        maxBarThickness: 20,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `Mean |SHAP|: ${ctx.raw.toFixed(4)}` } },
      },
      scales: {
        x: {
          title: { display: true, text: "Mean absolute SHAP value (test set)", font: { size: 11, weight: "600" }, color: C.muted },
          grid: { color: C.line, drawBorder: false },
          ticks: { font: { size: 11 }, callback: (v) => v.toFixed(3) },
        },
        y: {
          grid: { display: false, drawBorder: false },
          ticks: { font: { size: 11 }, color: C.ink },
        },
      },
    },
  });
}

// ============================================================
// DATASET TAB
// ============================================================
function renderDatasetTab() {
  renderDatasetStats();
  renderClassChart();
  renderZeroImputation();
  renderFeaturesTable();
}

function renderDatasetStats() {
  const d = RESULTS.dataset;
  const tile = (name, value, sub) => `
    <div class="metric-tile">
      <div class="label">${name}</div>
      <div class="value">${value}</div>
      <div class="sublabel">${sub}</div>
    </div>`;
  document.getElementById("datasetStats").innerHTML = `
    ${tile("Total patients", d.n_total, "Pima Indians")}
    ${tile("Training set", d.n_train, "80% stratified")}
    ${tile("Test set", d.n_test, "20% held out")}
    ${tile("Features", d.n_features, "clinical measurements")}
    ${tile("Diabetic", d.outcome_counts["1"], `${(100 * d.outcome_counts["1"] / d.n_total).toFixed(1)}% of patients`)}
    ${tile("Non-diabetic", d.outcome_counts["0"], `${(100 * d.outcome_counts["0"] / d.n_total).toFixed(1)}% of patients`)}
  `;
}

function renderClassChart() {
  const ctx = document.getElementById("classChart");
  if (charts.classDist) charts.classDist.destroy();
  const d = RESULTS.dataset.outcome_counts;
  const total = d["0"] + d["1"];
  charts.classDist = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Non-diabetic (0)", "Diabetic (1)"],
      datasets: [{
        data: [d["0"], d["1"]],
        backgroundColor: [C.good, C.bad],
        borderColor: "#fff",
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: {
          position: "right",
          labels: { font: { size: 12 }, color: C.ink, padding: 14 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: ${ctx.raw} (${(100 * ctx.raw / total).toFixed(1)}%)`,
          },
        },
      },
    },
  });
}

function renderZeroImputation() {
  const z = RESULTS.dataset.zero_imputation;
  const total = RESULTS.dataset.n_total;
  const html = Object.entries(z)
    .sort((a, b) => b[1] - a[1])
    .map(([feat, count]) => `
      <div class="zero-row">
        <span class="name">${feat}</span>
        <span class="count">${count} zeros → median imputed
          (${(100 * count / total).toFixed(1)}% of dataset)</span>
      </div>
    `).join("");
  document.getElementById("zeroImputation").innerHTML = html;
}

function renderFeaturesTable() {
  const stats = RESULTS.dataset.feature_stats;
  const html = FEATURES.map((f, i) => {
    const s = stats[f];
    const m = FEATURE_META[f];
    return `
      <div class="feature-row">
        <div class="num">${i + 1}</div>
        <div>
          <div class="name">${f} <span class="unit" style="color:var(--muted);font-weight:400;font-size:12px">(${m.unit})</span></div>
          <div class="desc">${m.desc}</div>
        </div>
        <div class="stats">range: ${s.min.toFixed(s.min % 1 ? 2 : 0)} – ${s.max.toFixed(s.max % 1 ? 2 : 0)}</div>
        <div class="stats">mean: ${s.mean.toFixed(2)}<br>median: ${s.median.toFixed(2)}</div>
      </div>
    `;
  }).join("");
  document.getElementById("featuresTable").innerHTML = html;
}

// ============================================================
// RULES TAB
// ============================================================
function renderRulesTab() {
  const html = RESULTS.fuzzy_rules.map(r => ruleCardHtml(r, false)).join("");
  document.getElementById("rulesList").innerHTML = html;
}

function ruleCardHtml(rule, showFiring) {
  const diabetic = rule.conclusion.includes("HIGH");
  const conclusionClass = diabetic ? "diabetic" : "healthy";
  const conclusionLabel = diabetic ? "HIGH risk" : "LOW risk";

  // Antecedents → IF clause
  const antecedents = rule.antecedents.map(a => {
    const termClass = `term-${a.label.toLowerCase()}`;
    return `<strong>${a.feature}</strong> is <span class="${termClass}">${a.label}</span>`;
  });
  const ifClause = antecedents.join(" AND<br>&nbsp;&nbsp;&nbsp;&nbsp;");

  const firingHtml = showFiring && rule.firing_strength !== undefined
    ? `<div class="rule-firing">Firing strength for this patient: <strong>${(rule.firing_strength * 100).toFixed(1)}%</strong></div>`
    : "";

  const supportHtml = rule.support_probability !== undefined && !showFiring
    ? `<div class="rule-firing">Average predicted probability when this rule dominates: <strong>${(rule.support_probability * 100).toFixed(1)}%</strong></div>`
    : "";

  return `
    <div class="rule-card">
      <div class="rule-header">
        <span class="rule-id">RULE ${rule.rule_id}</span>
        <span class="rule-conclusion ${conclusionClass}">→ ${conclusionLabel}</span>
      </div>
      <div class="rule-text">
        <strong>IF</strong>&nbsp;&nbsp;&nbsp;&nbsp;${ifClause}<br>
        <strong>THEN</strong> patient has <span class="${diabetic ? 'term-high' : 'term-low'}">${conclusionLabel.toLowerCase()}</span> of diabetes
      </div>
      ${firingHtml}
      ${supportHtml}
    </div>
  `;
}
