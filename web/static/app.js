// ============================================================
// Shared state
// ============================================================
let config = null;
let activePresetId = null;

let classificationResult = null;   // результат /api/v1/predict
let clientFeatures = {};            // последние значения формы

let salesArgsConfig = null;         // interaction types + examples
let selectedInteractionType = null; // "banner" | "push" | "voice"
let selectedArgument = null;        // сгенерированный sales-аргумент Stage 1

let selectedMethod = "llm";         // "llm" | "random" для Stage 1 метрик
let metricsResult = null;           // результат /api/v1/metrics/generate (Stage 1)
let propensityResult = null;        // результат /api/v1/propensity/score

// Stage 2 state
let selectedPropensityProduct = null; // выбранный продукт из top_products
let stage2InteractionType = null;     // "banner" | "push" | "voice" для Stage 2
let stage2Argument = null;            // сгенерированный sales-аргумент Stage 2
let selectedS2Method = "llm";         // "llm" | "random" для Stage 2 метрик
let stage2MetricsResult = null;       // результат генерации метрик Stage 2
let recycleInteractionType = "banner"; // формат следующего касания после двух попыток
let recycleResult = null;             // результат закольцованного цикла

// ============================================================
// Tab navigation
// ============================================================
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach((el) => el.classList.add("hidden"));
  document.querySelectorAll(".tab-btn").forEach((el) => el.classList.remove("active"));

  const tabEl = document.getElementById("tab-" + tabId);
  const btnEl = document.querySelector(`[data-tab="${tabId}"]`);
  if (tabEl) tabEl.classList.remove("hidden");
  if (btnEl) btnEl.classList.add("active");

  if (tabId === "sales") onEnterSalesTab();
  if (tabId === "metrics") onEnterMetricsTab();
  if (tabId === "propensity") onEnterPropensityTab();
  if (tabId === "stage2-sales") onEnterStage2SalesTab();
  if (tabId === "stage2-metrics") onEnterStage2MetricsTab();
  if (tabId === "recycle") onEnterRecycleTab();
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// ============================================================
// TAB 1 — Classification
// ============================================================
async function loadConfig() {
  const res = await fetch("/api/v1/config");
  if (!res.ok) throw new Error("Не удалось загрузить конфигурацию");
  config = await res.json();
  renderPresets();
  renderForm(config.default_overrides);
}

function renderPresets() {
  const container = document.getElementById("presets");
  container.innerHTML = "";
  config.presets.forEach((preset) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-btn";
    btn.dataset.presetId = preset.id;
    btn.innerHTML = `<strong>${preset.title}</strong><span>${preset.description}</span>`;
    btn.addEventListener("click", () => applyPreset(preset));
    container.appendChild(btn);
  });
}

function setActivePreset(presetId) {
  activePresetId = presetId;
  document.querySelectorAll(".preset-btn").forEach((el) => {
    el.classList.toggle("active", el.dataset.presetId === presetId);
  });
}

function applyPreset(preset) {
  setActivePreset(preset.id);
  const values = { ...config.default_overrides, ...preset.overrides };
  fillForm(values);
}

function renderForm(values) {
  const form = document.getElementById("predict-form");
  form.innerHTML = "";

  config.editable_features.forEach((name) => {
    const field = document.createElement("div");
    field.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", name);
    label.textContent = config.feature_labels[name] || name;
    field.appendChild(label);

    const options = config.field_options[name];
    let input;
    if (options && options.length) {
      input = document.createElement("select");
      options.forEach((opt) => {
        const o = document.createElement("option");
        o.value = opt.value;
        o.textContent = opt.label;
        input.appendChild(o);
      });
    } else {
      input = document.createElement("input");
      input.type =
        name === "days_from_ogrn" || name === "week_sum_transactions" ? "number" : "text";
    }
    input.id = name;
    input.name = name;
    input.value = values[name] ?? "";
    field.appendChild(input);
    form.appendChild(field);
  });
}

function fillForm(values) {
  config.editable_features.forEach((name) => {
    const el = document.getElementById(name);
    if (el && values[name] !== undefined) {
      el.value = values[name];
    }
  });
}

function collectPayload() {
  const payload = {};
  config.editable_features.forEach((name) => {
    const el = document.getElementById(name);
    if (!el) return;
    const raw = el.value;
    if (raw === "") return;
    if (el.type === "number") {
      payload[name] = Number(raw);
    } else {
      payload[name] = String(raw);
    }
  });
  return payload;
}

function displayValue(name, val) {
  if (val === undefined || val === null || val === "") return "—";
  const opts = config.field_options && config.field_options[name];
  if (opts) {
    const opt = opts.find((o) => String(o.value) === String(val));
    if (opt) return opt.label;
  }
  return String(val);
}

function renderClientFeaturesRows() {
  return config.editable_features
    .map((name) => {
      const val = clientFeatures[name];
      if (val === undefined || val === null || val === "") return "";
      const label = config.feature_labels[name] || name;
      return `<li>
        <span class="feat-name">${label}</span>
        <span class="feat-val">${displayValue(name, val)}</span>
      </li>`;
    })
    .filter(Boolean)
    .join("");
}

function renderClassificationResult(data) {
  document.getElementById("result-placeholder").classList.add("hidden");
  const box = document.getElementById("result-content");
  box.classList.remove("hidden");

  const probs = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, p]) => {
      const pct = (p * 100).toFixed(1);
      const classDesc = config.class_descriptions[cls] || cls;
      return `<li data-tooltip="${classDesc}">
        <span class="cls">${cls}</span>
        <span class="bar-wrap"><span class="bar" style="width:${pct}%"></span></span>
        <span class="prob-label">${pct}%</span>
      </li>`;
    })
    .join("");

  const shapRows = data.top5_feature_importance
    .map((item) => {
      const shapClass = item.shap >= 0 ? "shap-pos" : "shap-neg";
      return `<tr>
        <td>#${item.rank}</td>
        <td><code>${item.feature}</code></td>
        <td>${item.value}</td>
        <td class="${shapClass}">${item.shap >= 0 ? "+" : ""}${item.shap}</td>
        <td>${item.direction}</td>
      </tr>`;
    })
    .join("");

  box.innerHTML = `
    <span class="segment-badge">${data.predicted_class}</span>
    <p style="margin:0 0 0.5rem;font-size:1.05rem;font-weight:600">${data.class_description}</p>
    <p class="confidence">${(data.confidence * 100).toFixed(1)}%</p>

    <div class="product-box">
      <h3>Рекомендуемый якорный продукт</h3>
      <p>AME-${data.recommended_product.ame}: ${data.recommended_product.name}</p>
    </div>

    <h3 style="font-size:0.82rem;color:var(--muted);text-transform:uppercase;margin:0 0 0.5rem">Вероятности по сегментам</h3>
    <ul class="prob-list">${probs}</ul>

    <h3 style="font-size:0.82rem;color:var(--muted);text-transform:uppercase;margin:0 0 0.5rem">Top-5 SHAP — объяснение решения</h3>
    <table class="shap-table">
      <thead>
        <tr><th>#</th><th>Признак</th><th>Значение</th><th>SHAP</th><th>Направление</th></tr>
      </thead>
      <tbody>${shapRows}</tbody>
    </table>
  `;

  document.getElementById("to-sales-bar").classList.remove("hidden");
  document.getElementById("tab-btn-sales").classList.add("done");
}

function showClassifyError(message) {
  document.getElementById("result-placeholder").classList.add("hidden");
  const box = document.getElementById("result-content");
  box.classList.remove("hidden");
  box.innerHTML = `<div class="error-msg">${message}</div>`;
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  activePresetId = null;
  document.querySelectorAll(".preset-btn").forEach((el) => el.classList.remove("active"));

  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.textContent = "Считаем…";

  try {
    clientFeatures = collectPayload();
    const res = await fetch("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(clientFeatures),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }
    classificationResult = await res.json();
    selectedArgument = null;
    metricsResult = null;
    propensityResult = null;
    stage2Argument = null;
    stage2MetricsResult = null;
    recycleResult = null;
    selectedPropensityProduct = null;
    document.getElementById("tab-btn-metrics").classList.remove("done");
    document.getElementById("tab-btn-propensity").classList.remove("done");
    document.getElementById("tab-btn-stage2-sales").classList.remove("done");
    document.getElementById("tab-btn-stage2-metrics").classList.remove("done");
    document.getElementById("tab-btn-recycle").classList.remove("done");
    renderClassificationResult(classificationResult);
  } catch (err) {
    showClassifyError(err.message || "Ошибка классификации");
  } finally {
    btn.disabled = false;
    btn.textContent = "Классифицировать";
  }
});

// ============================================================
// TAB 2 — Sales argument (Stage 1)
// ============================================================
async function loadSalesArgsConfig() {
  if (salesArgsConfig) return;
  const res = await fetch("/api/v1/sales-args/config");
  if (!res.ok) throw new Error("Не удалось загрузить конфигурацию аргументов");
  salesArgsConfig = await res.json();
}

function onEnterSalesTab() {
  if (!classificationResult) {
    document.getElementById("sales-no-classify").classList.remove("hidden");
    document.getElementById("sales-client-info").classList.add("hidden");
    return;
  }
  document.getElementById("sales-no-classify").classList.add("hidden");
  document.getElementById("sales-client-info").classList.remove("hidden");
  renderSalesClientInfo();

  loadSalesArgsConfig().then(() => {
    renderInteractionTypeButtons();
    if (!selectedInteractionType) {
      selectInteractionType("banner");
    } else {
      updateSalesPrompt();
    }
  });
}

function renderSalesClientInfo() {
  const r = classificationResult;
  const top5 = r.top5_feature_importance || [];
  const shapRows = top5
    .map(
      (f) =>
        `<li>
          <span class="feat-name">${f.feature}</span>
          <span class="feat-val shap-${f.shap >= 0 ? "pos" : "neg"}">${f.value}</span>
        </li>`
    )
    .join("");

  document.getElementById("sales-client-info").innerHTML = `
    <div class="profile-summary">
      <p class="section-label">Ключевые признаки</p>
      <ul class="profile-features">${renderClientFeaturesRows()}</ul>

      <div class="divider"></div>

      <div class="profile-badge">
        <span class="profile-class">${r.predicted_class}</span>
        <span class="profile-name">${r.class_description}</span>
      </div>
      <div class="profile-product">AME-${r.recommended_product.ame}: ${r.recommended_product.name}</div>

      <p class="section-label">Top-5 SHAP — влияние признаков</p>
      <ul class="profile-features">${shapRows}</ul>
    </div>
  `;
}

function renderInteractionTypeButtons() {
  const container = document.getElementById("interaction-type-btns");
  container.innerHTML = "";
  salesArgsConfig.interaction_types.forEach((t) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "itype-btn" + (selectedInteractionType === t.id ? " active" : "");
    btn.dataset.itype = t.id;
    btn.innerHTML = `<strong>${t.label}</strong><span>${t.description}</span>`;
    btn.addEventListener("click", () => selectInteractionType(t.id));
    container.appendChild(btn);
  });
}

function selectInteractionType(typeId) {
  selectedInteractionType = typeId;
  selectedArgument = null;
  metricsResult = null;
  propensityResult = null;
  stage2Argument = null;
  stage2MetricsResult = null;
  recycleResult = null;
  selectedPropensityProduct = null;
  document.querySelectorAll("#interaction-type-btns .itype-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.itype === typeId);
  });
  document.getElementById("argument-content").classList.add("hidden");
  document.getElementById("to-metrics-bar").classList.add("hidden");
  document.getElementById("argument-placeholder").classList.remove("hidden");
  document.getElementById("argument-placeholder").textContent =
    "Нажмите «Сгенерировать аргумент», чтобы получить персонализированный текст";
  document.getElementById("tab-btn-metrics").classList.remove("done");
  document.getElementById("tab-btn-propensity").classList.remove("done");
  document.getElementById("tab-btn-stage2-sales").classList.remove("done");
  document.getElementById("tab-btn-stage2-metrics").classList.remove("done");
  document.getElementById("tab-btn-recycle").classList.remove("done");
  updateSalesPrompt();
}

async function updateSalesPrompt() {
  if (!classificationResult || !salesArgsConfig || !selectedInteractionType) return;
  const promptEl = document.getElementById("sales-prompt-text");
  promptEl.textContent = "Загружаем промпт…";

  try {
    const res = await fetch("/api/v1/sales-args/render-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        interaction_type: selectedInteractionType,
        client_features: clientFeatures,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    promptEl.textContent = data.rendered_prompt;
  } catch {
    promptEl.textContent = "Ошибка загрузки промпта";
  }
}

async function generateSalesArgument() {
  if (!classificationResult || !selectedInteractionType) return;

  const btn = document.getElementById("btn-get-argument");
  const placeholder = document.getElementById("argument-placeholder");
  btn.disabled = true;
  btn.textContent = "Генерируем…";
  placeholder.classList.remove("hidden");
  placeholder.textContent = "Отправляем запрос в Mistral…";
  document.getElementById("argument-content").classList.add("hidden");
  document.getElementById("to-metrics-bar").classList.add("hidden");

  try {
    const res = await fetch("/api/v1/sales-args/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        interaction_type: selectedInteractionType,
        client_features: clientFeatures,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }

    selectedArgument = await res.json();
    metricsResult = null;
    propensityResult = null;
    stage2Argument = null;
    stage2MetricsResult = null;
    recycleResult = null;
    renderArgumentCard(selectedArgument);

    if (selectedArgument.rendered_prompt) {
      document.getElementById("sales-prompt-text").textContent = selectedArgument.rendered_prompt;
    }

    document.getElementById("to-metrics-bar").classList.remove("hidden");
    document.getElementById("tab-btn-metrics").classList.add("done");
    document.getElementById("tab-btn-propensity").classList.remove("done");
    document.getElementById("tab-btn-stage2-sales").classList.remove("done");
    document.getElementById("tab-btn-stage2-metrics").classList.remove("done");
    document.getElementById("tab-btn-recycle").classList.remove("done");
  } catch (err) {
    placeholder.classList.remove("hidden");
    placeholder.innerHTML = `<div class="error-msg">${err.message || "Ошибка генерации аргумента"}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Сгенерировать аргумент";
  }
}

function renderArgumentCard(arg) {
  document.getElementById("argument-placeholder").classList.add("hidden");
  const box = document.getElementById("argument-content");
  box.classList.remove("hidden");

  const badgeClass = arg.channel === "digital" ? "badge-digital" : "badge-voice";
  const channelLabel = arg.channel === "digital" ? "Цифровой канал" : "Голосовой канал";
  const itypeLabel =
    salesArgsConfig.interaction_types.find((t) => t.id === arg.interaction_type)?.label ||
    arg.interaction_type;

  box.innerHTML = `
    <div class="argument-card">
      <div>
        <span class="argument-channel-badge ${badgeClass}">${channelLabel} · ${itypeLabel}</span>
      </div>
      <p class="argument-headline">${arg.headline}</p>
      <p class="argument-body">${arg.body}</p>
      ${arg.cta ? `<span class="argument-cta">${arg.cta}</span>` : ""}
      <div class="argument-note">
        <strong>Примечание к аргументу</strong>
        ${arg.note}
      </div>
    </div>
  `;
}

// ============================================================
// TAB 3 — Metrics generation (Stage 1)
// Канал определяется автоматически из interaction_type Stage 1
// ============================================================
function channelFromInteractionType(interactionType) {
  return interactionType === "voice" ? "voice" : "digital";
}

function onEnterMetricsTab() {
  const hasAll = classificationResult && selectedArgument;
  document.getElementById("metrics-no-prev").classList.toggle("hidden", hasAll);
  document.getElementById("metrics-client-summary").classList.toggle("hidden", !hasAll);

  if (!hasAll) return;

  const channel =
    selectedArgument.channel ||
    channelFromInteractionType(selectedArgument.interaction_type || selectedInteractionType);
  updateChannelDisplay("metrics-channel-info", channel);

  renderMetricsClientSummary();
  updateMetricsPromptPreview();
}

function updateChannelDisplay(elementId, channel) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const label = channel === "digital" ? "Цифровой канал" : "Голосовой канал";
  el.className = `channel-info ${channel}`;
  el.textContent = `Канал: ${label}`;
}

function renderMetricsClientSummary() {
  const r = classificationResult;
  const arg = selectedArgument;
  const itype = salesArgsConfig?.interaction_types.find((t) => t.id === arg.interaction_type);
  const top5 = r.top5_feature_importance || [];

  const shapRows = top5
    .map(
      (f) =>
        `<li>
          <span class="feat-name">${f.feature}</span>
          <span class="feat-val shap-${f.shap >= 0 ? "pos" : "neg"}">${f.value}</span>
        </li>`
    )
    .join("");

  document.getElementById("metrics-client-summary").innerHTML = `
    <div class="profile-summary">
      <p class="section-label">Ключевые признаки</p>
      <ul class="profile-features">${renderClientFeaturesRows()}</ul>

      <div class="divider"></div>

      <div class="profile-badge">
        <span class="profile-class">${r.predicted_class}</span>
        <span class="profile-name">${r.class_description}</span>
      </div>
      <div class="profile-product">AME-${r.recommended_product.ame}: ${r.recommended_product.name}</div>

      <p class="section-label">Top-5 SHAP — влияние признаков</p>
      <ul class="profile-features">${shapRows}</ul>

      <div class="divider"></div>

      <p class="section-label">Sales-аргумент Stage 1</p>
      <span class="argument-channel-badge ${arg.channel === "digital" ? "badge-digital" : "badge-voice"}"
            style="margin-bottom:0.5rem;display:inline-flex">
        ${itype?.label || arg.interaction_type}
      </span>
      <p style="font-weight:600;margin:0.35rem 0 0.25rem;font-size:0.92rem">${arg.headline}</p>
      <div class="metrics-argument-preview">${arg.body}</div>
    </div>
  `;
}

function selectMethod(method) {
  selectedMethod = method;
  document.querySelectorAll("#tab-metrics .method-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.method === method);
  });
  const generateBtn = document.getElementById("btn-generate-metrics");
  if (generateBtn) {
    generateBtn.textContent =
      method === "llm" ? "Сгенерировать метрики (LLM)" : "Сгенерировать (локально)";
  }
}

async function updateMetricsPromptPreview() {
  if (!classificationResult || !selectedArgument) return;
  const channel = selectedArgument.channel || "digital";
  const promptEl = document.getElementById("metrics-prompt-text");
  promptEl.textContent = "Загружаем промпт…";

  try {
    const res = await fetch("/api/v1/metrics/render-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        sales_argument: selectedArgument,
        channel: channel,
        client_features: clientFeatures,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    promptEl.textContent = data.rendered_prompt;
  } catch {
    promptEl.textContent = "Ошибка загрузки промпта";
  }
}

async function generateMetrics() {
  if (!classificationResult || !selectedArgument) {
    alert("Сначала выполните классификацию и выберите Sales-аргумент");
    return;
  }

  const channel = selectedArgument.channel || "digital";
  const btn = document.getElementById("btn-generate-metrics");
  btn.disabled = true;
  btn.textContent = "Генерируем…";

  document.getElementById("metrics-placeholder").classList.remove("hidden");
  document.getElementById("metrics-placeholder").textContent =
    selectedMethod === "llm" ? "Отправляем запрос в Mistral…" : "Генерируем локально…";
  document.getElementById("metrics-content").classList.add("hidden");
  document.getElementById("to-propensity-bar").classList.add("hidden");
  propensityResult = null;
  stage2Argument = null;
  stage2MetricsResult = null;
  recycleResult = null;
  document.getElementById("tab-btn-propensity").classList.remove("done");
  document.getElementById("tab-btn-stage2-sales").classList.remove("done");
  document.getElementById("tab-btn-stage2-metrics").classList.remove("done");
  document.getElementById("tab-btn-recycle").classList.remove("done");

  try {
    const res = await fetch("/api/v1/metrics/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        sales_argument: selectedArgument,
        channel: channel,
        client_features: clientFeatures,
        method: selectedMethod,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }

    metricsResult = await res.json();

    if (selectedMethod === "llm") {
      document.getElementById("metrics-prompt-text").textContent = metricsResult.rendered_prompt;
    }

    renderMetricsResult(metricsResult, "metrics-placeholder", "metrics-content");
    document.getElementById("to-propensity-bar").classList.remove("hidden");
  } catch (err) {
    document.getElementById("metrics-placeholder").classList.remove("hidden");
    document.getElementById("metrics-placeholder").innerHTML =
      `<div class="error-msg">${err.message || "Ошибка генерации метрик"}</div>`;
    document.getElementById("metrics-content").classList.add("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent =
      selectedMethod === "llm" ? "Сгенерировать метрики (LLM)" : "Сгенерировать (локально)";
  }
}

// Общая функция рендеринга результатов метрик (переиспользуется для Stage 1 и Stage 2)
function renderMetricsResult(data, placeholderId, boxId) {
  document.getElementById(placeholderId).classList.add("hidden");
  const box = document.getElementById(boxId);
  box.classList.remove("hidden");

  const scorePct = Math.round(data.interest_score * 100);

  const byLevel = {};
  (data.metrics || []).forEach((m) => {
    if (!byLevel[m.level]) byLevel[m.level] = { name: m.level_name, items: [] };
    byLevel[m.level].items.push(m);
  });

  const levelBlocks = Object.entries(byLevel)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([lvl, group]) => {
      const rows = group.items
        .map((m) => {
          const { cls, display } = formatMetricValue(m);
          return `<div class="metric-row">
            <span class="metric-label" title="${m.description}">${m.label}</span>
            <span class="metric-value ${cls}">${display}</span>
          </div>`;
        })
        .join("");
      return `
        <div class="metrics-level-group">
          <div class="metrics-level-title">
            <span class="level-num-badge">${lvl}</span>
            ${group.name}
          </div>
          ${rows}
        </div>`;
    })
    .join("");

  const methodBadge =
    data.raw_llm_response === ""
      ? `<span style="font-size:0.75rem;color:var(--muted);margin-left:0.5rem">(локально)</span>`
      : `<span style="font-size:0.75rem;color:var(--muted);margin-left:0.5rem">(Mistral LLM)</span>`;

  box.innerHTML = `
    <div class="interest-score-box">
      <div class="interest-score-value">${scorePct}%</div>
      <div class="interest-score-label">Интерес клиента<br>к предложению${methodBadge}</div>
    </div>

    <p class="section-label">Реакция клиента</p>
    <div class="reaction-text">${data.user_reaction_text}</div>

    <p class="section-label">Метрики по уровням</p>
    ${levelBlocks}
  `;
}

function formatMetricValue(m) {
  const v = m.value;
  if (v === null || v === undefined) return { cls: "val-no", display: "—" };

  if (m.type === "binary") {
    return v === 1 || v === true
      ? { cls: "val-yes", display: "Да ✓" }
      : { cls: "val-no", display: "Нет" };
  }
  if (m.type === "duration_sec") {
    return { cls: "", display: `${v} сек` };
  }
  if (m.type === "float") {
    return { cls: "", display: `${v} ${m.unit}` };
  }
  if (m.type === "integer") {
    const warn = m.level === 5 && v > 0;
    return { cls: warn ? "val-warn" : "", display: `${v} ${m.unit}` };
  }
  return { cls: "", display: String(v) };
}

// ============================================================
// TAB 4 — Product propensity scoring
// ============================================================
function onEnterPropensityTab() {
  const hasMetrics = classificationResult && selectedArgument && metricsResult;
  document.getElementById("propensity-no-prev").classList.toggle("hidden", hasMetrics);
  document.getElementById("propensity-client-summary").classList.toggle("hidden", !hasMetrics);

  if (!hasMetrics) return;

  renderPropensityContext();
  updatePropensityFeaturePrompt();
  if (propensityResult) {
    renderPropensityResult(propensityResult);
    document.getElementById("to-stage2-bar").classList.remove("hidden");
    document.getElementById("tab-btn-stage2-sales").classList.add("done");
  }
}

function renderPropensityContext() {
  const r = classificationResult;
  const arg = selectedArgument;
  const interestPct = metricsResult ? Math.round(metricsResult.interest_score * 100) : 0;

  document.getElementById("propensity-client-summary").innerHTML = `
    <div class="profile-summary">
      <p class="section-label">Портрет</p>
      <div class="profile-badge">
        <span class="profile-class">${r.predicted_class}</span>
        <span class="profile-name">${r.class_description}</span>
      </div>

      <p class="section-label">Реакция на Stage 1</p>
      <div class="interest-score-box compact">
        <div class="interest-score-value">${interestPct}%</div>
        <div class="interest-score-label">интерес к аргументу Stage 1</div>
      </div>

      <p class="section-label">Sales-аргумент Stage 1</p>
      <p style="font-weight:600;margin:0.35rem 0 0.25rem;font-size:0.92rem">${arg.headline}</p>
      <div class="metrics-argument-preview">${arg.body}</div>

      <p class="section-label">Ключевые признаки</p>
      <ul class="profile-features">${renderClientFeaturesRows()}</ul>
    </div>
  `;
}

function propensityRequestPayload(topK = 3) {
  return {
    classification: classificationResult,
    client_features: clientFeatures,
    metrics_result: metricsResult,
    sales_argument: selectedArgument,
    top_k: topK,
  };
}

async function updatePropensityFeaturePrompt() {
  if (!classificationResult || !metricsResult) return;
  const promptEl = document.getElementById("propensity-feature-prompt-text");
  promptEl.textContent = "Собираем промпт генерации фичей…";

  try {
    const res = await fetch("/api/v1/propensity/render-feature-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(propensityRequestPayload()),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    promptEl.textContent = data.rendered_prompt;
  } catch {
    promptEl.textContent = "Ошибка загрузки промпта генерации признаков";
  }
}

async function scorePropensity() {
  if (!classificationResult || !metricsResult) {
    alert("Сначала рассчитайте метрики взаимодействия");
    return;
  }

  const btn = document.getElementById("btn-score-propensity");
  const placeholder = document.getElementById("propensity-placeholder");
  btn.disabled = true;
  btn.textContent = "Генерируем фичи…";
  placeholder.classList.remove("hidden");
  placeholder.textContent = "Mistral генерирует признаки клиента для модели склонности…";
  document.getElementById("propensity-content").classList.add("hidden");
  document.getElementById("to-stage2-bar").classList.add("hidden");
  stage2Argument = null;
  stage2MetricsResult = null;
  recycleResult = null;
  selectedPropensityProduct = null;
  document.getElementById("tab-btn-stage2-sales").classList.remove("done");
  document.getElementById("tab-btn-stage2-metrics").classList.remove("done");
  document.getElementById("tab-btn-recycle").classList.remove("done");

  try {
    const res = await fetch("/api/v1/propensity/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(propensityRequestPayload(3)),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }

    propensityResult = await res.json();
    document.getElementById("propensity-feature-prompt-text").textContent =
      propensityResult.feature_generation_prompt || "";
    renderPropensityResult(propensityResult);
    document.getElementById("tab-btn-propensity").classList.add("done");
    document.getElementById("to-stage2-bar").classList.remove("hidden");
    document.getElementById("tab-btn-stage2-sales").classList.add("done");
  } catch (err) {
    placeholder.classList.remove("hidden");
    placeholder.innerHTML = `<div class="error-msg">${err.message || "Ошибка скоринга склонности"}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Сгенерировать фичи и рассчитать склонность";
  }
}

function renderGeneratedFeatures(features) {
  const priority = [
    "priority_segment",
    "week_sum_transactions",
    "week_mean_transactions",
    "share_last_month",
    "share_last_3_months",
    "srvpackage_sale_uk",
    "sourceattr_ccode",
    "acquiring_num_live",
    "zpp_num_live",
    "rko_num_live",
    "abm_entered",
    "mobile_app_entered",
    "plastic_card_issued",
    "cashback_selected",
  ];
  const rows = priority
    .filter((name) => features && features[name] !== undefined)
    .map(
      (name) => `
        <li>
          <span class="feat-name">${config.feature_labels[name] || name}</span>
          <span class="feat-val">${displayValue(name, features[name])}</span>
        </li>`
    )
    .join("");
  return `<ul class="profile-features generated-features">${rows}</ul>`;
}

function renderPropensityResult(data) {
  document.getElementById("propensity-placeholder").classList.add("hidden");
  const box = document.getElementById("propensity-content");
  box.classList.remove("hidden");

  const cards = (data.top_products || [])
    .map((p) => {
      const scorePct = Math.round(p.propensity_score * 100);
      const factors = (p.top_factors || [])
        .map(
          (f) => `
            <li>
              <span>
                <strong>${f.label}</strong>
                <small>${f.reason}</small>
              </span>
              <span class="${f.direction === "increases" ? "shap-pos" : "shap-neg"}">
                ${f.impact > 0 ? "+" : ""}${f.impact}
              </span>
            </li>`
        )
        .join("");

      const ame = p.product_ame ? `AME-${p.product_ame}` : "без AME";
      const anchor = p.anchor ? "Якорный" : "Дополнительный";

      return `
        <div class="propensity-card">
          <div class="propensity-card-head">
            <span class="rank-badge">#${p.rank}</span>
            <div>
              <h3>${p.product_name}</h3>
              <p>${ame} · ${anchor}</p>
            </div>
            <div class="propensity-score">${scorePct}%</div>
          </div>
          <div class="propensity-score-bar">
            <span style="width:${scorePct}%"></span>
          </div>
          <p class="propensity-description">${p.description}</p>
          <p class="section-label">Факторы скоринга</p>
          <ul class="propensity-factors">${factors}</ul>
        </div>`;
    })
    .join("");

  const sourceLabels = {
    lightgbm_propensity_lgbm: "LightGBM propensity_lgbm.pkl",
    rule_based_propensity_fallback: "Fallback-скорер из логики Николая",
  };
  const sourceLabel = sourceLabels[data.model_source] || data.model_source;

  box.innerHTML = `
    <div class="model-source-note">
      Источник: ${sourceLabel}<br>
      Фичи: ${data.feature_source === "llm_generated_features" ? "сгенерированы Mistral перед скорингом" : data.feature_source}
    </div>
    <div class="generated-feature-box">
      <p class="section-label">Сгенерированные признаки для модели</p>
      ${data.feature_generation_reasoning ? `<p class="hint">${data.feature_generation_reasoning}</p>` : ""}
      ${renderGeneratedFeatures(data.generated_features || {})}
    </div>
    ${cards}
  `;
}

// ============================================================
// TAB 5 — Stage 2 Sales Argument
// ============================================================
function onEnterStage2SalesTab() {
  const hasPropensity = classificationResult && selectedArgument && metricsResult && propensityResult;
  document.getElementById("s2sales-no-prev").classList.toggle("hidden", hasPropensity);
  document.getElementById("s2sales-context").classList.toggle("hidden", !hasPropensity);

  if (!hasPropensity) return;

  renderS2Context();

  loadSalesArgsConfig().then(() => {
    renderS2ProductButtons();
    renderS2InteractionTypeButtons();
    if (selectedPropensityProduct && stage2InteractionType) {
      updateStage2Prompt();
    }
  });
}

function renderS2Context() {
  const r = classificationResult;
  const arg = selectedArgument;
  const interestPct = metricsResult ? Math.round(metricsResult.interest_score * 100) : 0;

  document.getElementById("s2sales-context").innerHTML = `
    <div class="profile-summary">
      <p class="section-label">Портрет клиента</p>
      <div class="profile-badge">
        <span class="profile-class">${r.predicted_class}</span>
        <span class="profile-name">${r.class_description}</span>
      </div>

      <p class="section-label">Реакция на Stage 1</p>
      <div class="interest-score-box compact">
        <div class="interest-score-value">${interestPct}%</div>
        <div class="interest-score-label">${metricsResult?.user_reaction_text || "интерес к аргументу"}</div>
      </div>

      <p class="section-label">Аргумент Stage 1</p>
      <p style="font-weight:600;margin:0.35rem 0 0.25rem;font-size:0.92rem">${arg.headline}</p>
      <div class="metrics-argument-preview">${arg.body}</div>

      <p class="section-label">Ключевые признаки</p>
      <ul class="profile-features">${renderClientFeaturesRows()}</ul>
    </div>
  `;
}

function renderS2ProductButtons() {
  const container = document.getElementById("s2-product-btns");
  container.innerHTML = "";
  const products = propensityResult?.top_products || [];
  products.forEach((p) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      "itype-btn" + (selectedPropensityProduct?.product_id === p.product_id ? " active" : "");
    btn.dataset.productId = p.product_id;
    const scorePct = Math.round(p.propensity_score * 100);
    btn.innerHTML = `<strong>#${p.rank} ${p.product_name}</strong><span>Склонность: ${scorePct}%</span>`;
    btn.addEventListener("click", () => selectS2Product(p));
    container.appendChild(btn);
  });
}

function selectS2Product(product) {
  selectedPropensityProduct = product;
  stage2Argument = null;
  stage2MetricsResult = null;
  recycleResult = null;
  document.querySelectorAll("#s2-product-btns .itype-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.productId === product.product_id);
  });
  document.getElementById("s2-argument-content").classList.add("hidden");
  document.getElementById("to-s2-metrics-bar").classList.add("hidden");
  document.getElementById("s2-argument-placeholder").classList.remove("hidden");
  document.getElementById("s2-argument-placeholder").textContent =
    "Выберите тип взаимодействия и нажмите «Сгенерировать»";
  document.getElementById("tab-btn-stage2-metrics").classList.remove("done");
  document.getElementById("tab-btn-recycle").classList.remove("done");
  if (stage2InteractionType) updateStage2Prompt();
}

function renderS2InteractionTypeButtons() {
  const container = document.getElementById("s2-interaction-type-btns");
  container.innerHTML = "";
  (salesArgsConfig?.interaction_types || []).forEach((t) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "itype-btn" + (stage2InteractionType === t.id ? " active" : "");
    btn.dataset.itype = t.id;
    btn.innerHTML = `<strong>${t.label}</strong><span>${t.description}</span>`;
    btn.addEventListener("click", () => selectS2InteractionType(t.id));
    container.appendChild(btn);
  });
}

function selectS2InteractionType(typeId) {
  stage2InteractionType = typeId;
  stage2Argument = null;
  stage2MetricsResult = null;
  recycleResult = null;
  document.querySelectorAll("#s2-interaction-type-btns .itype-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.itype === typeId);
  });
  document.getElementById("s2-argument-content").classList.add("hidden");
  document.getElementById("to-s2-metrics-bar").classList.add("hidden");
  document.getElementById("s2-argument-placeholder").classList.remove("hidden");
  document.getElementById("s2-argument-placeholder").textContent =
    "Выберите продукт и нажмите «Сгенерировать»";
  document.getElementById("tab-btn-stage2-metrics").classList.remove("done");
  document.getElementById("tab-btn-recycle").classList.remove("done");
  if (selectedPropensityProduct) updateStage2Prompt();
}

async function updateStage2Prompt() {
  if (!classificationResult || !selectedPropensityProduct || !stage2InteractionType) return;
  const promptEl = document.getElementById("s2-sales-prompt-text");
  promptEl.textContent = "Загружаем промпт…";

  try {
    const res = await fetch("/api/v1/sales-args/render-prompt-stage2", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        interaction_type: stage2InteractionType,
        client_features: clientFeatures,
        propensity_product: selectedPropensityProduct,
        stage1_argument: selectedArgument,
        stage1_metrics: metricsResult,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    promptEl.textContent = data.rendered_prompt;
  } catch {
    promptEl.textContent = "Ошибка загрузки промпта Stage 2";
  }
}

async function generateStage2Argument() {
  if (!classificationResult || !selectedPropensityProduct || !stage2InteractionType) {
    alert("Выберите продукт и тип взаимодействия");
    return;
  }

  const btn = document.getElementById("btn-generate-s2-argument");
  const placeholder = document.getElementById("s2-argument-placeholder");
  btn.disabled = true;
  btn.textContent = "Генерируем…";
  placeholder.classList.remove("hidden");
  placeholder.textContent = "Отправляем запрос в Mistral…";
  document.getElementById("s2-argument-content").classList.add("hidden");
  document.getElementById("to-s2-metrics-bar").classList.add("hidden");

  try {
    const res = await fetch("/api/v1/sales-args/generate-stage2", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        interaction_type: stage2InteractionType,
        client_features: clientFeatures,
        propensity_product: selectedPropensityProduct,
        stage1_argument: selectedArgument,
        stage1_metrics: metricsResult,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }

    stage2Argument = await res.json();
    stage2MetricsResult = null;
    recycleResult = null;

    if (stage2Argument.rendered_prompt) {
      document.getElementById("s2-sales-prompt-text").textContent = stage2Argument.rendered_prompt;
    }

    renderStage2ArgumentCard(stage2Argument);
    document.getElementById("to-s2-metrics-bar").classList.remove("hidden");
    document.getElementById("tab-btn-stage2-metrics").classList.add("done");
  } catch (err) {
    placeholder.classList.remove("hidden");
    placeholder.innerHTML = `<div class="error-msg">${err.message || "Ошибка генерации Stage 2 аргумента"}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Сгенерировать аргумент Stage 2";
  }
}

function renderStage2ArgumentCard(arg) {
  document.getElementById("s2-argument-placeholder").classList.add("hidden");
  const box = document.getElementById("s2-argument-content");
  box.classList.remove("hidden");

  const badgeClass = arg.channel === "digital" ? "badge-digital" : "badge-voice";
  const channelLabel = arg.channel === "digital" ? "Цифровой канал" : "Голосовой канал";
  const itypeLabel =
    salesArgsConfig.interaction_types.find((t) => t.id === arg.interaction_type)?.label ||
    arg.interaction_type;

  const propScorePct =
    arg.propensity_score != null
      ? `<p style="font-size:0.82rem;color:var(--muted);margin:0 0 0.5rem">Склонность: <strong style="color:var(--alfa-red)">${Math.round(arg.propensity_score * 100)}%</strong></p>`
      : "";

  box.innerHTML = `
    <div class="argument-card">
      <div>
        <span class="argument-channel-badge ${badgeClass}">${channelLabel} · ${itypeLabel}</span>
      </div>
      ${propScorePct}
      <p class="argument-headline">${arg.headline}</p>
      <p class="argument-body">${arg.body}</p>
      ${arg.cta ? `<span class="argument-cta">${arg.cta}</span>` : ""}
      <div class="argument-note">
        <strong>Примечание к аргументу</strong>
        ${arg.note}
      </div>
    </div>
  `;
}

// ============================================================
// TAB 6 — Stage 2 Metrics
// Канал определяется автоматически из interaction_type Stage 2
// ============================================================
function onEnterStage2MetricsTab() {
  const hasS2Arg = classificationResult && stage2Argument;
  document.getElementById("s2metrics-no-prev").classList.toggle("hidden", hasS2Arg);
  document.getElementById("s2metrics-client-summary").classList.toggle("hidden", !hasS2Arg);

  if (!hasS2Arg) return;

  const s2Channel =
    stage2Argument.channel ||
    channelFromInteractionType(stage2Argument.interaction_type || stage2InteractionType);
  updateChannelDisplay("s2metrics-channel-info", s2Channel);

  renderS2MetricsClientSummary();
  updateS2MetricsPromptPreview();
}

function renderS2MetricsClientSummary() {
  const r = classificationResult;
  const arg = stage2Argument;
  const itype = salesArgsConfig?.interaction_types.find((t) => t.id === arg.interaction_type);

  document.getElementById("s2metrics-client-summary").innerHTML = `
    <div class="profile-summary">
      <p class="section-label">Портрет</p>
      <div class="profile-badge">
        <span class="profile-class">${r.predicted_class}</span>
        <span class="profile-name">${r.class_description}</span>
      </div>

      <p class="section-label">Продукт Stage 2</p>
      <div class="profile-product">${arg.product_name}${arg.product_ame ? ` (AME-${arg.product_ame})` : ""}</div>

      <p class="section-label">Sales-аргумент Stage 2</p>
      <span class="argument-channel-badge ${arg.channel === "digital" ? "badge-digital" : "badge-voice"}"
            style="margin-bottom:0.5rem;display:inline-flex">
        ${itype?.label || arg.interaction_type}
      </span>
      <p style="font-weight:600;margin:0.35rem 0 0.25rem;font-size:0.92rem">${arg.headline}</p>
      <div class="metrics-argument-preview">${arg.body}</div>
    </div>
  `;
}

function selectS2Method(method) {
  selectedS2Method = method;
  document.querySelectorAll("#tab-stage2-metrics .method-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.method === method);
  });
  const generateBtn = document.getElementById("btn-generate-s2-metrics");
  if (generateBtn) {
    generateBtn.textContent =
      method === "llm" ? "Сгенерировать метрики Stage 2 (LLM)" : "Сгенерировать (локально)";
  }
}

async function updateS2MetricsPromptPreview() {
  if (!classificationResult || !stage2Argument) return;
  const channel = stage2Argument.channel || "digital";
  const promptEl = document.getElementById("s2-metrics-prompt-text");
  promptEl.textContent = "Загружаем промпт…";

  try {
    const res = await fetch("/api/v1/metrics/render-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        sales_argument: stage2Argument,
        channel: channel,
        client_features: clientFeatures,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    promptEl.textContent = data.rendered_prompt;
  } catch {
    promptEl.textContent = "Ошибка загрузки промпта";
  }
}

async function generateStage2Metrics() {
  if (!classificationResult || !stage2Argument) {
    alert("Сначала сгенерируйте Stage 2 аргумент");
    return;
  }

  const channel = stage2Argument.channel || "digital";
  const btn = document.getElementById("btn-generate-s2-metrics");
  btn.disabled = true;
  btn.textContent = "Генерируем…";

  document.getElementById("s2-metrics-placeholder").classList.remove("hidden");
  document.getElementById("s2-metrics-placeholder").textContent =
    selectedS2Method === "llm" ? "Отправляем запрос в Mistral…" : "Генерируем локально…";
  document.getElementById("s2-metrics-content").classList.add("hidden");

  try {
    const res = await fetch("/api/v1/metrics/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        sales_argument: stage2Argument,
        channel: channel,
        client_features: clientFeatures,
        method: selectedS2Method,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }

    stage2MetricsResult = await res.json();

    if (selectedS2Method === "llm") {
      document.getElementById("s2-metrics-prompt-text").textContent =
        stage2MetricsResult.rendered_prompt;
    }

    renderMetricsResult(stage2MetricsResult, "s2-metrics-placeholder", "s2-metrics-content");
    document.getElementById("tab-btn-stage2-metrics").classList.add("done");
    document.getElementById("to-recycle-bar").classList.remove("hidden");
  } catch (err) {
    document.getElementById("s2-metrics-placeholder").classList.remove("hidden");
    document.getElementById("s2-metrics-placeholder").innerHTML =
      `<div class="error-msg">${err.message || "Ошибка генерации метрик Stage 2"}</div>`;
    document.getElementById("s2-metrics-content").classList.add("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent =
      selectedS2Method === "llm" ? "Сгенерировать метрики Stage 2 (LLM)" : "Сгенерировать (локально)";
  }
}

// ============================================================
// TAB 7 — Recycle onboarding loop
// ============================================================
function onEnterRecycleTab() {
  const hasFullHistory =
    classificationResult &&
    selectedArgument &&
    metricsResult &&
    propensityResult &&
    stage2Argument &&
    stage2MetricsResult;
  document.getElementById("recycle-no-prev").classList.toggle("hidden", hasFullHistory);
  document.getElementById("recycle-context").classList.toggle("hidden", !hasFullHistory);

  if (!hasFullHistory) return;

  renderRecycleContext();
  loadSalesArgsConfig().then(() => renderRecycleInteractionTypeButtons());
  if (recycleResult) {
    renderRecycleResult(recycleResult);
  }
}

function renderRecycleContext() {
  const s1Interest = Math.round((metricsResult?.interest_score || 0) * 100);
  const s2Interest = Math.round((stage2MetricsResult?.interest_score || 0) * 100);
  const s2Product = selectedPropensityProduct || {};

  document.getElementById("recycle-context").innerHTML = `
    <div class="profile-summary">
      <p class="section-label">Статус</p>
      <div class="model-source-note">
        Две попытки продажи не активировали клиента. Новый цикл учитывает всю историю и избегает повторов.
      </div>

      <p class="section-label">Сегмент</p>
      <div class="profile-badge">
        <span class="profile-class">${classificationResult.predicted_class}</span>
        <span class="profile-name">${classificationResult.class_description}</span>
      </div>

      <p class="section-label">Попытка 1</p>
      <div class="metrics-argument-preview">
        <strong>${selectedArgument.product_name || "Stage 1 продукт"}</strong><br>
        ${selectedArgument.headline}<br>
        Интерес: ${s1Interest}% · ${metricsResult?.user_reaction_text || ""}
      </div>

      <p class="section-label">Попытка 2</p>
      <div class="metrics-argument-preview">
        <strong>${stage2Argument.product_name || s2Product.product_name || "Stage 2 продукт"}</strong><br>
        ${stage2Argument.headline}<br>
        Интерес: ${s2Interest}% · ${stage2MetricsResult?.user_reaction_text || ""}
      </div>
    </div>
  `;
}

function renderRecycleInteractionTypeButtons() {
  const container = document.getElementById("recycle-interaction-type-btns");
  container.innerHTML = "";
  (salesArgsConfig?.interaction_types || []).forEach((t) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "itype-btn" + (recycleInteractionType === t.id ? " active" : "");
    btn.dataset.itype = t.id;
    btn.innerHTML = `<strong>${t.label}</strong><span>${t.description}</span>`;
    btn.addEventListener("click", () => selectRecycleInteractionType(t.id));
    container.appendChild(btn);
  });
}

function selectRecycleInteractionType(typeId) {
  recycleInteractionType = typeId;
  recycleResult = null;
  document.querySelectorAll("#recycle-interaction-type-btns .itype-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.itype === typeId);
  });
  document.getElementById("recycle-content").classList.add("hidden");
  document.getElementById("recycle-placeholder").classList.remove("hidden");
  document.getElementById("recycle-placeholder").textContent = "Нажмите «Запустить новый цикл»";
  document.getElementById("recycle-prompt-text").textContent = "Промпт появится после запуска нового цикла";
}

function recyclePayload() {
  return {
    classification: classificationResult,
    client_features: clientFeatures,
    stage1_argument: selectedArgument,
    stage1_metrics: metricsResult,
    propensity_result: propensityResult,
    stage2_argument: stage2Argument,
    stage2_metrics: stage2MetricsResult,
    selected_stage2_product: selectedPropensityProduct || {},
    interaction_type: recycleInteractionType,
    top_k: 3,
  };
}

async function generateRecycleCycle() {
  if (!classificationResult || !stage2MetricsResult) {
    alert("Сначала завершите Stage 2 и рассчитайте метрики второго касания");
    return;
  }

  const btn = document.getElementById("btn-generate-recycle");
  const placeholder = document.getElementById("recycle-placeholder");
  btn.disabled = true;
  btn.textContent = "Запускаем цикл…";
  placeholder.classList.remove("hidden");
  placeholder.textContent = "Обновляем фичи, пересчитываем склонность и генерируем следующий оффер…";
  document.getElementById("recycle-content").classList.add("hidden");

  try {
    const res = await fetch("/api/v1/recycle/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(recyclePayload()),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }

    recycleResult = await res.json();
    document.getElementById("recycle-prompt-text").textContent = recycleResult.rendered_prompt || "";
    renderRecycleResult(recycleResult);
    document.getElementById("tab-btn-recycle").classList.add("done");
  } catch (err) {
    placeholder.classList.remove("hidden");
    placeholder.innerHTML = `<div class="error-msg">${err.message || "Ошибка нового цикла"}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Запустить новый цикл";
  }
}

function renderRecycleResult(data) {
  document.getElementById("recycle-placeholder").classList.add("hidden");
  const box = document.getElementById("recycle-content");
  box.classList.remove("hidden");

  const arg = data.next_argument;
  const product = data.selected_product;
  const scorePct = Math.round((product.propensity_score || 0) * 100);
  const shown = (data.shown_product_ids || []).join(", ") || "нет";

  box.innerHTML = `
    <div class="model-source-note">
      Новый цикл #${data.cycle_number}. Исключены уже показанные продукты: ${shown}
    </div>

    <div class="generated-feature-box">
      <p class="section-label">Обновленные признаки</p>
      ${data.feature_generation_reasoning ? `<p class="hint">${data.feature_generation_reasoning}</p>` : ""}
      ${renderGeneratedFeatures(data.generated_features || {})}
    </div>

    <div class="propensity-card">
      <div class="propensity-card-head">
        <span class="rank-badge">→</span>
        <div>
          <h3>${product.product_name}</h3>
          <p>Склонность после обновления: ${scorePct}%</p>
        </div>
        <div class="propensity-score">${scorePct}%</div>
      </div>
      <div class="propensity-score-bar">
        <span style="width:${scorePct}%"></span>
      </div>
      <p class="propensity-description">${product.description}</p>
    </div>

    <div class="argument-card">
      <div>
        <span class="argument-channel-badge ${arg.channel === "digital" ? "badge-digital" : "badge-voice"}">
          ${arg.channel === "digital" ? "Цифровой канал" : "Голосовой канал"} · ${arg.interaction_type}
        </span>
      </div>
      <p class="argument-headline">${arg.headline}</p>
      <p class="argument-body">${arg.body}</p>
      ${arg.cta ? `<span class="argument-cta">${arg.cta}</span>` : ""}
      <div class="argument-note">
        <strong>Стратегия нового цикла</strong>
        ${arg.note}
      </div>
    </div>
  `;
}

// ============================================================
// Init
// ============================================================
loadConfig().catch((err) => {
  document.getElementById("result-placeholder").innerHTML =
    `<div class="error-msg">${err.message}</div>`;
});
