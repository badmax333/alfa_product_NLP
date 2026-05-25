// ============================================================
// Shared state
// ============================================================
let config = null;
let activePresetId = null;

let classificationResult = null;   // результат /api/v1/predict
let clientFeatures = {};            // последние значения формы

let salesArgsConfig = null;         // interaction types + mock arguments
let selectedInteractionType = null; // "banner" | "push" | "voice"
let selectedArgument = null;        // выбранный mock аргумент

let selectedChannel = "digital";    // "digital" | "voice"
let selectedMethod = "llm";         // "llm" | "random"
let metricsResult = null;           // результат /api/v1/metrics/generate

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

// Человекочитаемое значение поля (берёт label из field_options если есть)
function displayValue(name, val) {
  if (val === undefined || val === null || val === "") return "—";
  const opts = config.field_options && config.field_options[name];
  if (opts) {
    const opt = opts.find((o) => String(o.value) === String(val));
    if (opt) return opt.label;
  }
  return String(val);
}

// HTML-строки для 8 редактируемых признаков клиента
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
    renderClassificationResult(classificationResult);
  } catch (err) {
    showClassifyError(err.message || "Ошибка классификации");
  } finally {
    btn.disabled = false;
    btn.textContent = "Классифицировать";
  }
});

// ============================================================
// TAB 2 — Sales argument (mock)
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
  document.querySelectorAll(".itype-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.itype === typeId);
  });
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

function selectMockArgument() {
  if (!salesArgsConfig || !selectedInteractionType) return;
  const mockArg = salesArgsConfig.mock_arguments.find(
    (a) => a.interaction_type === selectedInteractionType
  );
  if (!mockArg) {
    document.getElementById("argument-placeholder").textContent = "Нет примера для этого типа";
    return;
  }
  selectedArgument = mockArg;
  renderArgumentCard(mockArg);
  document.getElementById("to-metrics-bar").classList.remove("hidden");
  document.getElementById("tab-btn-metrics").classList.add("done");
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
// TAB 3 — Metrics generation
// ============================================================
function onEnterMetricsTab() {
  const hasAll = classificationResult && selectedArgument;
  document.getElementById("metrics-no-prev").classList.toggle("hidden", hasAll);
  document.getElementById("metrics-client-summary").classList.toggle("hidden", !hasAll);

  if (!hasAll) return;

  renderMetricsClientSummary();
  updateMetricsPromptPreview();
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

      <p class="section-label">Sales-аргумент</p>
      <span class="argument-channel-badge ${arg.channel === "digital" ? "badge-digital" : "badge-voice"}"
            style="margin-bottom:0.5rem;display:inline-flex">
        ${itype?.label || arg.interaction_type}
      </span>
      <p style="font-weight:600;margin:0.35rem 0 0.25rem;font-size:0.92rem">${arg.headline}</p>
      <div class="metrics-argument-preview">${arg.body}</div>
    </div>
  `;
}

function selectChannel(channel) {
  selectedChannel = channel;
  document.querySelectorAll(".channel-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.channel === channel);
  });
  updateMetricsPromptPreview();
}

function selectMethod(method) {
  selectedMethod = method;
  document.querySelectorAll(".method-btn").forEach((btn) => {
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
  const promptEl = document.getElementById("metrics-prompt-text");
  promptEl.textContent = "Загружаем промпт…";

  try {
    const res = await fetch("/api/v1/metrics/render-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: classificationResult,
        sales_argument: selectedArgument,
        channel: selectedChannel,
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

  const btn = document.getElementById("btn-generate-metrics");
  btn.disabled = true;
  btn.textContent = "Генерируем…";

  document.getElementById("metrics-placeholder").classList.remove("hidden");
  document.getElementById("metrics-placeholder").textContent =
    selectedMethod === "llm" ? "Отправляем запрос в Mistral…" : "Генерируем локально…";
  document.getElementById("metrics-content").classList.add("hidden");

  try {
    const payload = {
      classification: classificationResult,
      sales_argument: selectedArgument,
      channel: selectedChannel,
      client_features: clientFeatures,
      method: selectedMethod,
    };

    const res = await fetch("/api/v1/metrics/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }

    metricsResult = await res.json();

    // Обновляем промпт из rendered_prompt только для LLM-режима
    if (selectedMethod === "llm") {
      document.getElementById("metrics-prompt-text").textContent = metricsResult.rendered_prompt;
    }

    renderMetricsResult(metricsResult);
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

function renderMetricsResult(data) {
  document.getElementById("metrics-placeholder").classList.add("hidden");
  const box = document.getElementById("metrics-content");
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
// Init
// ============================================================
loadConfig().catch((err) => {
  document.getElementById("result-placeholder").innerHTML =
    `<div class="error-msg">${err.message}</div>`;
});
