let config = null;
let activePresetId = null;

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

function renderResult(data) {
  document.getElementById("result-placeholder").classList.add("hidden");
  const box = document.getElementById("result-content");
  box.classList.remove("hidden");

  const probs = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, p]) => {
      const pct = (p * 100).toFixed(1);
      return `<li>
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
    <p style="margin:0 0 0.5rem;font-size:1.1rem;font-weight:600">${data.class_description}</p>
    <p class="confidence">${(data.confidence * 100).toFixed(1)}%</p>

    <div class="product-box">
      <h3>Рекомендуемый якорный продукт</h3>
      <p>AME-${data.recommended_product.ame}: ${data.recommended_product.name}</p>
    </div>

    <h3 style="font-size:0.85rem;color:var(--muted);text-transform:uppercase">Вероятности по сегментам</h3>
    <ul class="prob-list">${probs}</ul>

    <h3 style="font-size:0.85rem;color:var(--muted);text-transform:uppercase">Top-5 SHAP — объяснение решения</h3>
    <table class="shap-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Признак</th>
          <th>Значение</th>
          <th>SHAP</th>
          <th>Направление</th>
        </tr>
      </thead>
      <tbody>${shapRows}</tbody>
    </table>
  `;
}

function showError(message) {
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
    const res = await fetch("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${res.status}`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(err.message || "Ошибка классификации");
  } finally {
    btn.disabled = false;
    btn.textContent = "Классифицировать";
  }
});

loadConfig().catch((err) => {
  showError(err.message);
});
