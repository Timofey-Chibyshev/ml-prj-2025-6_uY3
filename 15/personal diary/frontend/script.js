// ==== КОНФИГУРАЦИЯ API =====
const API_URL = 'http://localhost:8000';

// ==== НАСТРОЙКА SUPABASE ====
var SUPABASE_URL = window.SUPABASE_URL || 'https://vlrimfflnucwkbtisgww.supabase.co';
var SUPABASE_ANON_KEY = window.SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZscmltZmZsbnVjd2tidGlzZ3d3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUyOTQ3MTUsImV4cCI6MjA4MDg3MDcxNX0.h6TwwXNP1pzBKKeS1iGptba_7kDkmPmTMVhLYm4ILjQ';

var supabase = window.supabaseClient || window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
window.supabaseClient = supabase;

// ==== ЭМОЦИИ И ШКАЛА ====
const emotionEmojis = {
  happiness: "😄",
  love: "❤️",
  pleasure: "😋",
  enthusiasm: "🔥",
  relief: "😌",
  surprise: "😲",
  calmness: "😇",
  boredom: "😑",
  worry: "😰",
  sadness: "😢",
  emptiness: "🌑",
  hatred: "😠",
  anger: "😡",
};

const apiToLocalEmotion = {
  "счастье": "happiness", "любовь": "love", "удовольствие": "pleasure",
  "энтузиазм": "enthusiasm", "облегчение": "relief", "удивление": "surprise",
  "спокойствие": "calmness", "скука": "boredom", "беспокойство": "worry",
  "грусть": "sadness", "пустота": "emptiness", "ненависть": "hatred", "злость": "anger",
};

const emotionRussian = {
  happiness: "Счастье",
  love: "Любовь",
  pleasure: "Удовольствие",
  enthusiasm: "Энтузиазм",
  relief: "Облегчение",
  surprise: "Удивление",
  calmness: "Спокойствие",
  boredom: "Скука",
  worry: "Беспокойство",
  sadness: "Грусть",
  emptiness: "Пустота",
  hatred: "Ненависть",
  anger: "Злость",
};

// русские метки, как они используются в модели / label2id
const emotionRussianModel = {
  happiness: "счастье",
  love: "любовь",
  pleasure: "удовольствие",
  enthusiasm: "энтузиазм",
  relief: "облегчение",
  surprise: "удивление",
  calmness: "спокойствие",
  boredom: "скука",
  worry: "беспокойство",
  sadness: "грусть",
  emptiness: "пустота",
  hatred: "ненависть",
  anger: "злость",
};

const emotionScaleValues = {
  happiness: 12,
  love: 11,
  pleasure: 10,
  enthusiasm: 9,
  relief: 8,
  surprise: 7,
  calmness: 6,
  boredom: 5,
  worry: 4,
  sadness: 3,
  emptiness: 2,
  hatred: 1,
  anger: 0,
};

const emotionColors = {
  happiness: "#FFD700",
  love: "#FF1493",
  pleasure: "#FF8C00",
  enthusiasm: "#FF6B00",
  relief: "#90EE90",
  surprise: "#FFD700",
  calmness: "#87CEEB",
  boredom: "#A9A9A9",
  worry: "#FFB6C1",
  sadness: "#4169E1",
  emptiness: "#4A4A4A",
  hatred: "#8B0000",
  anger: "#FF0000",
};

let entries = [];
let selectedEmotion = null;
let currentPredictionFromAPI = null;
let chart = null;
let currentPage = 1;
let filteredEntries = [];
let useAIModel = true;

// ==== ЗАГРУЗКА / СОХРАНЕНИЕ ЧЕРЕЗ SUPABASE ====
async function loadData() {
  const { data, error } = await supabase
    .from("entries")
    .select("*")
    .order("date", { ascending: true });

  if (error) {
    console.error("Ошибка загрузки:", error);
    showError("Не удалось загрузить записи с сервера");
    return;
  }

  entries = data || [];
  updateUI();
}

async function saveEntryRemote(entry) {
  const { data, error } = await supabase.from("entries").insert(entry).select();

  if (error) {
    console.error("Ошибка сохранения:", error);
    showError("Не удалось сохранить запись на сервер");
    return null;
  }

  return data[0];
}

async function deleteEntryRemote(id) {
  const { error } = await supabase.from("entries").delete().eq("id", id);

  if (error) {
    console.error("Ошибка удаления:", error);
    showError("Не удалось удалить запись на сервере");
    return false;
  }
  return true;
}

// ==== ПРЕДСКАЗАНИЕ С API (АДАПТИВНАЯ МОДЕЛЬ) ====
async function predictEmotionFromAPI(text) {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      console.warn("API недоступен, используем локальное предсказание");
      useAIModel = false;
      return predictEmotionLocal(text);
    }

    const data = await response.json();
    useAIModel = true;

    const localEmotion = apiToLocalEmotion[data.emotion.toLowerCase()] || "calmness";

    return {
      emotion: localEmotion,
      confidence: Math.round(data.confidence * 100),
      apiResponse: data,
    };
  } catch (error) {
    console.warn("Ошибка при обращении к API:", error);
    useAIModel = false;
    return predictEmotionLocal(text);
  }
}

// ==== ЛОКАЛЬНОЕ ПРЕДСКАЗАНИЕ (FALLBACK) ====
function predictEmotionLocal(text) {
  const keywords = {
    happiness: ["отлично", "хорошо", "счастлив", "рад", "супер", "классно", "замечательно", "прекрасно", "весело"],
    sadness: ["грусть", "грустно", "печаль", "плачу", "плакал", "спал", "усталый", "недоволен", "плохо"],
    anger: ["злой", "гневный", "бешенство", "ненавижу", "раздражает", "возмущен", "зло", "кипит", "злость"],
    worry: ["беспокоюсь", "волнуюсь", "тревога", "переживаю", "опасаюсь", "страшно"],
    love: ["люблю", "любовь", "нежность", "обожаю", "дорог", "милый"],
    enthusiasm: ["воодушевлен", "энтузиазм", "вдохновен", "азарт", "энергия", "мотивация"],
    calmness: ["спокойно", "мир", "гармония", "умиротворен", "расслаблен", "безмятежно"],
    surprise: ["удивлен", "поражен", "неожиданно", "сюрприз", "ошарашен"],
    pleasure: ["приятно", "удовлетворен", "удовольствие", "наслаждаюсь", "кайф"],
    hatred: ["ненависть", "противно", "мерзко", "отвратительно", "терпеть не могу"],
    boredom: ["скучно", "скука", "нудно", "монотонно", "неинтересно"],
    relief: ["облегчение", "облегчил", "выдох", "спасибо", "наконец"],
    emptiness: ["пусто", "пустота", "никого", "одиноко", "пустынно", "безразлично"],
  };

  const textLower = text.toLowerCase();
  const emotionScores = {};
  Object.keys(keywords).forEach((emotion) => (emotionScores[emotion] = 0));

  for (let [emotion, words] of Object.entries(keywords)) {
    words.forEach((word) => {
      if (textLower.includes(word)) {
        emotionScores[emotion] += 1;
      }
    });
  }

  const maxScore = Math.max(...Object.values(emotionScores));
  let predictedEmotion = "calmness";
  let confidence = 0;

  if (maxScore > 0) {
    predictedEmotion = Object.keys(emotionScores).find((k) => emotionScores[k] === maxScore);
    confidence = Math.min(0.5 + maxScore * 0.08, 0.95);
  } else {
    confidence = 0.5;
  }

  return {
    emotion: predictedEmotion,
    confidence: Math.round(confidence * 100),
  };
}

// ==== ОТПРАВКА FEEDBACK НА СЕРВЕР ====
async function sendFeedbackToAPI(text, predictedEmotion, correctedEmotion) {
  const predictedRu = emotionRussianModel[predictedEmotion] || predictedEmotion;
  const correctedRu = emotionRussianModel[correctedEmotion] || correctedEmotion;

  try {
    const response = await fetch(`${API_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        predicted_emotion: predictedRu,
        corrected_emotion: correctedRu,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.warn(
        "Не удалось отправить feedback, статус:",
        response.status,
        "ответ:",
        errText
      );
      return false;
    }

    console.log("✓ Feedback отправлен на сервер");
    return true;
  } catch (error) {
    console.warn("Ошибка при отправке feedback:", error);
    return false;
  }
}

// ==== ЛОГИКА ФОРМЫ ====
async function analyzeEntry() {
  const text = document.getElementById("entryText").value.trim();
  if (!text) {
    showError("Пожалуйста, напиши что-нибудь!");
    return;
  }

  const btn = document.getElementById("analyzeBtn");
  const originalText = btn.innerHTML;
  btn.innerHTML = ' Анализирую...';
  btn.disabled = true;

  try {
    const prediction = await predictEmotionFromAPI(text);
    selectedEmotion = prediction.emotion;
    currentPredictionFromAPI = prediction;

    document.getElementById("resultEmoji").textContent = emotionEmojis[prediction.emotion];
    document.getElementById("resultLabel").textContent = emotionRussian[prediction.emotion];

    const sourceIndicator = useAIModel ? " AI модель" : " Локальный анализ";
    document.getElementById("resultSource").textContent = sourceIndicator;
    document.getElementById("resultSource").style.display = "block";

    const buttonsContainer = document.getElementById("emotionButtons");
    buttonsContainer.innerHTML = "";
    Object.entries(emotionEmojis).forEach(([emotion, emoji]) => {
      const btnOpt = document.createElement("button");
      btnOpt.type = "button";
      btnOpt.className = "emotion-option" + (emotion === prediction.emotion ? " selected" : "");
      btnOpt.innerHTML = `
        <span class="emotion-option-emoji">${emoji}</span>
        <span class="emotion-option-text">${emotionRussian[emotion]}</span>
      `;
      btnOpt.onclick = (event) => selectEmotion(emotion, event);
      buttonsContainer.appendChild(btnOpt);
    });

    document.getElementById("emotionResult").classList.add("show");
  } catch (error) {
    showError("Ошибка при анализе: " + error.message);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

function selectEmotion(emotion, event) {
  selectedEmotion = emotion;

  document.querySelectorAll(".emotion-option").forEach((btn) => btn.classList.remove("selected"));
  if (event && event.target) {
    event.target.closest(".emotion-option").classList.add("selected");
  }

  const emojiEl = document.getElementById("resultEmoji");
  const labelEl = document.getElementById("resultLabel");
  if (emojiEl) emojiEl.textContent = emotionEmojis[emotion] || "❓";
  if (labelEl) labelEl.textContent = emotionRussian[emotion] || "";
}

// ==== СОХРАНЕНИЕ ЗАПИСИ ====
async function saveEntry() {
  const text = document.getElementById("entryText").value.trim();

  if (!selectedEmotion) {
    showError("Выбери эмоцию перед сохранением!");
    return;
  }

  const now = new Date();
  const entry = {
    date: now.toISOString(),
    emotion: selectedEmotion,
    text: text,
  };

  const saved = await saveEntryRemote(entry);
  if (saved) {
    entries.push(saved);

    if (currentPredictionFromAPI && currentPredictionFromAPI.emotion !== selectedEmotion) {
      await sendFeedbackToAPI(text, currentPredictionFromAPI.emotion, selectedEmotion);
    }

    showSuccess("✅ Запись сохранена!");
    document.getElementById("entryText").value = "";
    document.getElementById("emotionResult").classList.remove("show");
    selectedEmotion = null;
    currentPredictionFromAPI = null;
    updateUI();
  }
}

// ==== ОБНОВЛЕНИЕ UI ====
function updateUI() {
  updateEntriesList();
  updateChart();
  updateStats();
}

// ==== СПИСОК ПОСЛЕДНИХ ЗАПИСЕЙ ====
function updateEntriesList() {
  const list = document.getElementById("entriesList");
  if (!list) return;

  if (entries.length === 0) {
    list.innerHTML =
      '<p style="color: var(--color-text-secondary); text-align: center; padding: var(--space-16);">Нет записей. Начни с первой! 👇</p>';
    return;
  }

  list.innerHTML = entries
    .slice()
    .reverse()
    .slice(0, 5)
    .map(
      (entry) => `
        <div class="entry-item">
          <div class="entry-content">
            <div class="entry-text">"${escapeHtml(
              entry.text.substring(0, 80)
            )}..."</div>
            <div class="entry-meta">${new Date(entry.date).toLocaleDateString(
              "ru-RU"
            )}</div>
          </div>
          <div class="entry-emotion">${
            emotionEmojis[entry.emotion] || "❓"
          }</div>
          <button class="btn btn-secondary btn-small" onclick="deleteEntry(${entry.id})" title="Удалить запись">✕</button>
        </div>
      `
    )
    .join("");
}

// ==== АРХИВ ====
function filterArchive() {
  const emotionFilter = document.getElementById("emotionFilter")?.value || "";
  const sortFilter = document.getElementById("sortFilter")?.value || "newest";
  const perPage = parseInt(
    document.getElementById("perPageFilter")?.value || "10",
    10
  );

  filteredEntries = entries.filter((entry) => {
    const matchesEmotion = !emotionFilter || entry.emotion === emotionFilter;
    return matchesEmotion;
  });

  if (sortFilter === "newest") {
    filteredEntries.sort((a, b) => new Date(b.date) - new Date(a.date));
  } else {
    filteredEntries.sort((a, b) => new Date(a.date) - new Date(b.date));
  }

  currentPage = 1;
  displayArchive(perPage);
}

function displayArchive(perPage) {
  const archiveList = document.getElementById("archiveList");
  const paginationEl = document.getElementById("pagination");
  const statsEl = document.getElementById("archiveStats");
  if (!archiveList || !paginationEl || !statsEl) return;

  const startIndex = (currentPage - 1) * perPage;
  const endIndex = startIndex + perPage;
  const pageEntries = filteredEntries.slice(startIndex, endIndex);

  if (filteredEntries.length === 0) {
    archiveList.innerHTML =
      '<p style="color: var(--color-text-secondary); text-align: center; padding: var(--space-16);">Записей не найдено</p>';
    paginationEl.innerHTML = "";
    statsEl.innerHTML = "Всего записей: 0";
    return;
  }

  archiveList.innerHTML = pageEntries
    .map(
      (entry, index) => `
      <div class="entry-item">
        <div class="entry-content">
          <div class="entry-text">"${escapeHtml(entry.text)}"</div>
          <div class="entry-meta">
            ${new Date(entry.date).toLocaleDateString("ru-RU")} •
            ${new Date(entry.date).toLocaleTimeString("ru-RU", {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </div>
        <div class="entry-emotion">${
          emotionEmojis[entry.emotion] || "❓"
        }</div>
        <button class="btn btn-secondary btn-small"
                onclick="deleteEntryFromArchive(${startIndex + index})"
                title="Удалить запись">🗑️</button>
      </div>
    `
    )
    .join("");

  const totalPages = Math.ceil(filteredEntries.length / perPage);
  const paginationHTML = [];

  if (currentPage > 1) {
    paginationHTML.push(
      `<button class="btn-pagination" onclick="goToPage(${currentPage - 1})">← Предыдущая</button>`
    );
  }

  for (let i = 1; i <= totalPages; i++) {
    if (i === currentPage) {
      paginationHTML.push(
        `<button class="btn-pagination active">${i}</button>`
      );
    } else if (
      i === 1 ||
      i === totalPages ||
      (i >= currentPage - 1 && i <= currentPage + 1)
    ) {
      paginationHTML.push(
        `<button class="btn-pagination" onclick="goToPage(${i})">${i}</button>`
      );
    } else if (i === currentPage - 2 || i === currentPage + 2) {
      paginationHTML.push(
        `<button class="btn-pagination" disabled>...</button>`
      );
    }
  }

  if (currentPage < totalPages) {
    paginationHTML.push(
      `<button class="btn-pagination" onclick="goToPage(${currentPage + 1})">Следующая →</button>`
    );
  }

  paginationEl.innerHTML = paginationHTML.join("");

  const startNum = startIndex + 1;
  const endNum = Math.min(endIndex, filteredEntries.length);
  statsEl.innerHTML = `Показано ${startNum}-${endNum} из ${filteredEntries.length} записей`;
}

function goToPage(page) {
  currentPage = page;
  const perPage = parseInt(
    document.getElementById("perPageFilter")?.value || "10",
    10
  );
  displayArchive(perPage);
  const archiveList = document.getElementById("archiveList");
  if (archiveList) archiveList.scrollTop = 0;
}

// ==== УДАЛЕНИЕ ИЗ АРХИВА ПО ИНДЕКСУ В filteredEntries ====
async function deleteEntryFromArchive(globalIndex) {
  const entry = filteredEntries[globalIndex];
  if (!entry) return;

  const ok = await deleteEntryRemote(entry.id);
  if (!ok) return;

  // удаляем из общего списка
  entries = entries.filter((e) => e.id !== entry.id);

  // пересобираем фильтр и UI
  filterArchive();
  updateUI();
}

// ==== ВСПОМОГАТЕЛЬНОЕ ====
function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}

// ==== ГРАФИК ====
function updateChart() {
  const canvas = document.getElementById("emotionChart");
  if (!canvas) return;

  const last30Days = {};
  const today = new Date();

  for (let i = 29; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const key = date.toLocaleDateString("ru-RU", {
      day: "2-digit",
      month: "2-digit",
    });
    last30Days[key] = [];
  }

  entries.forEach((entry) => {
    const entryDate = new Date(entry.date);
    const key = entryDate.toLocaleDateString("ru-RU", {
      day: "2-digit",
      month: "2-digit",
    });
    if (last30Days[key]) {
      last30Days[key].push({
        emotion: entry.emotion,
        value: emotionScaleValues[entry.emotion],
        emoji: emotionEmojis[entry.emotion],
      });
    }
  });

  const labels = Object.keys(last30Days);

  const scatterData = [];
  labels.forEach((label, dateIndex) => {
    last30Days[label].forEach((entry) => {
      scatterData.push({
        x: dateIndex,
        y: entry.value,
        emoji: entry.emoji,
        emotion: entry.emotion,
        color: emotionColors[entry.emotion],
      });
    });
  });

  const ctx = canvas.getContext("2d");

  if (chart) {
    chart.destroy();
  }

  const emojiPlugin = {
    id: "emojiPlugin",
    afterDatasetsDraw(chartInstance) {
      const ctx = chartInstance.ctx;
      const xScale = chartInstance.scales.x;
      const yScale = chartInstance.scales.y;

      scatterData.forEach((point) => {
        const x = xScale.getPixelForValue(point.x);
        const y = yScale.getPixelForValue(point.y);

        ctx.font = "20px system-ui";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(point.emoji || "❓", x, y);
      });
    },
  };

  chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Эмоции",
          data: scatterData.map((p) => ({ x: p.x, y: p.y })),
          showLine: false,
          pointRadius: 0,
          pointHoverRadius: 0,
          backgroundColor: "transparent",
          borderColor: "transparent",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "nearest",
        intersect: true,
      },
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: labels.length - 1,
          ticks: {
            callback: function (value) {
              const idx = Math.round(value);
              return labels[idx] || "";
            },
            maxRotation: 0,
            autoSkip: true,
          },
          title: { display: true, text: "Дата" },
        },
        y: {
          beginAtZero: true,
          min: 0,
          max: 12,
          ticks: {
            stepSize: 1,
            callback: function (value) {
              const emotionKey = Object.keys(emotionScaleValues).find(
                (e) => emotionScaleValues[e] === value
              );
              return emotionKey ? emotionRussian[emotionKey] : "";
            },
          },
          title: { display: true, text: "Эмоциональное состояние" },
        },
      },
    },
    plugins: [emojiPlugin],
  });
}

// ==== СТАТИСТИКА ====
function updateStats() {
  const totalEl = document.getElementById("totalEntries");
  const topEmotionEl = document.getElementById("topEmotion");
  const lastEntryEl = document.getElementById("lastEntry");
  const distribEl = document.getElementById("emotionDistribution");

  if (totalEl) totalEl.textContent = entries.length;

  if (entries.length === 0) {
    if (topEmotionEl) topEmotionEl.textContent = "—";
    if (lastEntryEl) lastEntryEl.textContent = "—";
    if (distribEl) distribEl.innerHTML = "";
    return;
  }

  const emotionCounts = {};
  entries.forEach((e) => {
    emotionCounts[e.emotion] = (emotionCounts[e.emotion] || 0) + 1;
  });

  const topEmotionKey = Object.keys(emotionCounts).reduce((a, b) =>
    emotionCounts[a] > emotionCounts[b] ? a : b
  );
  if (topEmotionEl)
    topEmotionEl.textContent = emotionEmojis[topEmotionKey] || "❓";

  if (lastEntryEl)
    lastEntryEl.textContent =
      emotionEmojis[entries[entries.length - 1].emotion] || "❓";

  if (distribEl) {
    const distribution = Object.entries(emotionCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([emotion, count]) => {
        const percentage = Math.round((count / entries.length) * 100);
        return `
          <div style="margin-bottom: var(--space-16);">
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
              <span style="font-size: 1.3em; margin-right: var(--space-8);">${
                emotionEmojis[emotion] || "❓"
              }</span>
              <span style="font-weight: 500;">${emotionRussian[emotion]}</span>
              <span style="margin-left: auto; color: var(--color-text-secondary);">${percentage}%</span>
            </div>
            <div style="width: 100%; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;">
              <div style="width: ${percentage}%; height: 100%; background: ${
          emotionColors[emotion]
        }; border-radius: 4px;"></div>
            </div>
          </div>
        `;
      })
      .join("");

    distribEl.innerHTML = distribution;
  }
}

// ==== СООБЩЕНИЯ ====
function showSuccess(msg) {
  const el = document.getElementById("successMsg");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("success");
  el.classList.remove("error");
  el.style.display = "block";
  setTimeout(() => {
    el.style.display = "none";
  }, 3000);
}

function showError(msg) {
  const el = document.getElementById("errorMsg");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("error");
  el.classList.remove("success");
  el.style.display = "block";
  setTimeout(() => {
    el.style.display = "none";
  }, 3000);
}

// ==== ВКЛАДКИ ====
function switchTab(tabName) {
  document
    .querySelectorAll(".tab-content")
    .forEach((el) => el.classList.remove("active"));
  document
    .querySelectorAll(".tab-btn")
    .forEach((el) => el.classList.remove("active"));
  const tab = document.getElementById(tabName);
  if (tab) tab.classList.add("active");

  const btn = document.querySelector(`.tab-btn[onclick*="${tabName}"]`);
  if (btn) btn.classList.add("active");

  if (tabName === "chart") {
    setTimeout(() => {
      if (chart) chart.resize();
    }, 0);
  } else if (tabName === "archive") {
    filterArchive();
  }
}

// ==== ОЧИСТКА ВСЕХ ДАННЫХ ====
async function clearAllData() {
  if (
    !confirm(
      "Ты уверена? Это удалит все записи в удалённой базе (таблица entries)!"
    )
  ) {
    return;
  }

  const { error } = await supabase.from("entries").delete().neq("id", 0);

  if (error) {
    console.error("Ошибка очистки:", error);
    showError("Не удалось удалить все записи на сервере");
    return;
  }

  entries = [];
  updateUI();
  showSuccess("Все данные удалены из базы");
}

// ==== УДАЛЕНИЕ ОДНОЙ ЗАПИСИ (из блока последних записей) ====
async function deleteEntry(id) {
  const ok = await deleteEntryRemote(id);
  if (!ok) return;
  entries = entries.filter((e) => e.id !== id);
  updateUI();
}

// ==== ИНИЦИАЛИЗАЦИЯ ====
loadData();
