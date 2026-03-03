document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("autocompleteInput");
  const box   = document.getElementById("suggestionsList");

  if (!input || !box) {
    console.error("Missing DOM elements: Input or suggestionsList not found.");
    return;
  }

  let activeIndex = -1;

  // ── Feedback dedup cache ─────────────────────────────────────────────────
  const feedbackCache   = new Map();
  const DEDUP_WINDOW_MS = 20000;

  function shouldSendFeedback(word) {
    const now  = Date.now();
    const last = feedbackCache.get(word);
    if (last && (now - last) < DEDUP_WINDOW_MS) {
      console.log("Skipped duplicate within 20s:", word);
      return false;
    }
    feedbackCache.set(word, now);
    return true;
  }

  // ── Auto-save word ───────────────────────────────────────────────────────
  async function autoSaveWord(inputText, selectedWord) {
    try {
      const lang     = await getActiveLang();
      const isNumber = isEnglishDigits(selectedWord) || isLocalDigits(selectedWord, lang);
      if (isNumber) return;
      if (!shouldSendFeedback(selectedWord)) return;
      await sendFeedback(inputText, selectedWord);
      console.log("Auto-saved word:", selectedWord);
    } catch (err) {
      console.error("Auto-save error:", err);
    }
  }

  // ── Check and save completed address ─────────────────────────────────────
  async function checkAndSaveAddress(value) {
    const segments = getSegments(value.trim());
    if (segments.length < 3) return;

    const lang       = await getActiveLang();
    const lastSeg    = segments[segments.length - 1];
    const lastArabic = toArabic(lastSeg, lang);
    if (!/^\d{4}$/.test(lastArabic)) return;

    const wordSegments = segments.slice(1, -1).filter(seg =>
      !isEnglishDigits(seg) && !isLocalDigits(seg, lang)
    );

    if (!wordSegments.length) return;

    try {
      for (const word of wordSegments) {
        if (!shouldSendFeedback(word)) continue;
        await sendFeedback(word, word);
      }
    } catch (err) {
      console.error("Auto-save feedback error:", err);
    }
  }

  // ── Suggestion routing ───────────────────────────────────────────────────
  async function getSuggestions(value) {
    const cursorPos = input.selectionStart;
    const { segmentIndex, prefix } = getCursorSegmentInfo(value, cursorPos);
    const lang = await getActiveLang();

    if (!prefix) return [];

    if (segmentIndex === 0) {
      if (isEnglishDigits(prefix) || isLocalDigits(prefix, lang)) {
        return await generateFirstCodes(prefix);
      }
      return [];
    }

    const segments = value.trim().split(/\s+/);
    if (segmentIndex === segments.length - 1) {
      if (isEnglishDigits(prefix) || isLocalDigits(prefix, lang)) {
        return await generateLastCodes(prefix);
      }
    }

    return await fetchSuggestions(prefix);
  }

  // ── Render ───────────────────────────────────────────────────────────────
  function renderSuggestions(list) {
    box.innerHTML = "";
    activeIndex   = -1;

    if (!Array.isArray(list) || list.length === 0) return;

    list.forEach(item => {
      const li       = document.createElement("li");
      li.className   = "item";
      li.textContent = item;
      li.addEventListener("click", () => selectWord(item));
      box.appendChild(li);
    });
  }

  async function selectWord(word) {
    const cursorPos = input.selectionStart;
    const value     = input.value;
    const { segmentIndex, prefix } = getCursorSegmentInfo(value, cursorPos);

    const parts         = value.split(/\s+/);
    parts[segmentIndex] = word;

    input.value   = parts.join(" ") + " ";
    box.innerHTML = "";
    input.focus();

    await autoSaveWord(prefix, word);
  }

  // ── Event listeners ──────────────────────────────────────────────────────
  input.addEventListener("input", async () => {
    const value = input.value;
    if (!value.trim()) { box.innerHTML = ""; return; }
    const list = await getSuggestions(value);
    renderSuggestions(list);
  });

  input.addEventListener("keydown", async (e) => {
    if (e.key === " " || e.key === "Enter") {
      await checkAndSaveAddress(input.value);
    }

    const items = box.querySelectorAll(".item");
    if (!items.length) return;

    if (e.key === "ArrowDown") {
      activeIndex = (activeIndex + 1) % items.length;
    } else if (e.key === "ArrowUp") {
      activeIndex = (activeIndex - 1 + items.length) % items.length;
    } else if (e.key === "Enter" && activeIndex >= 0) {
      items[activeIndex].click();
      e.preventDefault();
      return;
    } else {
      return;
    }

    items.forEach(i => i.classList.remove("active"));
    if (items[activeIndex]) items[activeIndex].classList.add("active");
    e.preventDefault();
  });

  document.addEventListener("click", (e) => {
    if (!box.contains(e.target) && e.target !== input) {
      box.innerHTML = "";
    }
  });
});