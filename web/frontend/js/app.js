const API_BASE = "http://127.0.0.1:8001";
let activeLang = null;

async function getActiveLang() {
  if (activeLang) return activeLang;
  try {
    const res  = await fetch(`${API_BASE}/autocomplete/status`);
    const data = await res.json();
    activeLang = data.language || "english";
    return activeLang;
  } catch {
    return "english";
  }
}

const NUMBER_MAPS = {
  nepali:   ["०","१","२","३","४","५","६","७","८","९"],
  hindi:    ["०","१","२","३","४","५","६","७","८","९"],
  bengali:  ["০","১","২","৩","৪","৫","৬","৭","৮","৯"],
  tamil:    ["௦","௧","௨","௩","௪","௫","௬","௭","௮","௯"],
  kannada:  ["೦","೧","೨","೩","೪","೫","೬","೭","೮","೯"],
  gujarati: ["૦","૧","૨","૩","૪","૫","૬","૭","૮","૯"],
  punjabi:  ["੦","੧","੨","੩","੪","੫","੬","੭","੮","੯"],
  sanskrit: ["०","१","२","३","४","५","६","७","८","९"],
  english:  ["0","1","2","3","4","5","6","7","8","9"],
};

function toLocalNumber(n, lang) {
  const digits = NUMBER_MAPS[lang] || NUMBER_MAPS["english"];
  return String(n).padStart(3, "0").split("").map(d => digits[parseInt(d)]).join("");
}

function toLocalPinCode(n, lang) {
  const digits = NUMBER_MAPS[lang] || NUMBER_MAPS["english"];
  return String(n).padStart(4, "0").split("").map(d => digits[parseInt(d)]).join("");
}

function toArabic(localNum, lang) {
  const digits = NUMBER_MAPS[lang] || NUMBER_MAPS["english"];
  return localNum.split("").map(ch => {
    // Already an English digit
    if (/\d/.test(ch)) return ch;
    const idx = digits.indexOf(ch);
    return idx >= 0 ? String(idx) : ch;
  }).join("");
}

async function logout() {
  await fetch(`${API_BASE}/autocomplete/reset`, { method: "POST" });
  window.location.href = "index.html";
}

document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("autocompleteInput");
  const box   = document.getElementById("suggestionsList");

  if (!input || !box) {
    console.error("Missing DOM elements: Input or suggestionsList not found.");
    return;
  }

  let activeIndex = -1;
  let token       = null;

  // ── Address segment definitions ───────────────────────────────────────────
  // Pattern: [kataho_code] [name...] [kataho_code]
  // Segment 0: kataho code (2-3 chars, numeric)
  // Segment 1..N-1: address words (model suggestions)
  // Last segment: kataho code (4 chars, numeric)

  function getSegments(value) {
    return value.trimStart().split(/\s+/);
  }

  function getCurrentSegmentIndex(value) {
    const segs = getSegments(value);
    // If value ends with space, user is starting a new segment
    const endsWithSpace = value.endsWith(" ");
    return endsWithSpace ? segs.length : segs.length - 1;
  }

  function getCurrentPrefix(value) {
    const segs = getSegments(value);
    if (value.endsWith(" ")) return "";
    return segs[segs.length - 1] || "";
  }

  function isEnglishDigits(str) {
    return str.length > 0 && /^\d+$/.test(str);
  }

  function isLocalDigits(str, lang) {
    if (!str.length) return false;
    const digits = NUMBER_MAPS[lang] || NUMBER_MAPS["english"];
    return str.split("").every(ch => digits.includes(ch));
  }

  // ── Number generators ─────────────────────────────────────────────────────
  async function generateFirstCodes(prefix) {
    const lang   = await getActiveLang();
    const arabic = toArabic(prefix, lang);
    const results = [];

    if (arabic.length === 1) {
      // 1 char typed → show 2-digit numbers starting with that digit (00-99)
      for (let i = 0; i <= 99; i++) {
        const num = String(i).padStart(2, "0");
        if (num.startsWith(arabic)) {
          results.push(toLocalNumber(i, lang).slice(-2)); // 2-digit display
          if (results.length >= 10) break;
        }
      }
    } else {
      // 2+ chars typed → show 3-digit numbers (000-999)
      for (let i = 0; i <= 999; i++) {
        const num = String(i).padStart(3, "0");
        if (num.startsWith(arabic)) {
          results.push(toLocalNumber(i, lang));
          if (results.length >= 10) break;
        }
      }
    }

    return results;
  }

  async function generateLastCodes(prefix) {
    const lang    = await getActiveLang();
    const arabic  = toArabic(prefix, lang);
    const results = [];
    for (let i = 0; i <= 9999; i++) {
      const code = String(i).padStart(4, "0");
      if (code.startsWith(arabic)) {
        results.push(toLocalPinCode(i, lang));
        if (results.length >= 10) break;
      }
    }
    return results;
  }

  // ── Token ─────────────────────────────────────────────────────────────────
  async function getToken() {
    if (token) return token;
    try {
      const res  = await fetch(`${API_BASE}/autocomplete/token`);
      const data = await res.json();
      token = data.access_token;
      return token;
    } catch (err) {
      console.error("Token fetch error:", err);
      return null;
    }
  }

  // ── Fetch model suggestions ───────────────────────────────────────────────
  async function fetchSuggestions(query) {
    if (!query) return [];
    try {
      const t   = await getToken();
      const res = await fetch(`${API_BASE}/autocomplete/suggest`, {
        method:  "POST",
        headers: {
          "Content-Type":  "application/json",
          "Authorization": `Bearer ${t}`,
        },
        body: JSON.stringify({ text: query.toLowerCase().trim() }),
      });
      const data = await res.json();
      return data.data?.map(item => item.label) || [];
    } catch (err) {
      console.error("Suggestion fetch error:", err);
      return [];
    }
  }

  // ── Decide which suggestions to show ─────────────────────────────────────
  async function getSuggestions(value) {
    const segIndex = getCurrentSegmentIndex(value);
    const prefix   = getCurrentPrefix(value);
    const lang     = await getActiveLang();

    if (segIndex === 0) {
      if (!prefix) return [];

      // Both English digits and local script → always show local script suggestions
      if (isEnglishDigits(prefix) || isLocalDigits(prefix, lang)) {
        return await generateFirstCodes(prefix, lang);
      }

      return [];
    }

    if (segIndex >= 1) {
      if (!prefix) return [];

      // Both English digits and local script → always show local script pin codes
      if (isEnglishDigits(prefix) || isLocalDigits(prefix, lang)) {
        return await generateLastCodes(prefix, lang);
      }
    }

    // Text prefix → model suggestion
    if (prefix.length >= 1) {
      return await fetchSuggestions(prefix);
    }

    return [];
  }

  // ── Render ────────────────────────────────────────────────────────────────
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

  // ── Select a suggestion ───────────────────────────────────────────────────
  async function selectWord(word) {
    const value    = input.value;
    const segments = getSegments(value);
    const endsWithSpace = value.endsWith(" ");

    // Replace current segment or append
    if (!endsWithSpace && segments.length > 0) {
      segments[segments.length - 1] = word;
    } else {
      segments.push(word);
    }

    input.value   = segments.join(" ") + " ";
    box.innerHTML = "";

    // Send feedback only for model words (not numbers)
    const segIndex = getCurrentSegmentIndex(value);
    if (segIndex > 0 && !/^\d+$/.test(word)) {
      try {
        const t = await getToken();
        await fetch(`${API_BASE}/autocomplete/feedback`, {
          method:  "POST",
          headers: {
            "Content-Type":  "application/json",
            "Authorization": `Bearer ${t}`,
          },
          body: JSON.stringify({ input: getCurrentPrefix(value), label: word }),
        });
      } catch (err) {
        console.error("Feedback error:", err);
      }
    }
  }

  // ── Input typing ──────────────────────────────────────────────────────────
  input.addEventListener("input", async () => {
    const value = input.value;
    if (!value.trim()) {
      box.innerHTML = "";
      return;
    }
    const list = await getSuggestions(value);
    renderSuggestions(list);
  });

  // ── Keyboard navigation ───────────────────────────────────────────────────
  input.addEventListener("keydown", (e) => {
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

  // ── Close on outside click ────────────────────────────────────────────────
  document.addEventListener("click", (e) => {
    if (!box.contains(e.target) && e.target !== input) {
      box.innerHTML = "";
    }
  });
});