
const API_BASE = "http://127.0.0.1:8001";

async function logout() {
  await fetch(`${API_BASE}/autocomplete/reset`, {
    method: "POST",
  });
  window.location.href = "index.html";
}

document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("autocompleteInput");
  const box   = document.getElementById("suggestionsList");

  if (!input || !box) {
    console.error("Missing DOM elements: Input or suggestionsList div not found.");
    return;
  }

  let suggestions  = [];
  let activeIndex  = -1;
  let token        = null;


  // ── Fetch token once ──────────────────────────────────────────────────────
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

  // ── Fetch suggestions ─────────────────────────────────────────────────────
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

  // ── Render suggestions ────────────────────────────────────────────────────
  function renderSuggestions(list) {
    box.innerHTML = "";
    activeIndex   = -1;

    if (!Array.isArray(list) || list.length === 0) return;

    list.forEach(item => {
      const div       = document.createElement("div");
      div.className   = "item";
      div.textContent = item;

      div.addEventListener("click", () => selectWord(item, input.value));
      box.appendChild(div);
    });
  }
  

  // ── Handle selection ──────────────────────────────────────────────────────
  async function selectWord(word, query) {
    input.value   = word;
    box.innerHTML = "";

    try {
      const t = await getToken();
      const res = await fetch(`${API_BASE}/autocomplete/feedback`, {
        method:  "POST",
        headers: {
          "Content-Type":  "application/json",
          "Authorization": `Bearer ${t}`,
        },
        body: JSON.stringify({ input: query, label: word }),
      });
      const data = await res.json();
      console.log("Feedback sent:", data);
    } catch (err) {
      console.error("Feedback error:", err);
    }
  }

  // ── Input typing ──────────────────────────────────────────────────────────
  input.addEventListener("input", async () => {
    const q = input.value.trim();
    if (!q) {
      box.innerHTML = "";
      return;
    }
    suggestions = await fetchSuggestions(q);
    renderSuggestions(suggestions);
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

    items.forEach(item => item.classList.remove("active"));
    if (items[activeIndex]) items[activeIndex].classList.add("active");
    e.preventDefault();
  });

  // ── Close suggestions on outside click ───────────────────────────────────
  document.addEventListener("click", (e) => {
    if (!box.contains(e.target) && e.target !== input) {
      box.innerHTML = "";
    }
  });

  
});

