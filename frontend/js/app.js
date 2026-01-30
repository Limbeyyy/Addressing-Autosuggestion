const input = document.getElementById("search");
const box = document.getElementById("suggestions");

let suggestions = [];
let activeIndex = -1;

const API_BASE = "http://127.0.0.1:8001";

// Fetch suggestions
async function fetchSuggestions(query) {
  if (!query) return [];
  const response = await fetch(`${API_BASE}/suggest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: query.toLowerCase().trim() }),
  });

  const data = await response.json();
  return data.suggestions || [];
}

// Render suggestions
function renderSuggestions(list) {
  box.innerHTML = "";
  activeIndex = -1;

  if (!Array.isArray(list) || list.length === 0) return;

  list.forEach(item => {
    const div = document.createElement("div");
    div.className = "item"; // Important for keyboard navigation
    div.textContent = item;

    // Click handler
    div.addEventListener("click", () => {
      selectWord(item, input.value);
    });

    box.appendChild(div);
  });
}

// Handle selection
function selectWord(word, query) {
  input.value = word;
  box.innerHTML = "";

  // Send feedback to backend
  fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: query,
      label: word
    })
  }).then(res => res.json())
    .then(data => console.log("Feedback sent:", data))
    .catch(err => console.error("Feedback error:", err));
}

// Input typing
input.addEventListener("input", async () => {
  const q = input.value.trim();
  if (!q) {
    box.innerHTML = "";
    return;
  }

  suggestions = await fetchSuggestions(q);
  renderSuggestions(suggestions);
});

// Keyboard navigation
input.addEventListener("keydown", (e) => {
  const items = document.querySelectorAll(".item");
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
