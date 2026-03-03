let _token = null;

async function getToken() {
  if (_token) return _token;
  try {
    const res  = await fetch(`${API_BASE}/autocomplete/token`);
    const data = await res.json();
    _token = data.access_token;
    return _token;
  } catch (err) {
    console.error("Token fetch error:", err);
    return null;
  }
}

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

async function sendFeedback(inputText, label) {
  const t = await getToken();
  await fetch(`${API_BASE}/autocomplete/feedback`, {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${t}`,
    },
    body: JSON.stringify({ input: inputText.toLowerCase(), label }),
  });
}

async function logout() {
  await fetch(`${API_BASE}/autocomplete/reset`, { method: "POST" });
  window.location.href = "index.html";
}
