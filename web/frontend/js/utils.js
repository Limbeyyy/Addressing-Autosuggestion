// ── Language utilities ────────────────────────────────────────────────────

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
    if (/\d/.test(ch)) return ch;
    const idx = digits.indexOf(ch);
    return idx >= 0 ? String(idx) : ch;
  }).join("");
}

function isEnglishDigits(str) {
  return str.length > 0 && /^\d+$/.test(str);
}

function isLocalDigits(str, lang) {
  if (!str.length) return false;
  const digits = NUMBER_MAPS[lang] || NUMBER_MAPS["english"];
  return str.split("").every(ch => digits.includes(ch));
}

// ── Segment utilities ─────────────────────────────────────────────────────

function getSegments(value) {
  return value.trimStart().split(/\s+/);
}

function getCursorSegmentInfo(value, cursorPos) {
  const segments = value.split(/\s+/);
  let count = 0;
  let segmentIndex = 0;

  for (let i = 0; i < segments.length; i++) {
    count += segments[i].length + 1;
    if (cursorPos <= count) {
      segmentIndex = i;
      break;
    }
  }

  const leftText = value.slice(0, cursorPos);
  const prefix   = leftText.split(/\s+/).pop() || "";

  return { segmentIndex, prefix };
}

// ── Number generators ─────────────────────────────────────────────────────

async function generateFirstCodes(prefix) {
  const lang    = await getActiveLang();
  const arabic  = toArabic(prefix, lang);
  const results = [];

  if (arabic.length === 1) {
    for (let i = 0; i <= 99; i++) {
      const num = String(i).padStart(2, "0");
      if (num.startsWith(arabic)) {
        results.push(toLocalNumber(i, lang).slice(-2));
        if (results.length >= 10) break;
      }
    }
  } else {
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