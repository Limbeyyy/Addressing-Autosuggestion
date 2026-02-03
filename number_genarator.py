import re

# ======================================================
# Digit Maps
# ======================================================
ARABIC_TO_NEPALI = {
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९'
}

NEPALI_TO_ARABIC = {v: k for k, v in ARABIC_TO_NEPALI.items()}


# ======================================================
# Conversion Helpers
# ======================================================
def to_nepali(number_str: str) -> str:
    """Convert Arabic digits to Nepali digits"""
    return ''.join(ARABIC_TO_NEPALI.get(ch, ch) for ch in number_str)


def to_arabic(number_str: str) -> str:
    """Convert Nepali digits to Arabic digits"""
    return ''.join(NEPALI_TO_ARABIC.get(ch, ch) for ch in number_str)


# ======================================================
# Number Generators (Precomputed)
# ======================================================
TWO_DIGIT_NUMBERS = [f"{i:02d}" for i in range(100)]
THREE_DIGIT_NUMBERS = [f"{i:03d}" for i in range(1000)]


# ======================================================
# Core Suggestion Logic
# ======================================================
def suggest_prefix_numbers(text: str):
    """
    Rules:
    - Extract ONLY the FIRST numeric prefix (Nepali or Arabic)
    - Ignore suffix text and numbers
    - Suggest matching 2-digit and 3-digit numbers
    """

    match = re.search(r'[०-९0-9]+', text)
    if not match:
        return ["No suggestions"]

    raw_prefix = match.group()
    prefix = to_arabic(raw_prefix)

    suggestions = []

    for num in TWO_DIGIT_NUMBERS:
        if num.startswith(prefix):
            suggestions.append(to_nepali(num))

    for num in THREE_DIGIT_NUMBERS:
        if num.startswith(prefix):
            suggestions.append(to_nepali(num))

    return suggestions if suggestions else ["No suggestions"]


# ======================================================
# CLI Test Mode
# ======================================================
if __name__ == "__main__":
    print("Type text containing number prefix (Nepali or Arabic).")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() == "exit":
            break

        results = suggest_prefix_numbers(user_input)
        print("Prefix suggestions:", ", ".join(results))
