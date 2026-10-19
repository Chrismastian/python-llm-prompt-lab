"""Pure logic for building LLM prompts. No network, no print -> testable.

Pemisahan yang sama seperti repo sebelumnya:
  prompts.py  = logika (input -> output, bisa dites tanpa internet)
  client.py   = I/O (HTTP)
Kalau aturan prompt ada di file ini, kita bisa mengukur kualitas prompt
tanpa memanggil API sama sekali.
"""
import json

# System prompt = instruksi tetap yang mengatur PERILAKU model.
# Ini bukan pertanyaan, tapi kontrak: bahasa apa, panjang berapa, format apa.
SYSTEM_PROMPT = (
    "You are a precise assistant. "
    "Answer in the same language as the user's question. "
    "Answer in at most two sentences."
)


def build_messages(question: str) -> list[dict]:
    """Build the chat history the API expects: a list of role/content dicts.

    The API is stateless, so this list IS the conversation. Two roles here:
      system -> how to behave (constant across turns)
      user   -> what is asked (changes every turn)
    """
    # BLANK-1: bangun list berisi DUA dict.
    # Dict pertama: {"role": ..., "content": ...} -> role "system", content SYSTEM_PROMPT
    # Dict kedua  : {"role": ..., "content": ...} -> role "user",   content question
    # Perhatikan: di sisi KANAN titik dua, isinya VARIABEL (tanpa kutip).
    # Di sisi KIRI, nama key pakai kutip.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def strict_messages(question: str) -> list[dict]:
    """Same question, but force a machine-readable answer: a JSON object only.

    Used to compare a loose prompt against a strict one. The extra sentence is
    the whole trick: an output CONTRACT the caller can rely on.
    """
    # BLANK-2: panggil build_messages(question) -> simpan ke variabel `messages`.
    messages = build_messages(question)
    # BLANK-3: tambahkan aturan format ke system prompt.
    # messages[0] itu dict; ambil isinya lewat key, tambah string lewat +=
    # String yang ditambahkan: ' Reply with a JSON object only: {"answer": "...", "confidence": 0.0}'
    messages[0]["content"] += (
        ' Reply with a JSON object only: {"answer": "...", "confidence": 0.0}'
    )
    return messages


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a reply, even if it is wrapped in a code fence.

    Models often answer with ```json ... ``` despite being told not to, so the
    parser has to tolerate the fence instead of assuming a clean object.
    """
    # BLANK-4: buang pagar markdown kalau ada.
    # Petunjuk: pakai str.replace() dua kali -> buang "```json" dan "```",
    # lalu .strip() buang spasi/newline sisa. Cari sendiri posisi tanda kutipnya.
    cleaned = text.replace("```json", "").replace("```", "").strip()
    # BLANK-5: ubah string JSON menjadi dict. Fungsi di modul json.
    return json.loads(cleaned)


def usage_totals(raw: dict) -> dict:
    """Reduce the wire format to the numbers that matter for cost.

    usage is the only honest measure of what a prompt costs: a longer system
    prompt is paid for on EVERY call, because history is resent each time.
    """
    usage = raw.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# --- Temporary teaching check. Delete before the public commit. ---
if __name__ == "__main__":
    msgs = strict_messages("Name one HTTP status code for a bad request")
    print(json.dumps(msgs, indent=2))
    print("parsed:", extract_json('```json\n{"answer": "400", "confidence": 0.9}\n```'))