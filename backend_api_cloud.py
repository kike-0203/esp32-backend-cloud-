import os, base64, time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import requests
from flask_cors import CORS

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("❌ Falta OPENAI_API_KEY en .env o variables del entorno")

client = OpenAI(api_key=api_key)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Permite hasta ~6 MB por si usas VGA con compresión moderada
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 6 * 1024 * 1024))

state = {"stage": "idle", "error": None, "bytes": 0, "source": None, "last_ts": 0}
last_answer = None

def _set_state(stage, **kw):
    state["stage"] = stage
    state["last_ts"] = time.time()
    state.update(kw)
    print(f"[stage] {stage} | {kw}", flush=True)

def analyze_image_bytes(image_bytes: bytes, style: str) -> str:
    """Detalle ALTO y salida estructurada, apta para OLED si style=oled."""
    _set_state("analyzing", bytes=len(image_bytes))
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    max_tokens = int(os.getenv("MAX_TOKENS", "600"))
    oled_hint = (
        "Formatea en bloques claros, líneas cortas (<= 40–60 chars) y viñetas. Evita párrafos largos."
        if style == "oled" else
        "Estructura la respuesta con encabezados y viñetas claras."
    )

    system_msg = (
        "Eres un asistente experto en OCR y análisis técnico. "
        "Extraes texto con precisión y respondes preguntas presentes en la imagen, "
        "mostrando cálculos y resultados si aplica."
    )
    user_text = (
        "1) Extrae el texto exactamente como aparece (corrige errores obvios). "
        "2) Si hay pregunta/fórmula, resuélvela paso a paso de forma clara. "
        "3) Termina con una conclusión breve y útil.\n"
        f"{oled_hint}\n"
        "Estructura:\n"
        "TÍTULO\n— EXTRACCIÓN (puntos)\n— ANÁLISIS/EXPLICACIÓN (pasos)\n— RESULTADO/CONCLUSIÓN\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system_msg},
            {"role":"user","content":[
                {"type":"text","text":user_text},
                {"type":"image_url","image_url":{
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high"        # ALTA calidad para mejor OCR/visión
                }}
            ]}
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text or "No pude interpretar texto o pregunta clara en la imagen."

@app.route("/process_upload", methods=["POST"])
def process_upload():
    global last_answer
    try:
        style = (request.args.get("style") or "").lower()  # ?style=oled opcional
        _set_state("receiving", source="upload", error=None, bytes=0)
        if "file" not in request.files:
            _set_state("error", error="No se recibió 'file'")
            return jsonify({"ok": False, "error": "No se recibió 'file'"}), 400

        file = request.files["file"]
        image_bytes = file.read()
        if not image_bytes:
            _set_state("error", error="Imagen vacía")
            return jsonify({"ok": False, "error": "Imagen vacía"}), 400

        _set_state("receiving", bytes=len(image_bytes))
        answer = analyze_image_bytes(image_bytes, style)
        last_answer = answer
        _set_state("done", error=None)
        return jsonify({"ok": True, "answer": answer}), 200

    except Exception as e:
        _set_state("error", error=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/process_url", methods=["POST"])
def process_url():
    global last_answer
    data = request.get_json(silent=True) or {}
    photo_url = data.get("photo_url")
    style = (data.get("style") or "").lower()
    if not photo_url:
        return jsonify({"ok": False, "error": "Falta photo_url"}), 400

    try:
        _set_state("downloading", source="url", error=None, bytes=0)
        image_bytes = None
        for i in range(3):
            try:
                r = requests.get(photo_url, timeout=15)
                r.raise_for_status()
                image_bytes = r.content
                break
            except Exception as e:
                print(f"[download] intento {i+1}/3 fallo: {e}", flush=True)
                time.sleep(0.8)

        if not image_bytes:
            _set_state("error", error="Timeout o fallo al descargar la imagen")
            return jsonify({"ok": False, "error": "Timeout al descargar la imagen"}), 500

        _set_state("downloading", bytes=len(image_bytes))
        answer = analyze_image_bytes(image_bytes, style)
        last_answer = answer
        _set_state("done", error=None)
        return jsonify({"ok": True, "answer": answer}), 200

    except Exception as e:
        _set_state("error", error=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/answer", methods=["GET"])
def get_answer():
    if last_answer:
        return jsonify({"answer": last_answer}), 200
    return "", 204

@app.route("/status", methods=["GET"])
def status():
    return jsonify(state), 200

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "ok": True,
        "routes": ["/process_upload (POST file)[?style=oled]", "/process_url (POST photo_url, style)", "/answer (GET)", "/status (GET)", "/health (GET)"]
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"[i] Backend listo en http://0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port)
