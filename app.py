import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import logging
import sys
import gc
import warnings
warnings.filterwarnings("ignore")

# === LIMPIEZA TOTAL (como al principio) ===
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# === CONFIGURACIÓN BÁSICA (la que funcionaba) ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
CORS(app)
logger.info("✅ CORS configurado")

# === OPENAI SENCILLO (el que funcionaba) ===
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("❌ No hay API key")
    cliente_openai = None
else:
    try:
        # La forma más simple que funcionó
        openai.api_key = API_KEY
        cliente_openai = openai
        logger.info("✅ OpenAI listo (modo simple)")
    except Exception as e:
        logger.error(f"❌ Error OpenAI: {e}")
        cliente_openai = None

def get_openai_client():
    return cliente_openai

# === MODELO PEQUEÑO (el que funcionaba) ===
_modelo = None
def get_model():
    global _modelo
    if _modelo is None:
        logger.info("🔄 Cargando modelo...")
        try:
            from sentence_transformers import SentenceTransformer
            _modelo = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            logger.info("✅ Modelo cargado")
        except Exception as e:
            logger.error(f"❌ Error modelo: {e}")
    return _modelo

# === DATOS ===
logger.info("📚 Cargando datos...")
fragmentos = []
embeddings = []

# Verificar archivos .pkl
pkl_path = os.path.join(os.path.dirname(__file__), 'fragmentos.pkl')
emb_path = os.path.join(os.path.dirname(__file__), 'embeddings.pkl')

if os.path.exists(pkl_path) and os.path.exists(emb_path):
    logger.info("✅ Archivos .pkl encontrados")
    try:
        with open(pkl_path, "rb") as f:
            fragmentos = pickle.load(f)
        logger.info(f"   - {len(fragmentos)} fragmentos")
        
        with open(emb_path, "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f"   - {len(embeddings)} embeddings")
        
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
    except Exception as e:
        logger.error(f"❌ Error cargando .pkl: {e}")
else:
    logger.error("❌ No hay .pkl")
    logger.info(f"Archivos en directorio: {os.listdir('.')}")

# === BÚSQUEDA SIMPLE (la que funcionaba) ===
def buscar_fragmentos(pregunta, top_k=5):
    if not fragmentos or len(embeddings) == 0:
        return []
    
    try:
        modelo = get_model()
        if modelo is None:
            return []
        
        emb_pregunta = modelo.encode(pregunta)
        similitudes = []
        for i, emb in enumerate(embeddings):
            sim = np.dot(emb_pregunta, emb) / (np.linalg.norm(emb_pregunta) * np.linalg.norm(emb))
            similitudes.append((i, sim))
        
        similitudes.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in similitudes[:top_k]]
        logger.info(f"🔍 Top similitud: {similitudes[0][1]:.4f}")
        return [fragmentos[i] for i in indices]
    except Exception as e:
        logger.error(f"❌ Error búsqueda: {e}")
        return []

# === RUTAS (las que funcionaban) ===
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Asistente Opositores (Versión Original)",
        "fragmentos": len(fragmentos),
        "modelo": _modelo is not None,
        "openai": cliente_openai is not None
    })

@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        "fragmentos": len(fragmentos),
        "embeddings": len(embeddings),
        "modelo": _modelo is not None,
        "archivos": os.listdir('.')[:10]
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "JSON inválido"}), 400
            
        pregunta = data.get('pregunta', '').strip()
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        logger.info(f"📨 {pregunta[:100]}...")
        
        fragmentos_relevantes = buscar_fragmentos(pregunta, top_k=5)
        if not fragmentos_relevantes:
            return jsonify({"respuesta": "No encontré información."})
        
        contexto = "\n\n---\n\n".join(fragmentos_relevantes)
        
        # === Código compatible con openai 0.28.1 y 1.x ===
        try:
            # Intento con la sintaxis antigua (0.28.1)
            respuesta = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto para opositores."},
                    {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
                ],
                temperature=0.7,
                max_tokens=400
            )
            respuesta_texto = respuesta.choices[0].message.content
        except AttributeError:
            # Si falla, usamos la sintaxis nueva (1.x)
            from openai import OpenAI
            cliente = OpenAI(api_key=openai.api_key)
            respuesta = cliente.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto para opositores."},
                    {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
                ],
                temperature=0.7,
                max_tokens=400
            )
            respuesta_texto = respuesta.choices[0].message.content
        
        return jsonify({"respuesta": respuesta_texto})
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
