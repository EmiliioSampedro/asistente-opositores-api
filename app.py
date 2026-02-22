import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
import httpx
import gc
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)
# Configuración CORS


# Configuración CORS ultra simple
CORS(app)  # Sin opciones, la configuración por defecto
logger.info("✅ CORS configurado en modo simple")


# Configuración
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("❌ No se encuentra la API key de OpenAI")
else:
    logger.info("✅ API key de OpenAI encontrada")

# Inicializar OpenAI
try:
    http_client = httpx.Client(
        base_url="https://api.openai.com/v1",
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, connect=5.0)
    )
    cliente = OpenAI(api_key=API_KEY, http_client=http_client)
    logger.info("✅ Cliente OpenAI inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando OpenAI: {e}")
    cliente = None

# --- CARGA DIFERIDA DEL MODELO ULTRA LIGERO ---
_modelo = None

def get_model():
    """Carga el modelo de embeddings solo cuando es necesario."""
    global _modelo
    if _modelo is None:
        logger.info("🔄 Cargando modelo MiniLM L3 de 60MB...")
        try:
            # Forzar recolección de basura antes de cargar
            gc.collect()
            _modelo = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            logger.info("✅ Modelo L3 cargado correctamente")
        except Exception as e:
            logger.error(f"❌ Error cargando el modelo: {e}")
            raise e
    return _modelo

# --- CARGA DE LOS .PKL (FRAGMENTOS Y EMBEDDINGS) ---
logger.info("📚 Cargando datos del asistente...")
fragmentos = []
embeddings = []
try:
    with open("fragmentos.pkl", "rb") as f:
        fragmentos = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    logger.info(f"✅ Datos cargados: {len(fragmentos)} fragmentos")
except FileNotFoundError as e:
    logger.error(f"❌ No se encuentran los archivos .pkl: {e}")
except Exception as e:
    logger.error(f"❌ Error cargando datos: {e}", exc_info=True)

# --- FUNCIÓN DE BÚSQUEDA SEMÁNTICA ---
def buscar_fragmentos(pregunta, top_k=5):
    """Busca los fragmentos más relevantes para la pregunta."""
    if not fragmentos or len(embeddings) == 0:
        logger.warning("⚠️ No hay fragmentos o embeddings cargados")
        return []
    
    try:
        # Obtener el modelo
        modelo = get_model()
        
        # Crear embedding de la pregunta
        embedding_pregunta = modelo.encode(pregunta)
        
        # Calcular similitudes (coseno)
        similitudes = []
        for i, emb in enumerate(embeddings):
            similitud = np.dot(embedding_pregunta, emb) / (np.linalg.norm(embedding_pregunta) * np.linalg.norm(emb))
            similitudes.append((i, similitud))
        
        # Ordenar por similitud
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver los top_k fragmentos
        return [fragmentos[i] for i, _ in similitudes[:min(top_k, len(similitudes))]]
    
    except Exception as e:
        logger.error(f"❌ Error en búsqueda semántica: {e}", exc_info=True)
        return []

# --- RUTAS DE LA API ---

@app.route('/', methods=['GET'])
def home():
    """Endpoint de salud y verificación."""
    return jsonify({
        "status": "online",
        "message": "Asistente Opositores API (Modelo L3 ultra ligero)",
        "fragmentos_cargados": len(fragmentos),
        "modelo_cargado": _modelo is not None,
        "openai_ok": cliente is not None
    })

@app.route('/debug', methods=['GET'])
def debug():
    """Endpoint para depuración."""
    return jsonify({
        "modelo_cargado": _modelo is not None,
        "num_fragmentos": len(fragmentos),
        "num_embeddings": len(embeddings),
        "openai_ok": cliente is not None,
        "python_version": sys.version,
        "memoria": f"{gc.get_count()}"
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Manejar preflight OPTIONS
    if request.method == 'OPTIONS':
        return '', 200
    
    try:  # ← Este try debe estar aquí
        data = request.json
        if not data:
            return jsonify({"error": "Formato JSON inválido"}), 400
            
        pregunta = data.get('pregunta', '').strip()
        
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        logger.info(f"📨 Procesando pregunta: {pregunta[:100]}...")
        
        # Buscar fragmentos (AHORA CON TOP_K=5)
        fragmentos_relevantes = buscar_fragmentos(pregunta, top_k=5)
        
        if not fragmentos_relevantes:
            return jsonify({"respuesta": "No encontré información relevante en los documentos."})
        
        # Crear contexto
        contexto = "\n\n---\n\n".join(fragmentos_relevantes)
        
        # Verificar cliente OpenAI
        if not cliente:
            return jsonify({"error": "Cliente OpenAI no configurado"}), 500
        
        # Consultar a OpenAI con prompt mejorado
        respuesta = cliente.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto para opositores. Responde a la pregunta usando la información del contexto. Si el contexto contiene la respuesta, úsala directamente."},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        respuesta_texto = respuesta.choices[0].message.content
        
        return jsonify({"respuesta": respuesta_texto})
        
    except Exception as e:  # ← Este except debe estar aquí
        logger.error(f"Error en /chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Arrancando aplicación en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)