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
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)

# Configuración CORS simple y efectiva
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["POST", "OPTIONS", "GET"])
logger.info("✅ CORS configurado para recibir peticiones desde cualquier origen")

# Aumentar tiempo de espera para peticiones largas
app.config['TIMEOUT'] = 120

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

# --- CARGA DEL MODELO AL ARRANCAR (AHORA CON 2GB DE RAM) ---
logger.info("🔄 Cargando modelo multilingüe al arrancar...")
try:
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("✅ Modelo multilingüe cargado correctamente")
except Exception as e:
    logger.error(f"❌ Error fatal cargando el modelo: {e}")
    modelo = None

def get_model():
    """Devuelve el modelo ya cargado."""
    return modelo

# --- CARGA DE LOS .PKL ---
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
def buscar_fragmentos(pregunta, top_k=7):
    """Busca los fragmentos más relevantes para la pregunta."""
    if not fragmentos or len(embeddings) == 0:
        logger.warning("⚠️ No hay fragmentos o embeddings cargados")
        return []
    
    try:
        modelo_local = get_model()
        if modelo_local is None:
            logger.error("❌ Modelo no disponible")
            return []
            
        emb_pregunta = modelo_local.encode(pregunta)
        
        similitudes = []
        for i, emb in enumerate(embeddings):
            norma_pregunta = np.linalg.norm(emb_pregunta)
            norma_frag = np.linalg.norm(emb)
            if norma_pregunta > 0 and norma_frag > 0:
                sim = np.dot(emb_pregunta, emb) / (norma_pregunta * norma_frag)
            else:
                sim = 0
            similitudes.append((i, sim))
        
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver los top_k fragmentos
        indices = [i for i, _ in similitudes[:min(top_k, len(similitudes))]]
        return [fragmentos[i] for i in indices]
    
    except Exception as e:
        logger.error(f"❌ Error en búsqueda semántica: {e}", exc_info=True)
        return []

# --- RUTAS DE LA API ---

@app.route('/', methods=['GET'])
def home():
    """Endpoint de salud."""
    return jsonify({
        "status": "online",
        "message": "Asistente Opositores API (Versión Professional)",
        "fragmentos_cargados": len(fragmentos),
        "modelo_cargado": modelo is not None,
        "openai_ok": cliente is not None,
        "plan": "Professional 2GB"
    })

@app.route('/debug', methods=['GET'])
def debug():
    """Endpoint para depuración."""
    return jsonify({
        "modelo_cargado": modelo is not None,
        "num_fragmentos": len(fragmentos),
        "num_embeddings": len(embeddings),
        "openai_ok": cliente is not None,
        "python_version": sys.version,
        "modelo_nombre": "paraphrase-multilingual-MiniLM-L12-v2"
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Endpoint principal para el chat."""
    # Manejar preflight OPTIONS
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response, 200
    
    try:
        # Verificar datos de entrada
        data = request.json
        if not data:
            return jsonify({"error": "Formato JSON inválido"}), 400
            
        pregunta = data.get('pregunta', '').strip()
        
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        logger.info(f"📨 Procesando pregunta: {pregunta[:100]}...")
        
        # Verificar que el modelo está disponible
        if modelo is None:
            logger.error("❌ Modelo no disponible")
            return jsonify({"error": "Modelo no disponible"}), 500
        
        # Buscar fragmentos (top_k=7 para más contexto)
        fragmentos_relevantes = buscar_fragmentos(pregunta, top_k=7)
        
        if not fragmentos_relevantes:
            return jsonify({"respuesta": "No encontré información relevante en los documentos."})
        
        # Crear contexto
        contexto = "\n\n---\n\n".join(fragmentos_relevantes)
        
        # Verificar cliente OpenAI
        if not cliente:
            logger.error("Cliente OpenAI no disponible")
            return jsonify({"error": "Cliente OpenAI no configurado"}), 500
        
        # Consultar a OpenAI
        logger.info("🤖 Consultando a OpenAI...")
        try:
            respuesta = cliente.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto para opositores. Responde a la pregunta usando la información del contexto. Si el contexto contiene la respuesta, úsala directamente. No digas que no tienes información si el contexto la contiene."},
                    {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            respuesta_texto = respuesta.choices[0].message.content
            logger.info("✅ Respuesta generada correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error en OpenAI: {e}", exc_info=True)
            return jsonify({"error": f"Error al consultar OpenAI: {str(e)}"}), 500
        
        # Preparar respuesta con CORS
        response = jsonify({"respuesta": respuesta_texto})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
        
    except Exception as e:
        logger.error(f"Error en /chat: {e}", exc_info=True)
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Arrancando aplicación en puerto {port} con plan Professional 2GB")
    app.run(host='0.0.0.0', port=port, debug=False)