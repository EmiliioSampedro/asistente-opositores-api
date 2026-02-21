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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)
CORS(app)

# Configuración
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("No se encuentra la API key de OpenAI")

# Inicializar OpenAI
http_client = httpx.Client(
    base_url="https://api.openai.com/v1",
    follow_redirects=True,
    timeout=httpx.Timeout(60.0, connect=5.0)
)
cliente = OpenAI(api_key=API_KEY, http_client=http_client)

# --- CARGA DIFERIDA DEL MODELO PARA AHORRAR MEMORIA ---
_modelo = None

def get_model():
    """Carga el modelo de embeddings solo cuando es necesario."""
    global _modelo
    if _modelo is None:
        logger.info("Cargando modelo de embeddings en español...")
        # Modelo pequeño y eficaz para español
        _modelo = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
        logger.info("Modelo cargado correctamente")
    return _modelo

# --- CARGA DE LOS .PKL (FRAGMENTOS Y EMBEDDINGS) ---
logger.info("Cargando datos del asistente...")
fragmentos = []
embeddings = []
try:
    with open("fragmentos.pkl", "rb") as f:
        fragmentos = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    logger.info(f"Datos cargados: {len(fragmentos)} fragmentos")
except Exception as e:
    logger.error(f"Error cargando datos: {e}")

# --- FUNCIÓN DE BÚSQUEDA SEMÁNTICA ---
def buscar_fragmentos(pregunta, top_k=3):
    if not fragmentos or len(embeddings) == 0:
        return []
    
    # Obtener el modelo (se carga aquí la primera vez)
    modelo = get_model()
    
    # Crear embedding de la pregunta
    embedding_pregunta = modelo.encode(pregunta)
    
    # Calcular similitudes (producto punto, asumiendo embeddings normalizados)
    similitudes = []
    for i, emb in enumerate(embeddings):
        similitud = np.dot(embedding_pregunta, emb) / (np.linalg.norm(embedding_pregunta) * np.linalg.norm(emb))
        similitudes.append((i, similitud))
    
    # Ordenar por similitud
    similitudes.sort(key=lambda x: x[1], reverse=True)
    
    # Devolver los top_k fragmentos
    return [fragmentos[i] for i, _ in similitudes[:min(top_k, len(similitudes))]]

# --- RUTAS DE LA API ---

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Asistente Opositores API (Embeddings español)",
        "fragmentos_cargados": len(fragmentos)
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        pregunta = data.get('pregunta', '').strip()
        
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        logger.info(f"Pregunta recibida: {pregunta[:50]}...")
        
        # Buscar fragmentos relevantes
        fragmentos_relevantes = buscar_fragmentos(pregunta)
        
        if not fragmentos_relevantes:
            return jsonify({"respuesta": "No encontré información relevante en los documentos."})
        
        # Crear contexto con los fragmentos más relevantes
        contexto = "\n\n---\n\n".join(fragmentos_relevantes)
        
        # Consultar a OpenAI
        respuesta = cliente.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente para opositores. Responde basándote ÚNICAMENTE en el contexto proporcionado. Si la respuesta no está en el contexto, di que no tienes esa información. Sé conciso y útil."},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        respuesta_texto = respuesta.choices[0].message.content
        
        # Limpieza de memoria (opcional)
        gc.collect()
        
        return jsonify({
            "respuesta": respuesta_texto,
            "fragmentos_usados": len(fragmentos_relevantes)
        })
        
    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)