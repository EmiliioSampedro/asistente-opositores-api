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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)
CORS(app)  # Permitir peticiones desde cualquier dominio

# Configuración
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("No se encuentra la API key de OpenAI")
    # No salimos porque Render necesita arrancar

# Inicializar OpenAI (con configuración para Render)
http_client = httpx.Client(
    base_url="https://api.openai.com/v1",
    follow_redirects=True,
    timeout=httpx.Timeout(60.0, connect=5.0)
)
cliente = OpenAI(api_key=API_KEY, http_client=http_client)

# Cargar modelo de embeddings (esto puede tomar unos segundos)
logger.info("Cargando modelo de embeddings...")
modelo_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.info("Modelo cargado")

# Cargar fragmentos y embeddings
logger.info("Cargando datos del asistente...")
try:
    with open("fragmentos.pkl", "rb") as f:
        fragmentos = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    logger.info(f"Datos cargados: {len(fragmentos)} fragmentos")
except Exception as e:
    logger.error(f"Error cargando datos: {e}")
    fragmentos = []
    embeddings = []

# Función para buscar fragmentos relevantes
def buscar_fragmentos(pregunta, top_k=3):
    if not fragmentos or len(embeddings) == 0:
        return []
    
    # Crear embedding de la pregunta
    embedding_pregunta = modelo_embeddings.encode(pregunta)
    
    # Calcular similitudes
    similitudes = []
    for i, emb in enumerate(embeddings):
        similitud = np.dot(embedding_pregunta, emb) / (np.linalg.norm(embedding_pregunta) * np.linalg.norm(emb))
        similitudes.append((i, similitud))
    
    # Ordenar por similitud
    similitudes.sort(key=lambda x: x[1], reverse=True)
    
    # Devolver los top_k fragmentos
    return [fragmentos[i] for i, _ in similitudes[:min(top_k, len(similitudes))]]

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Asistente Opositores API",
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
        
        # Crear contexto
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
        
        return jsonify({
            "respuesta": respuesta_texto,
            "fragmentos_usados": len(fragmentos_relevantes)
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)