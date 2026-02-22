import os
# Matar proxies (por si acaso)
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# INTERVENCIÓN DIRECTA: Parcheamos la clase OpenAI ANTES de importarla
import sys
import types

# Crear un módulo falso que intercepte la importación
class OpenAIPatcher:
    def __init__(self):
        self._real_openai = None
    
    def __getattr__(self, name):
        if self._real_openai is None:
            # Importar el real solo cuando sea necesario
            import openai as real_openai
            self._real_openai = real_openai
            # Parchear después de importar
            original_init = self._real_openai.OpenAI.__init__
            def patched_init(self_obj, *args, **kwargs):
                if 'proxies' in kwargs:
                    print(f"🔪 Matando proxies: {kwargs['proxies']}")
                    del kwargs['proxies']
                original_init(self_obj, *args, **kwargs)
            self._real_openai.OpenAI.__init__ = patched_init
        return getattr(self._real_openai, name)

# Sobrescribir el módulo 'openai' en sys.modules ANTES de que se importe
sys.modules['openai'] = OpenAIPatcher()

# AHORA importamos todo lo demás (incluyendo openai, que usará el parche)
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import warnings
import logging
import sys as sys_module
import gc
from openai import OpenAI as OpenAIClient  # Esto usará nuestro parche

warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)

# Configuración CORS
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["POST", "OPTIONS", "GET"])
logger.info("✅ CORS configurado")

# Configuración OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("❌ No se encuentra la API key de OpenAI")
    cliente_openai = None
else:
    try:
        # La forma MÁS SIMPLE posible - sin argumentos extra
        cliente_openai = OpenAIClient(api_key=API_KEY)
        logger.info("✅ Cliente OpenAI inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando OpenAI: {e}")
        cliente_openai = None

def get_openai_client():
    return cliente_openai

# Inicializar cliente al arrancar
if API_KEY:
    get_openai_client()

# --- CARGA DEL MODELO ---
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
            logger.error(f"❌ Error: {e}")
    return _modelo

# --- CARGA DE FRAGMENTOS Y EMBEDDINGS ---
logger.info("📚 CARGANDO DATOS AL INICIO...")
fragmentos = []
embeddings = []

# Verificar que los archivos existen
if os.path.exists("fragmentos.pkl") and os.path.exists("embeddings.pkl"):
    logger.info("✅ Archivos .pkl encontrados")
    try:
        with open("fragmentos.pkl", "rb") as f:
            fragmentos = pickle.load(f)
        logger.info(f"   - fragmentos.pkl cargado: {len(fragmentos)} fragmentos")
        
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f"   - embeddings.pkl cargado: {len(embeddings)} embeddings")
        
        # Convertir a numpy array si es necesario
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            logger.info("   - embeddings convertidos a numpy array")
            
    except Exception as e:
        logger.error(f"❌ Error cargando archivos .pkl: {e}", exc_info=True)
else:
    logger.error("❌ No se encuentran los archivos .pkl en el directorio actual")
    logger.info(f"   Directorio actual: {os.getcwd()}")
    logger.info(f"   Archivos presentes: {os.listdir('.')}")

# --- FUNCIÓN DE BÚSQUEDA SEMÁNTICA ---
def buscar_fragmentos(pregunta, top_k=5):
    if not fragmentos or len(embeddings) == 0:
        logger.warning("⚠️ No hay datos cargados")
        return []
    
    try:
        modelo = get_model()
        if modelo is None:
            return []
        
        emb_pregunta = modelo.encode(pregunta)
        
        # Calcular similitudes (producto punto normalizado)
        similitudes = []
        for i, emb in enumerate(embeddings):
            sim = np.dot(emb_pregunta, emb) / (np.linalg.norm(emb_pregunta) * np.linalg.norm(emb))
            similitudes.append((i, sim))
        
        similitudes.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in similitudes[:top_k]]
        
        logger.info(f"🔍 Top similitud: {similitudes[0][1]:.4f}")
        return [fragmentos[i] for i in indices]
    
    except Exception as e:
        logger.error(f"❌ Error en búsqueda: {e}", exc_info=True)
        return []

# --- RUTAS DE LA API ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Asistente Opositores API (Versión Definitiva)",
        "fragmentos_cargados": len(fragmentos),
        "modelo_cargado": _modelo is not None,
        "openai_ok": API_KEY is not None,
        "plan": "Render 2GB",
        "parche_activo": "✅"
    })

@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        "fragmentos": len(fragmentos),
        "embeddings": len(embeddings) if isinstance(embeddings, list) else embeddings.shape[0] if hasattr(embeddings, 'shape') else 0,
        "modelo_cargado": _modelo is not None,
        "archivos": os.listdir('.')[:10]
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Formato JSON inválido"}), 400
            
        pregunta = data.get('pregunta', '').strip()
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        logger.info(f"📨 Pregunta: {pregunta[:100]}...")
        
        fragmentos_relevantes = buscar_fragmentos(pregunta, top_k=5)
        
        if not fragmentos_relevantes:
            return jsonify({"respuesta": "No encontré información relevante."})
        
        contexto = "\n\n---\n\n".join(fragmentos_relevantes)
        
        # Limpiar memoria antes de usar OpenAI
        gc.collect()
        
        cliente = get_openai_client()
        if not cliente:
            return jsonify({"error": "OpenAI no disponible"}), 500
        
        respuesta = cliente.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto para opositores. Responde usando la información del contexto."},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        return jsonify({"respuesta": respuesta.choices[0].message.content})
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Arrancando aplicación definitiva en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)