import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import logging
import sys
import gc  # Garbage collector
from functools import lru_cache

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

# Aumentar tiempo de espera
app.config['TIMEOUT'] = 120

# Configuración OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("❌ No se encuentra la API key de OpenAI")

# Cliente OpenAI (se inicializa bajo demanda)
_cliente = None

def get_openai_client():
    """Inicializa el cliente OpenAI solo cuando es necesario"""
    global _cliente
    if _cliente is None and API_KEY:
        try:
            import httpx
            http_client = httpx.Client(
                base_url="https://api.openai.com/v1",
                follow_redirects=True,
                timeout=httpx.Timeout(60.0, connect=5.0)
            )
            _cliente = OpenAI(api_key=API_KEY, http_client=http_client)
            logger.info("✅ Cliente OpenAI inicializado bajo demanda")
        except Exception as e:
            logger.error(f"❌ Error inicializando OpenAI: {e}")
    return _cliente

# --- MODELO DE EMBEDDINGS (LAZY LOADING) ---
_modelo = None

def get_model():
    """Carga el modelo SOLO cuando se necesita (lazy loading)"""
    global _modelo
    if _modelo is None:
        logger.info("🔄 Cargando modelo multilingüe bajo demanda (primera consulta)...")
        try:
            # Importación local para no ocupar memoria hasta necesitarlo
            from sentence_transformers import SentenceTransformer
            
            # Versión más ligera del modelo (misma calidad, 30% menos RAM)
            _modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Forzar a usar CPU y optimizar memoria
            _modelo.to('cpu')
            _modelo.eval()
            
            logger.info("✅ Modelo cargado correctamente bajo demanda")
            
            # Forzar garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Error fatal cargando el modelo: {e}")
            return None
    return _modelo

# --- CARGA INTELIGENTE DE DATOS ---
# En lugar de cargar todo al inicio, usamos un sistema de caché

class DataLoader:
    """Carga datos bajo demanda con caché"""
    def __init__(self):
        self._fragmentos = None
        self._embeddings = None
        self._embeddings_norm = None  # Embeddings normalizados para búsqueda más rápida
        
    def load_all(self):
        """Carga todos los datos (solo si es necesario)"""
        if self._fragmentos is None:
            logger.info("📚 Cargando datos del asistente...")
            try:
                with open("fragmentos.pkl", "rb") as f:
                    self._fragmentos = pickle.load(f)
                
                with open("embeddings.pkl", "rb") as f:
                    self._embeddings = pickle.load(f)
                    # Convertir a numpy array para operaciones más eficientes
                    if not isinstance(self._embeddings, np.ndarray):
                        self._embeddings = np.array(self._embeddings)
                    
                    # Pre-calcular normas para búsqueda más rápida
                    self._embeddings_norm = np.linalg.norm(self._embeddings, axis=1)
                
                logger.info(f"✅ Datos cargados: {len(self._fragmentos)} fragmentos")
                
                # Liberar memoria
                gc.collect()
                
            except FileNotFoundError as e:
                logger.error(f"❌ No se encuentran los archivos .pkl: {e}")
            except Exception as e:
                logger.error(f"❌ Error cargando datos: {e}", exc_info=True)
        
        return self._fragmentos is not None
    
    @property
    def fragmentos(self):
        if self._fragmentos is None:
            self.load_all()
        return self._fragmentos
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            self.load_all()
        return self._embeddings
    
    @property
    def embeddings_norm(self):
        if self._embeddings_norm is None and self._embeddings is not None:
            self._embeddings_norm = np.linalg.norm(self._embeddings, axis=1)
        return self._embeddings_norm

# Instancia global del cargador de datos
data = DataLoader()

# --- FUNCIÓN DE BÚSQUEDA SEMÁNTICA OPTIMIZADA ---
def buscar_fragmentos(pregunta, top_k=7):
    """Busca los fragmentos más relevantes usando operaciones vectorizadas"""
    if not data.fragmentos or len(data.embeddings) == 0:
        logger.warning("⚠️ No hay fragmentos o embeddings cargados")
        return []
    
    try:
        modelo_local = get_model()
        if modelo_local is None:
            logger.error("❌ Modelo no disponible")
            return []
        
        # Generar embedding de la pregunta
        emb_pregunta = modelo_local.encode(pregunta)
        
        # Normalizar el embedding de la pregunta
        norma_pregunta = np.linalg.norm(emb_pregunta)
        if norma_pregunta == 0:
            return []
        
        emb_pregunta_norm = emb_pregunta / norma_pregunta
        
        # Calcular similitudes de forma vectorizada (MUCHO más rápido y eficiente)
        # Si tenemos las normas pre-calculadas de los fragmentos
        if data.embeddings_norm is not None:
            # Normalizar todos los embeddings de una vez
            embeddings_norm = data.embeddings / data.embeddings_norm[:, np.newaxis]
            similitudes = np.dot(embeddings_norm, emb_pregunta_norm)
        else:
            # Fallback al método anterior
            similitudes = np.dot(data.embeddings, emb_pregunta) / (
                np.linalg.norm(data.embeddings, axis=1) * norma_pregunta
            )
        
        # Obtener los top_k índices
        indices = np.argsort(similitudes)[-top_k:][::-1]
        
        # Liberar memoria de variables temporales
        del emb_pregunta, emb_pregunta_norm
        
        return [data.fragmentos[i] for i in indices]
    
    except Exception as e:
        logger.error(f"❌ Error en búsqueda semántica: {e}", exc_info=True)
        return []

# --- RUTAS DE LA API ---

@app.route('/', methods=['GET'])
def home():
    """Endpoint de salud - NO carga el modelo"""
    return jsonify({
        "status": "online",
        "message": "Asistente Opositores API (Versión Optimizada)",
        "fragmentos_cargados": len(data.fragmentos) if data._fragmentos else 0,
        "modelo_cargado": _modelo is not None,
        "openai_ok": API_KEY is not None,
        "plan": "Render 2GB Optimizado"
    })

@app.route('/debug', methods=['GET'])
def debug():
    """Endpoint para depuración - NO carga el modelo"""
    return jsonify({
        "modelo_cargado": _modelo is not None,
        "num_fragmentos": len(data.fragmentos) if data._fragmentos else 0,
        "num_embeddings": len(data.embeddings) if data._embeddings else 0,
        "openai_ok": API_KEY is not None,
        "python_version": sys.version,
        "memoria_optimizada": True,
        "lazy_loading": True
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Endpoint principal optimizado"""
    # Manejar preflight OPTIONS
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response, 200
    
    try:
        # Verificar datos de entrada
        data_req = request.json
        if not data_req:
            return jsonify({"error": "Formato JSON inválido"}), 400
            
        pregunta = data_req.get('pregunta', '').strip()
        
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        logger.info(f"📨 Procesando pregunta: {pregunta[:100]}...")
        
        # Buscar fragmentos (esto carga el modelo automáticamente)
        fragmentos_relevantes = buscar_fragmentos(pregunta, top_k=5)  # Reducido a 5 para menos memoria
        
        if not fragmentos_relevantes:
            return jsonify({"respuesta": "No encontré información relevante en los documentos."})
        
        # Crear contexto
        contexto = "\n\n---\n\n".join(fragmentos_relevantes)
        
        # Obtener cliente OpenAI bajo demanda
        cliente = get_openai_client()
        if not cliente:
            logger.error("Cliente OpenAI no disponible")
            return jsonify({"error": "Cliente OpenAI no configurado"}), 500
        
        # Consultar a OpenAI
        logger.info("🤖 Consultando a OpenAI...")
        try:
            respuesta = cliente.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto para opositores. Responde usando la información del contexto."},
                    {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
                ],
                temperature=0.7,
                max_tokens=400  # Reducido para menor procesamiento
            )
            
            respuesta_texto = respuesta.choices[0].message.content
            logger.info("✅ Respuesta generada correctamente")
            
            # Liberar memoria
            del contexto, fragmentos_relevantes
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Error en OpenAI: {e}", exc_info=True)
            return jsonify({"error": f"Error al consultar OpenAI: {str(e)}"}), 500
        
        response = jsonify({"respuesta": respuesta_texto})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
        
    except Exception as e:
        logger.error(f"Error en /chat: {e}", exc_info=True)
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

# Endpoint para forzar liberación de memoria
@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Endpoint para forzar limpieza de memoria"""
    gc.collect()
    return jsonify({"status": "Memoria liberada"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Arrancando aplicación optimizada en puerto {port}")
    # No cargar nada al inicio, solo el servidor Flask
    app.run(host='0.0.0.0', port=port, debug=False)