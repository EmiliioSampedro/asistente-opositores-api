import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

print("🔧 OPTIMIZADOR DE FRAGMENTOS")
print("="*50)

# 1. Cargar el modelo (el mismo que usa app.py)
print("🔄 Cargando modelo MiniLM L3...")
modelo = SentenceTransformer('paraphrase-MiniLM-L3-v2')
print("✅ Modelo cargado")

# 2. Leer todos los documentos .txt
carpeta_docs = "documentos"
todos_fragmentos = []

if not os.path.exists(carpeta_docs):
    print(f"❌ No existe la carpeta {carpeta_docs}")
    exit()

archivos_txt = [f for f in os.listdir(carpeta_docs) if f.endswith('.txt')]
print(f"\n📚 Procesando {len(archivos_txt)} archivos...")

for archivo in archivos_txt:
    ruta = os.path.join(carpeta_docs, archivo)
    print(f"\n  📄 {archivo}:")
    
    with open(ruta, 'r', encoding='utf-8') as f:
        texto = f.read()
    
    # División INTELIGENTE por párrafos y oraciones
    # Primero por párrafos (doble salto de línea)
    parrafos = texto.split('\n\n')
    
    fragmentos_archivo = []
    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            continue
            
        # Si el párrafo es muy largo, dividir por oraciones
        if len(parrafo) > 500:
            # Dividir por puntos, interrogaciones, exclamaciones
            import re
            oraciones = re.split(r'(?<=[.!?])\s+', parrafo)
            
            fragmento_actual = ""
            for oracion in oraciones:
                if len(fragmento_actual) + len(oracion) < 400:
                    fragmento_actual += " " + oracion if fragmento_actual else oracion
                else:
                    if fragmento_actual:
                        fragmentos_archivo.append(fragmento_actual.strip())
                    fragmento_actual = oracion
            
            if fragmento_actual:
                fragmentos_archivo.append(fragmento_actual.strip())
        else:
            # Párrafo corto, se queda como está
            fragmentos_archivo.append(parrafo)
    
    print(f"    → {len(fragmentos_archivo)} fragmentos generados")
    todos_fragmentos.extend(fragmentos_archivo)

print(f"\n📊 TOTAL: {len(todos_fragmentos)} fragmentos")

# 3. Crear embeddings
print("\n🧠 Generando embeddings (puede tomar unos segundos)...")
embeddings = modelo.encode(todos_fragmentos)
print("✅ Embeddings generados")

# 4. Guardar los nuevos archivos
print("\n💾 Guardando fragmentos.pkl...")
with open("fragmentos.pkl", "wb") as f:
    pickle.dump(todos_fragmentos, f)

print("💾 Guardando embeddings.pkl...")
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("\n📝 Ejemplo de fragmentos guardados:")
for i, frag in enumerate(todos_fragmentos[:3]):
    print(f"\n  {i+1}. {frag[:100]}...")

print("\n" + "="*50)
print("🎉 ¡OPTIMIZACIÓN COMPLETADA!")
print(f"✅ {len(todos_fragmentos)} fragmentos listos")
print("✅ Ya puedes subir los archivos a GitHub")