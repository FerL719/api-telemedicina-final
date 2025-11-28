import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pymongo import MongoClient

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()

app = FastAPI(
    title="API del Chatbot de Telemedicina",
    description="Backend optimizado con MongoDB Atlas y JSON Mode."
)

# --- 2. CONFIGURACIÓN DE CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. CONEXIÓN A MONGODB ATLAS ---
MONGO_URI = os.getenv('MONGO_URI')

# Verificamos que exista para no romper todo
if not MONGO_URI:
    raise ValueError("No se encontró la MONGO_URI en el archivo .env")

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["VirtualMedDB"]  # Base de datos correcta
    collection = db["doctor"]         # Colección correcta
    # Ping rápido para verificar conexión al iniciar
    mongo_client.admin.command('ping')
    print("✅ ¡Conexión exitosa a MongoDB Atlas!")
except Exception as e:
    print(f"❌ Error CRÍTICO conectando a Mongo: {e}")

# --- 4. CONFIGURACIÓN DE IA ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("No se encontró la GOOGLE_API_KEY en el .env")

genai.configure(api_key=GOOGLE_API_KEY)

# A. SAFETY SETTINGS
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# B. MODELO
generation_config = {
    "temperature": 0.5,
    "top_p": 1, 
    "max_output_tokens": 2048,
    "response_mime_type": "application/json" 
}

# Versión estable 2.5-flash
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- 5. MODELOS DE DATOS ---
class ChatInput(BaseModel):
    user_id: Optional[str] = "anonimo"
    mensaje: str
    contexto_medico: Optional[str] = "Ninguno"

class ChatOutput(BaseModel):
    respuesta: Dict[str, Any]

# --- 6. ENDPOINT DEL CHAT (Lógica con Mongo) ---
@app.post("/chat", response_model=ChatOutput)
async def handle_chat(input: ChatInput):
    try:
        # PASO A: Leer doctores de MongoDB en tiempo real
        # Esto reemplaza la lectura del CSV
        texto_doctores_mongo = ""
        try:
            cursor_doctores = collection.find({})
            for doc in cursor_doctores:
                # Mapeamos los campos de Mongo a texto para la IA
                mongo_id = str(doc.get("_id", "")) # ID único
                nombre = doc.get("nombre", "")
                apellido = doc.get("apellido", "")
                nombre_completo = f"{nombre} {apellido}"
                
                especialidad = doc.get("especialidad", "Medicina General")
                subespecialidad = doc.get("subespecialidad", "")
                
                info = f"- ID: {mongo_id} | Dr/a: {nombre_completo} | Esp: {especialidad}"
                if subespecialidad:
                    info += f" ({subespecialidad})"
                
                texto_doctores_mongo += info + "\n"
                
        except Exception as e_mongo:
            print(f"Error leyendo Mongo en request: {e_mongo}")
            texto_doctores_mongo = "Error al acceder a la base de datos de doctores."

        # PASO B: Prompt Base
        prompt = f"""
        Eres MediChat, un asistente médico de triaje inteligente.
        
        TUS DATOS (DOCTORES DISPONIBLES EN BASE DE DATOS):
        {texto_doctores_mongo}
        
        INPUT DEL USUARIO:
        - Historial/Contexto: {input.contexto_medico}
        - Mensaje actual: "{input.mensaje}"
        
        INSTRUCCIONES:
        1. Analiza si el usuario describe síntomas, dolores o dudas médicas.
        2. Si NO es tema médico, responde educadamente que no puedes ayudar.
        3. Si ES tema médico:
           - Identifica la especialidad necesaria.
           - Busca en la lista de doctores arriba quién es el MÁS adecuado.
           - Si no hay especialista exacto, sugiere uno afín o Medicina General.
           
        FORMATO DE RESPUESTA JSON (OBLIGATORIO):
        {{
            "es_medico": true/false,
            "mensaje_al_usuario": "Tu respuesta empática y clara aquí...",
            "recomendaciones": [
                {{
                    "id_doctor": "El ID exacto de la lista (mongo_id)",
                    "nombre": "Nombre del doctor",
                    "especialidad": "Su especialidad",
                    "motivo": "Breve razón de por qué este doctor sirve"
                }}
            ]
        }}
        """

        # Llamada a Gemini (Async)
        response = await model.generate_content_async(prompt)
        
        # Limpieza y parseo
        json_str = response.text.strip()
        parsed_response = json.loads(json_str)

        return ChatOutput(respuesta=parsed_response)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error de formato IA (No JSON).")
    except Exception as e:
        print(f"Error general: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. ENDPOINT DE PRUEBA ---
@app.get("/")
def read_root():

    return {"status": "Online", "database": "MongoDB Atlas Connected"}
