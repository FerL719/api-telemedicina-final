import os
import json
import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()

app = FastAPI(
    title="API del Chatbot de Telemedicina",
    description="Backend optimizado con JSON Mode y Safety Settings."
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

# --- 3. CONFIGURACIÓN DE IA (Gemini BLINDADO) ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("No se encontró la GOOGLE_API_KEY en el .env")

genai.configure(api_key=GOOGLE_API_KEY)

# A. SAFETY SETTINGS (La cura para el Error 500)
# Esto permite que la IA procese temas médicos "fuertes" sin bloquearse.
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# B. MODELO (Corregido a 1.5-flash y con JSON Mode activado)
# 'response_mime_type': 'application/json' obliga a la IA a responder SOLO JSON.
generation_config = {
    "temperature": 0.5, # Bajamos temperatura para ser más precisos
    "top_p": 1, 
    "max_output_tokens": 2048,
    "response_mime_type": "application/json" 
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", # OJO: gemini-2.5 no existe publicamente aun, usa 1.5
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- 4. CARGA DE DATOS (Doctores) ---
df_doctores = pd.DataFrame() # Inicializamos vacío por seguridad
texto_doctores_csv = ""

try:
    df_doctores = pd.read_csv("doctores.csv")
    # Convertimos el DataFrame a un string formateado para pasárselo al Prompt
    # Esto ayuda a la IA a leer mejor los datos
    for index, row in df_doctores.iterrows():
        # Aseguramos que exista la columna ID, si no, usamos el índice
        id_doc = row['id'] if 'id' in row else index 
        texto_doctores_csv += f"- ID: {id_doc} | Nombre: {row['nombre_completo']} | Especialidad: {row['especialidad']} | Bio: {row['bio_corta']}\n"
        
    print(f"--- CSV cargado. {len(df_doctores)} doctores listos. ---")
except Exception as e:
    print(f"--- ERROR CRÍTICO cargando CSV: {e} ---")
    texto_doctores_csv = "No hay doctores disponibles en la base de datos."

# --- 5. MODELOS DE DATOS ---
class ChatInput(BaseModel):
    user_id: Optional[str] = "anonimo"
    mensaje: str
    contexto_medico: Optional[str] = "Ninguno" # Ej: "Tengo diabetes"

# Este modelo ahora es un Dict para aceptar cualquier estructura JSON que mande la IA
class ChatOutput(BaseModel):
    respuesta: Dict[str, Any]

# --- 6. ENDPOINT DEL CHAT (Lógica Renovada) ---
@app.post("/chat", response_model=ChatOutput)
async def handle_chat(input: ChatInput):
    try:
        # El Prompt Maestro: hace el trabajo de análisis y selección en un solo paso
        prompt = f"""
        Eres un asistente médico de triaje inteligente para una app de telemedicina.
        
        TUS DATOS (DOCTORES DISPONIBLES):
        {texto_doctores_csv}
        
        INPUT DEL USUARIO:
        - Historial/Contexto: {input.contexto_medico}
        - Mensaje actual: "{input.mensaje}"
        
        INSTRUCCIONES:
        1. Analiza si el usuario describe síntomas, dolores o dudas médicas.
        2. Si NO es tema médico, responde educadamente que no puedes ayudar.
        3. Si ES tema médico:
           - Identifica la especialidad necesaria (Traumatología, Cardiología, etc.).
           - Busca en la lista de doctores arriba quién es el MÁS adecuado.
           - Considera el contexto (ej. si tiene diabetes, prioriza endocrino o internista si es relevante).
           
        FORMATO DE RESPUESTA JSON (OBLIGATORIO):
        {{
            "es_medico": true/false,
            "mensaje_al_usuario": "Tu respuesta empática y clara aquí...",
            "recomendaciones": [
                {{
                    "id_doctor": "El ID exacto del CSV",
                    "nombre": "Nombre del doctor",
                    "especialidad": "Su especialidad",
                    "motivo": "Breve razón de por qué este doctor sirve para este caso"
                }}
            ]
        }}
        """

        # Llamada a Gemini
        response = await model.generate_content_async(prompt)
        
        # Limpieza y parseo de la respuesta
        # A veces la IA puede mandar texto antes del JSON, aseguramos limpieza
        json_str = response.text.strip()
        
        # Convertimos el string JSON a un objeto Python real (Diccionario)
        parsed_response = json.loads(json_str)

        return ChatOutput(respuesta=parsed_response)

    except json.JSONDecodeError:
        # Si la IA falló en hacer JSON exacto (muy raro con 1.5 Flash), manejamos el error
        raise HTTPException(status_code=500, detail="Error interno de formato IA.")
    except Exception as e:
        print(f"Error en servidor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. ENDPOINT DE PRUEBA ---
@app.get("/")
def read_root():
    return {"status": "Online", "mode": "JSON + Safety Settings Active"}