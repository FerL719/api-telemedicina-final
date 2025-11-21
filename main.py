import os
import pandas as pd  # Importamos PANDAS
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  # Importamos CORS

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()

# Configura la app FastAPI
app = FastAPI(
    title="API del Chatbot de Telemedicina",
    description="Backend para el chatbot RAG y el sistema de alertas."
)

# --- 2. CONFIGURACIÓN DE CORS (¡MUY IMPORTANTE!) ---
# Esto permite que la app de Frontend (HTML/JS)
# pueda hacerle peticiones a tu Backend (esta API)
origins = ["*"]  # Permite que CUALQUIER sitio web llame a tu API

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc)
    allow_headers=["*"],  # Permite todos los encabezados
)

# --- 3. CONFIGURACIÓN DE IA (Gemini) ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("No se encontró la GOOGLE_API_KEY en el .env")
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {"temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = genai.GenerativeModel(model_name="gemini-2.5-flash", generation_config=generation_config)

# --- 4. CARGA ÚNICA DE DATOS (El gran cambio) ---
# Leemos el CSV al arrancar el servidor y lo guardamos en un DataFrame de Pandas
try:
    # Asegúrate que tu archivo se llama 'doctores.csv'
    df_doctores = pd.read_csv("doctores.csv")
    print(f"--- CSV 'doctores.csv' cargado exitosamente. {len(df_doctores)} doctores encontrados. ---")
except FileNotFoundError:
    print("--- ERROR: No se encontró el archivo 'doctores.csv'. El bot funcionará sin datos de doctores. ---")
    # Creamos un DataFrame vacío para que el código no falle más adelante
    df_doctores = pd.DataFrame(columns=["nombre_completo", "especialidad", "bio_corta"])
except Exception as e:
    print(f"--- ERROR al cargar 'doctores.csv': {e} ---")
    df_doctores = pd.DataFrame(columns=["nombre_completo", "especialidad", "bio_corta"])


# --- 5. PROMPTS (Igual que antes) ---
PROMPT_SISTEMA_GENERAL = """
Eres 'SaludBot', un asistente de telemedicina. Tu único propósito es ayudar a los usuarios
a identificar posibles causas de sus síntomas y conectarlos con doctores de la plataforma.
Tus reglas son estrictas:
1. NUNCA des un diagnóstico definitivo.
2. Si la información es vaga (ej. "me duele"), DEBES hacer preguntas de seguimiento.
3. Si el usuario pregunta algo no médico, responde: "Lo siento, mi función es solo ayudarte con consultas de salud."
"""

PROMPT_EXTRACTOR_ESPECIALIDAD = """
Analiza el siguiente mensaje de un usuario. Responde ÚNICAMENTE con la especialidad médica más relevante
para los síntomas descritos. Si NO es una consulta médica o no está claro, responde solo con "N/A".
Ejemplos:
Usuario: "Me duele la cabeza y veo luces." -> Respuesta: Neurología
Usuario: "Creo que me rompí el brazo." -> Respuesta: Traumatología
Usuario: "Tengo mucha tos y fiebre." -> Respuesta: Medicina General
Mensaje del usuario: "{mensaje_usuario}"
Respuesta (solo la especialidad o N/A):
"""

# --- 6. MODELOS DE DATOS (Igual que antes) ---
class ChatInput(BaseModel):
    user_id: str
    mensaje: str

class ChatOutput(BaseModel):
    respuesta_bot: str

# --- 7. ENDPOINT DEL CHAT (Lógica RAG modificada) ---
@app.post("/chat", response_model=ChatOutput)
async def handle_chat(input: ChatInput):
    try:
        # --- PASO A: PRE-ANÁLISIS (1ra Llamada a IA) ---
        prompt_especialidad = PROMPT_EXTRACTOR_ESPECIALIDAD.format(mensaje_usuario=input.mensaje)
        response_especialidad = await model.generate_content_async(prompt_especialidad)
        especialidad = response_especialidad.text.strip()

        # --- PASO B: MANEJAR TEMAS NO MÉDICOS ---
        if especialidad == "N/A":
            return ChatOutput(respuesta_bot="Lo siento, mi función es solo ayudarte con consultas de salud. ¿Tienes algún síntoma del que quieras hablarme?")

        # --- PASO C: BÚSQUEDA (Retrieval) - ¡LA PARTE MODIFICADA! ---
        # Buscamos en el DataFrame de Pandas (en memoria), no en la BD.
        # Nos aseguramos de que ambas cadenas (la del CSV y la del LLM) estén "limpias".
        if df_doctores.empty:
            lista_doctores_filtrados = []
        else:
            resultados = df_doctores[df_doctores['especialidad'].str.strip().str.lower() == especialidad.strip().str.lower()]
            # Convertimos los resultados de Pandas a una lista de diccionarios
            lista_doctores_filtrados = resultados.head(3).to_dict('records') # .head(3) limita a 3 doctores

        contexto_doctores = "No se encontraron doctores para esta especialidad."
        if lista_doctores_filtrados:
            contexto_doctores = "Doctores disponibles encontrados:\n"
            for doc in lista_doctores_filtrados:
                contexto_doctores += f"- {doc['nombre_completo']} ({doc['especialidad']}): {doc['bio_corta']}\n"

        # --- PASO D: GENERACIÓN (2da Llamada a IA) ---
        prompt_final = f"""
        {PROMPT_SISTEMA_GENERAL}
        ---
        Contexto (Doctores encontrados en nuestro archivo CSV):
        {contexto_doctores}
        ---
        Mensaje del usuario:
        "{input.mensaje}"
        ---
        Tu respuesta (recuerda tus reglas, no diagnostiques, y si los doctores son relevantes, menciónalos):
        """

        response_final = await model.generate_content_async(prompt_final)
        
        return ChatOutput(respuesta_bot=response_final.text.strip())

    except Exception as e:
        print(f"Error en el endpoint /chat: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error al procesar tu solicitud.")

# --- 8. ENDPOINT DE BIENVENIDA (Para probar) ---
@app.get("/")
def read_root():
    return {"mensaje": "API del Chatbot de Telemedicina funcionando (Modo CSV)"}