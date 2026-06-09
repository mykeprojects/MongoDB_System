# Documentación técnica del proyecto RAG - MongoDB_System

## Visión general

Proyecto RAG (Retrieval-Augmented Generation) orientado a: ingestión de documentos, generación de embeddings, almacenamiento en MongoDB y exposición de un backend HTTP (Flask) para preguntas y respuestas que combinan contexto recuperado y un LLM (Groq o fallback local).

Estructura principal:

- `backend/`: API y lógica del servidor.
- `data/`: activos y almacén local de imágenes.
- `frontend/`: interfaz de chat estática (HTML/JS/CSS).
- `scripts/`: utilidades para ingesta, configuración y análisis de embeddings.
- `requirements.txt`: dependencias.

---

## Configuración y variables de entorno

El proyecto usa `python-dotenv`. Archivo `.env` (no incluido) debe contener al menos:

- `MONGO_URI`: URI de conexión a MongoDB.
- `DATABASE_NAME`: nombre de la base de datos.
- `EMBEDDING_MODEL` (opcional): modelo de sentence-transformers (por defecto `all-MiniLM-L6-v2`).
- `EMBEDDING_DIMENSIONS` (opcional): dimensiones del embedding (por defecto `384`).
- `GROQ_API_KEY` (opcional): clave para usar Groq; si no está presente se usa la lógica local de `LLMService`.
- `GROQ_MODEL` (opcional): modelo Groq a usar.
- `CORS_ORIGINS` (opcional): orígenes permitidos para CORS.
- `RAG_RETRIEVAL_LIMIT`, `RAG_CHUNK_STRATEGY`, `MONGO_VECTOR_INDEX` (opcional) para parámetros de recuperación.

---

## Backend (`backend/`)

### `app.py`
- Punto de arranque de la aplicación Flask.
- Funciones/acción principal:
  - `create_app() -> Flask`:
    - Carga configuración con `load_config()` (desde `config_service`).
    - Inicializa servicios: `DatabaseService`, `EmbeddingService`, `RetrievalService`, `ImageService`, `LLMService`, `ChatService`.
    - Registra blueprint `chat_blueprint` y configura rutas:
      - `GET /` : devuelve un JSON con endpoints disponibles.
      - `GET /data/images/<filename>`: sirve imágenes desde `data/images`.
    - Añade `app.config` con instancias `APP_CONFIG`, `DATABASE_SERVICE`, `CHAT_SERVICE` usadas por controladores.
  - Si se ejecuta como script, arranca en `0.0.0.0:8000` con `debug=True`.

### `controllers/chatController.py`
- Define el blueprint `chat_blueprint` con prefijo `/api`.
- Endpoints:
  - `POST /api/chat`:
    - Recibe payload JSON, construye `ChatRequestDTO.from_payload`.
    - Llama a `current_app.config["CHAT_SERVICE"].handle_chat(dto)`.
    - Retorna `ChatResponseDTO.to_dict()` con status 200.
    - Maneja `ValueError` para validación (400) y errores generales (500).
  - `GET /api/health`:
    - Llama a `database.ping()` y devuelve estado de la app, indicador de Groq, configuración de recuperación y nombre de índice vector.

### `models/chat_dto.py`
- DTOs usados por la API:
  - `ChatRequestDTO` (dataclass)
    - Campos: `message: str`, `image_path: str | None`.
    - `from_payload(payload)` : normaliza claves (`imagePath` o `image_path`) y limpia strings.
    - `validate()` : lanza `ValueError` si ni `message` ni `image_path` están presentes.
  - `ChatResponseDTO` (dataclass)
    - Campos: `response: str`, `image_path: str | None`, `sources: list[dict]`, `mode: str`.
    - `to_dict()` : convierte `image_path` a `imagePath` y devuelve dict serializable.

### `models/rag_models.py`
- Clases ligeras para resultados de recuperación:
  - `RetrievalHit`:
    - Campos: `collection`, `title`, `text`, `score`, `chunk_index`, `strategy`, `resource_id`, `resource_type`, `metadata`.
    - `to_source()` : transforma el hit a un dict con campos expuestos en la respuesta.
  - `ImageMatch`:
    - Campos: `path`, `title`, `score`.

### `services/config_service.py`
- Define rutas base y carga configuración del entorno.
- `AppConfig` (dataclass) con opciones usadas por servicios.
- `load_config()` : lee variables desde `.env` y construye `AppConfig`.

### `services/database_service.py`
- `DatabaseService`:
  - Encapsula `MongoClient` y selección de base.
  - Propiedades:
    - `client`: inicializa `MongoClient` si es necesario, lanza error si `MONGO_URI` no está definido.
    - `db`: devuelve `client[database_name]` y valida `DATABASE_NAME`.
  - Métodos:
    - `ping()` : ejecuta `client.admin.command("ping")`.
    - `close()` : cierra el cliente.

### `services/embedding_service.py`
- `EmbeddingService`:
  - Carga de forma perezosa `SentenceTransformer` (thread-safe con `Lock`).
  - `encode_one(text) -> np.ndarray` : codifica un texto a vector numpy (dtype float).
  - `cosine_similarity(a,b) -> float` : similaridad coseno segura (manejo norma cero).

### `services/image_service.py`
- `ImageService` para manejo de imágenes locales:
  - `resolve_public_path(image_path)` : normaliza rutas y convierte a `data/images/<name>`.
  - `local_file_exists(image_path)` : comprueba existencia en `IMAGES_DIR`.
  - `describe_image_path(image_path)` : extrae título heurístico (stem del nombre).
  - `find_best_local_image(query, fallback)` : compara tokens entre `query` y nombres de archivos y devuelve `ImageMatch` con score; usa fallback si no encuentra.

### `services/llm_service.py`
- `LLMService` maneja generación de respuestas:
  - `generate(question, hits, image_context)` : construye contexto y si `GROQ_API_KEY` está presente llama `_generate_with_groq`, si no, cae a `_fallback_answer`.
  - `_generate_with_groq(question, context)` : realiza una petición HTTP a la API de Groq con el `chat/completions` payload.
  - `_build_context(hits, image_context)` : formatea los `RetrievalHit` para incluirlos en la petición al LLM.
  - `_fallback_answer(question, hits, image_context)` : lógica local para devolver una respuesta razonable cuando no hay LLM disponible (usa el primer hit o lista de hits en modo keyword-fallback).

### `services/retrieval_service.py`
- `RetrievalService` encapsula las estrategias de búsqueda sobre MongoDB:
  - `EMBEDDING_COLLECTIONS` : mapea colecciones de embeddings (`embedding_normatividad`, `embedding_productos`) a campos y tipos.
  - `search(query, limit, strategy)` : flujo principal:
    1. Si `query` vacía → return []
    2. Calcula `query_vector` usando `EmbeddingService.encode_one`.
    3. Intenta `_search_with_vector_search` (usa `$vectorSearch` / pipeline de MongoDB Atlas si está disponible).
    4. Si no hay resultados, intenta `_search_with_cosine_similarity` (iterando documentos y calculando similitud localmente).
    5. Si todavía no hay resultados, hace `_fallback_keyword_search` que usa búsquedas regex simples sobre `productos` y `normatividades` y crea `RetrievalHit` con score bajo y `strategy = "keyword-fallback"`.
  - `_vector_search_collection(...)` : arma pipeline con `$vectorSearch` y proyecta campos esperados para crear `RetrievalHit`.
  - `_search_with_cosine_similarity(...)` : recupera embeddings almacenados y calcula similitud coseno con `EmbeddingService.cosine_similarity`.
  - `get_product_image(product_id)` : busca en `productos` por `_id` y devuelve la primera URL en `imagenesUrl`.

---

## Scripts (`scripts/`)

Los scripts proveen utilidades de preparación de datos, ingesta y análisis experimental:

- `setup_mongodb.py`:
  - Crea colecciones con validadores JSON Schema y crea índices importantes.
  - `create_vector_search_indexes()` : intenta crear índices de Vector Search (Atlas) para `embedding_normatividad` y `embedding_productos` con la definición de vector.
  - Ejecutar para inicializar la base de datos y los índices requeridos.

- `ingestdocuments.py`:
  - Interfaz (GUI minimal con `tkinter`) para seleccionar un documento (.txt o .pdf), realizar chunking (por frases y semántico), generar embeddings con `SentenceTransformer` y guardar chunks en las colecciones `embedding_normatividad` o `embedding_productos`.
  - Funciones principales: `dividir_en_frases`, `chunking_por_frases`, `chunking_semantico`, `generar_embeddings`, `construir_documentos`, `guardar_chunks`.

- `ingestdb.py`:
  - Generador de datos sintéticos para poblar varias colecciones (`normatividades`, `productos`, `usuarios`, `ordenes`, etc.).
  - Útil para pruebas y para tener documentos textuales con los que el pipeline RAG pueda trabajar.

- `comparatechunks.py`:
  - Script de evaluación comparativa entre estrategias de chunking (`frases` vs `semantico`).
  - Carga chunks, genera embeddings para un conjunto de consultas predefinidas y calcula métricas (similitud promedio, medianas, victorias por consulta) y guarda un reporte JSON en `scripts/reportes/`.

- `environment_config.py`:
  - Script de verificación de conexión a MongoDB: hace `ping`, `server_info`, lista bases de datos y loggea información útil para diagnóstico.

---

## Frontend (`frontend/`)

- `views/index.html` : interfaz de chat estática. Elementos principales:
  - Panel de mensajes, panel de respuesta, control para adjuntar imagen y textarea para el mensaje.
  - Carga `../js/fetching.js` y `../js/main.js`.

- `js/fetching.js`:
  - `API_CONFIG` con `baseUrl` y `endpoints`.
  - `sendChatMessage(payload)` : realiza `fetch` POST a `POST /api/chat` y normaliza la respuesta.
  - `resolveImageUrl(imagePath)` : convierte rutas locales `data/images/...` a URLs cargables (concatena con `API_CONFIG.baseUrl`).

- `js/main.js`:
  - Lógica de interacción: captura archivos desde `<input type=file>`, muestra vista previa local, envía payload con `message` y `imagePath` y renderiza mensajes en la UI.
  - Añade manejo de estado (loading), control de errores y actualización del DOM.

- `css/index.css` : estilos (archivo presente pero no analizado en detalle aquí).

---

## Dependencias principales

Revisar `requirements.txt`. Las más relevantes:
- `Flask`, `flask-cors` (API)
- `pymongo` (MongoDB)
- `python-dotenv` (config)
- `sentence-transformers`, `numpy` (embeddings)
- `groq` (cliente Groq) y `requests` / urllib para comunicarse con LLM externo
- `pypdf`, `Faker` para scripts de ingesta

---

## Flujo de ejecución (resumen)

1. Ejecutar `python backend/app.py` (o usar `flask`/gunicorn) para arrancar API en el puerto 8000.
2. El backend carga configuración y crea instancias de servicios. `ChatService` es el punto de entrada para consultas.
3. Frontend estático envía `POST /api/chat` con `{"message": "...", "imagePath": "data/images/.."}`.
4. `ChatService.handle_chat` valida DTO, construye consulta (combina texto y descripción de imagen), llama a `RetrievalService.search` para obtener `RetrievalHit`s.
5. `LLMService.generate` crea la respuesta final (usando Groq si está configurado, o lógica fallback local).
6. Respuesta JSON contiene `response`, `imagePath` (si aplica), `sources` (lista de fragments) y `mode`.

---

## Puntos de atención y recomendaciones

- `.env` crítico: sin `MONGO_URI` y `DATABASE_NAME` muchos servicios fallarán.
- Indexes Vector Search: la funcionalidad con `$vectorSearch` requiere Atlas o instancia con soporte de vector search; si no está disponible el código cae a cálculo local por similitud.
- Seguridad: la app actual expone CORS configurable; producción debe limitar `CORS_ORIGINS` y usar HTTPS.
- LLM externo (Groq): la integración está en `LLMService._generate_with_groq`; fallbacks y timeouts deben manejarse en despliegue.
- Datos de prueba y scripts (`ingestdb.py`) generan muchos documentos útiles para pruebas; limpiar/validar antes de usar en entornos reales.

---

## Archivos clave (lista rápida)

- `backend/app.py` — inicio y wiring de servicios.
- `backend/controllers/chatController.py` — endpoints `/api/chat` y `/api/health`.
- `backend/models/chat_dto.py` — DTOs de request/response.
- `backend/models/rag_models.py` — `RetrievalHit`, `ImageMatch`.
- `backend/services/*.py` — implementación de `DatabaseService`, `EmbeddingService`, `RetrievalService`, `ImageService`, `LLMService`, `ChatService`, `config_service`.
- `scripts/setup_mongodb.py` — crea colecciones y índices.
- `scripts/ingestdocuments.py` — UI para ingesta y chunking.
- `scripts/ingestdb.py` — generador de datos sintéticos.
- `scripts/comparatechunks.py` — script de evaluación de estrategias de chunking.
- `frontend/views/index.html`, `frontend/js/*` — UI cliente.

---

Fin de la documentación.
