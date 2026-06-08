import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("comparate_chunks")

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLECCION = "embedding_normatividad"
MODEL_NAME = "all-MiniLM-L6-v2"

if not MONGO_URI or not DATABASE_NAME:
    raise ValueError("Las variables MONGO_URI y DATABASE_NAME deben estar definidas en el archivo .env")

ESTRATEGIAS = {
    "sentences": "frases",
    "semantic": "semantico",
}

CONSULTAS = [
    (
        "¿Cuáles son los canales y el plazo máximo legal que tiene un responsable del "
        "tratamiento para responder un reclamo por corrección o supresión de datos "
        "según la Ley 1581 de 2012?"
    ),
    (
        'Defina el concepto de "Dato Sensible" bajo la normatividad colombiana y '
        "especifique en qué casos excepcionales está permitido su tratamiento sin "
        "autorización explícita."
    ),
    (
        "¿Qué nuevas obligaciones o sanciones introduce la Ley 2439 de 2024 respecto "
        "al tratamiento de datos personales recolectados a través de sistemas de "
        "Inteligencia Artificial y comercio electrónico?"
    ),
    (
        "¿Ante qué entidad pública colombiana se deben reportar los incidentes de "
        "seguridad o violaciones a los códigos de seguridad de las bases de datos, y "
        "cuál es el término legal para hacerlo?"
    ),
    (
        'Enumere los requisitos mínimos que debe contener un formato de "Autorización '
        'de Tratamiento de Datos Personales" para que sea considerado válido y '
        "demostrable ante la Superintendencia de Industria y Comercio (SIC)."
    ),
    (
        "¿Cuál es el nuevo plazo máximo legal en días calendario que establece la Ley "
        "2439 de 2024 para que un proveedor y las entidades bancarias completen la "
        "devolución del dinero cuando un consumidor ejerce su derecho de retracto en "
        "comercio electrónico?"
    ),
    (
        "Según el artículo 26 de la Ley 1581 de 2012, ¿cuál es la regla general "
        "respecto a la transferencia de datos personales a terceros países y qué "
        'entidad tiene la facultad de declarar si un país ofrece un "nivel adecuado" '
        "de protección de datos?"
    ),
    (
        '¿Cómo define la Ley 2439 de 2024 el concepto de "portal de contacto" en el '
        "marco de las relaciones de consumo por medios electrónicos y qué tipo de "
        "plataformas (como los marketplaces) quedan expresamente cobijadas bajo esta "
        "definición?"
    ),
    (
        "Enumere las tres excepciones explícitas contempladas en el artículo 10 de la "
        "Ley 1581 de 2012 en las cuales el Responsable o Encargado no requiere la "
        "autorización del titular para proceder con el tratamiento de sus datos "
        "personales."
    ),
    (
        "De acuerdo con las obligaciones de los proveedores en canales electrónicos "
        "actualizadas por la Ley 2439 de 2024, ¿qué información técnica, de identidad "
        "y de contacto específica debe estar visible y disponible en todo momento en la "
        "interfaz de la plataforma para los usuarios?"
    ),
]


@dataclass
class ChunkMatch:
    consulta_indice: int
    consulta: str
    estrategia: str
    estrategia_chunking: str
    similitud: float
    chunk_index: int
    titulo: str
    texto: str
    normatividad_id: str


@dataclass
class MetricasEstrategia:
    estrategia: str
    estrategia_chunking: str
    similitud_promedio: float
    similitud_mediana: float
    similitud_minima: float
    similitud_maxima: float
    desviacion_estandar: float
    consultas_ganadas: int
    consultas_empatadas: int
    consultas_perdidas: int
    porcentaje_victorias: float


def similitud_coseno(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def cargar_chunks_por_estrategia(db) -> dict[str, list[dict[str, Any]]]:
    coleccion = db[COLECCION]
    chunks_por_estrategia: dict[str, list[dict[str, Any]]] = {
        valor: [] for valor in ESTRATEGIAS.values()
    }

    cursor = coleccion.find(
        {"activo": True, "estrategiaChunking": {"$in": list(ESTRATEGIAS.values())}},
        {
            "_id": 0,
            "normatividadId": 1,
            "titulo": 1,
            "texto": 1,
            "embedding": 1,
            "estrategiaChunking": 1,
            "chunkIndex": 1,
        },
    )

    for documento in cursor:
        estrategia = documento.get("estrategiaChunking")
        if estrategia in chunks_por_estrategia:
            chunks_por_estrategia[estrategia].append(documento)

    for nombre, chunks in chunks_por_estrategia.items():
        logger.info("Chunks cargados para '%s': %s", nombre, len(chunks))

    return chunks_por_estrategia


def encontrar_chunk_mas_cercano(
    consulta_embedding: np.ndarray,
    chunks: list[dict[str, Any]],
    estrategia: str,
    estrategia_chunking: str,
    consulta_indice: int,
    consulta: str,
) -> ChunkMatch | None:
    if not chunks:
        return None

    mejor_similitud = -1.0
    mejor_chunk: dict[str, Any] | None = None

    for chunk in chunks:
        embedding = np.asarray(chunk["embedding"], dtype=float)
        similitud = similitud_coseno(consulta_embedding, embedding)
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_chunk = chunk

    if mejor_chunk is None:
        return None

    return ChunkMatch(
        consulta_indice=consulta_indice,
        consulta=consulta,
        estrategia=estrategia,
        estrategia_chunking=estrategia_chunking,
        similitud=mejor_similitud,
        chunk_index=int(mejor_chunk["chunkIndex"]),
        titulo=str(mejor_chunk.get("titulo", "")),
        texto=str(mejor_chunk.get("texto", "")),
        normatividad_id=str(mejor_chunk.get("normatividadId", "")),
    )


def calcular_metricas(
    resultados: dict[str, list[ChunkMatch]],
    comparaciones: list[dict[str, Any]],
) -> dict[str, MetricasEstrategia]:
    metricas: dict[str, MetricasEstrategia] = {}

    for estrategia, matches in resultados.items():
        similitudes = np.array([match.similitud for match in matches], dtype=float)
        ganadas = sum(1 for c in comparaciones if c["ganadora"] == estrategia)
        empatadas = sum(1 for c in comparaciones if c["ganadora"] is None)
        perdidas = len(comparaciones) - ganadas - empatadas
        total = len(comparaciones)

        metricas[estrategia] = MetricasEstrategia(
            estrategia=estrategia,
            estrategia_chunking=ESTRATEGIAS[estrategia],
            similitud_promedio=float(np.mean(similitudes)),
            similitud_mediana=float(np.median(similitudes)),
            similitud_minima=float(np.min(similitudes)),
            similitud_maxima=float(np.max(similitudes)),
            desviacion_estandar=float(np.std(similitudes)),
            consultas_ganadas=ganadas,
            consultas_empatadas=empatadas,
            consultas_perdidas=perdidas,
            porcentaje_victorias=round((ganadas / total) * 100, 2) if total else 0.0,
        )

    return metricas


def determinar_ganadora(metricas: dict[str, MetricasEstrategia]) -> tuple[str, str]:
    ordenadas = sorted(
        metricas.values(),
        key=lambda m: (
            m.consultas_ganadas,
            m.similitud_promedio,
            -m.desviacion_estandar,
        ),
        reverse=True,
    )
    ganadora = ordenadas[0]
    segunda = ordenadas[1]

    diferencia_promedio = ganadora.similitud_promedio - segunda.similitud_promedio
    razon = (
        f"'{ganadora.estrategia}' ({ganadora.estrategia_chunking}) obtuvo mayor "
        f"similitud promedio ({ganadora.similitud_promedio:.4f} vs "
        f"{segunda.similitud_promedio:.4f}, dif={diferencia_promedio:.4f}) y ganó "
        f"{ganadora.consultas_ganadas} de {ganadora.consultas_ganadas + ganadora.consultas_perdidas + ganadora.consultas_empatadas} consultas."
    )
    return ganadora.estrategia, razon


def truncar_texto(texto: str, limite: int = 220) -> str:
    texto = " ".join(texto.split())
    if len(texto) <= limite:
        return texto
    return texto[: limite - 3] + "..."


def texto_para_consola(texto: str) -> str:
    return texto.encode("cp1252", errors="replace").decode("cp1252")


def imprimir_resultados(
    resultados: dict[str, list[ChunkMatch]],
    comparaciones: list[dict[str, Any]],
    metricas: dict[str, MetricasEstrategia],
    ganadora: str,
    razon: str,
) -> None:
    print("\n" + "=" * 90)
    print("COMPARACIÓN DE ESTRATEGIAS DE CHUNKING - embedding_normatividad")
    print("=" * 90)

    for indice, consulta in enumerate(CONSULTAS, start=1):
        print(f"\nConsulta {indice}:")
        print(f"  {texto_para_consola(consulta)}")

        comparacion = comparaciones[indice - 1]
        for estrategia in ESTRATEGIAS:
            match = resultados[estrategia][indice - 1]
            print(f"\n  Estrategia {estrategia} ({match.estrategia_chunking}):")
            print(f"    Similitud coseno : {match.similitud:.4f}")
            print(f"    Chunk index      : {match.chunk_index}")
            print(f"    Titulo           : {texto_para_consola(match.titulo)}")
            print(f"    Texto (preview)  : {texto_para_consola(truncar_texto(match.texto))}")

        ganadora_consulta = comparacion["ganadora"]
        if ganadora_consulta:
            print(
                f"\n  Mejor estrategia para esta consulta: {ganadora_consulta} "
                f"(dif={comparacion['diferenciaSimilitud']:.4f})"
            )
        else:
            print("\n  Empate entre estrategias.")

    print("\n" + "-" * 90)
    print("MÉTRICAS AGREGADAS")
    print("-" * 90)

    for estrategia, m in metricas.items():
        print(f"\nEstrategia: {estrategia} (estrategiaChunking='{m.estrategia_chunking}')")
        print(f"  Similitud promedio     : {m.similitud_promedio:.4f}")
        print(f"  Similitud mediana      : {m.similitud_mediana:.4f}")
        print(f"  Similitud mínima       : {m.similitud_minima:.4f}")
        print(f"  Similitud máxima       : {m.similitud_maxima:.4f}")
        print(f"  Desviación estándar    : {m.desviacion_estandar:.4f}")
        print(f"  Consultas ganadas      : {m.consultas_ganadas}")
        print(f"  Consultas empatadas    : {m.consultas_empatadas}")
        print(f"  Consultas perdidas     : {m.consultas_perdidas}")
        print(f"  Porcentaje de victorias: {m.porcentaje_victorias:.2f}%")

    print("\n" + "-" * 90)
    print(f"ESTRATEGIA RECOMENDADA: {ganadora}")
    print(f"Justificacion: {texto_para_consola(razon)}")
    print("-" * 90 + "\n")


def guardar_reporte(
    resultados: dict[str, list[ChunkMatch]],
    comparaciones: list[dict[str, Any]],
    metricas: dict[str, MetricasEstrategia],
    ganadora: str,
    razon: str,
) -> Path:
    reporte_dir = Path(__file__).resolve().parent / "reportes"
    reporte_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ruta_reporte = reporte_dir / f"comparate_chunks_{timestamp}.json"

    payload = {
        "generadoEn": datetime.now(timezone.utc).isoformat(),
        "coleccion": COLECCION,
        "modeloEmbedding": MODEL_NAME,
        "totalConsultas": len(CONSULTAS),
        "estrategiasEvaluadas": ESTRATEGIAS,
        "comparacionesPorConsulta": comparaciones,
        "resultadosPorEstrategia": {
            estrategia: [asdict(match) for match in matches]
            for estrategia, matches in resultados.items()
        },
        "metricas": {estrategia: asdict(m) for estrategia, m in metricas.items()},
        "estrategiaRecomendada": ganadora,
        "justificacion": razon,
    }

    with open(ruta_reporte, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, ensure_ascii=False, indent=2)

    return ruta_reporte


def ejecutar_comparacion() -> None:
    logger.info("Cargando modelo SentenceTransformer: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[DATABASE_NAME]
        chunks_por_estrategia = cargar_chunks_por_estrategia(db)

        for estrategia, chunks in chunks_por_estrategia.items():
            if not chunks:
                raise ValueError(
                    f"No se encontraron chunks activos para estrategiaChunking='{estrategia}' "
                    f"en la colección '{COLECCION}'."
                )
    except ConnectionFailure as exc:
        logger.error("No se pudo conectar a MongoDB: %s", exc)
        raise
    finally:
        client.close()

    logger.info("Generando embeddings para %s consultas...", len(CONSULTAS))
    consultas_embeddings = model.encode(CONSULTAS, convert_to_numpy=True)

    resultados: dict[str, list[ChunkMatch]] = {nombre: [] for nombre in ESTRATEGIAS}
    comparaciones: list[dict[str, Any]] = []

    for indice, (consulta, consulta_embedding) in enumerate(
        zip(CONSULTAS, consultas_embeddings), start=1
    ):
        matches_consulta: dict[str, ChunkMatch] = {}

        for estrategia, estrategia_chunking in ESTRATEGIAS.items():
            match = encontrar_chunk_mas_cercano(
                consulta_embedding=consulta_embedding,
                chunks=chunks_por_estrategia[estrategia_chunking],
                estrategia=estrategia,
                estrategia_chunking=estrategia_chunking,
                consulta_indice=indice,
                consulta=consulta,
            )
            if match is None:
                raise ValueError(
                    f"No fue posible encontrar chunk para la consulta {indice} "
                    f"con estrategia '{estrategia}'."
                )
            resultados[estrategia].append(match)
            matches_consulta[estrategia] = match

        sim_sentences = matches_consulta["sentences"].similitud
        sim_semantic = matches_consulta["semantic"].similitud

        if abs(sim_sentences - sim_semantic) < 1e-9:
            ganadora_consulta = None
        elif sim_sentences > sim_semantic:
            ganadora_consulta = "sentences"
        else:
            ganadora_consulta = "semantic"

        comparaciones.append(
            {
                "consultaIndice": indice,
                "consulta": consulta,
                "similitudSentences": round(sim_sentences, 6),
                "similitudSemantic": round(sim_semantic, 6),
                "diferenciaSimilitud": round(abs(sim_sentences - sim_semantic), 6),
                "ganadora": ganadora_consulta,
            }
        )

    metricas = calcular_metricas(resultados, comparaciones)
    ganadora, razon = determinar_ganadora(metricas)

    imprimir_resultados(resultados, comparaciones, metricas, ganadora, razon)
    ruta_reporte = guardar_reporte(resultados, comparaciones, metricas, ganadora, razon)
    logger.info("Reporte JSON guardado en: %s", ruta_reporte)


if __name__ == "__main__":
    try:
        ejecutar_comparacion()
    except (ValueError, ConnectionFailure, PyMongoError) as exc:
        logger.error("La comparación falló: %s", exc)
        raise
