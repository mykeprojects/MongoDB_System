import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import json_util

# ====================== CONFIGURACIÓN ======================
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("valuable_queries")

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGO_URI or not DATABASE_NAME:
    raise ValueError("Las variables MONGO_URI y DATABASE_NAME deben estar en el archivo .env")

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

def print_results(titulo: str, pregunta: str, cursor):
    """Función auxiliar para imprimir los resultados de forma legible"""
    # print(f"\n{'='*80}")
    print(f" {titulo}")
    print(f" {pregunta}")
    print(f"{'-'*80}")
    
    resultados = list(cursor)
    if not resultados:
        print("No se encontraron resultados (probablemente no hay datos de prueba aún).")
    else:
        # Usamos json_util para que formatee correctamente los ObjectId y Fechas
        print(json_util.dumps(resultados, indent=4, ensure_ascii=False))
    print(f"{'='*80}\n")

def run_queries():
    logger.info(f"Conectado a la base de datos: {DATABASE_NAME}. Ejecutando agregaciones...\n")

    # CONSULTA 1: Top 5 Vendedores por Volumen de Ventas
    # Identifica a los vendedores que generan más ingresos en la plataforma, 
    # desglosando las órdenes para sumar el subtotal de los items vendidos por cada uno.
    pipeline_ventas_vendedores = [
        {"$unwind": "$items"}, # Desglosa el array de items de cada orden
        {"$group": {
            "_id": "$items.vendedor",
            "totalIngresosGenerados": {"$sum": "$items.subtotal"},
            "cantidadProductosVendidos": {"$sum": "$items.cantidad"}
        }},
        {"$sort": {"totalIngresosGenerados": -1}}, # Ordena de mayor a menor ingreso
        {"$limit": 5}, # Trae solo el Top 5
        {"$lookup": { # Hace join con la colección de vendedores para traer su nombre
            "from": "vendedores",
            "localField": "_id",
            "foreignField": "_id",
            "as": "datosVendedor"
        }},
        {"$unwind": "$datosVendedor"},
        {"$project": {
            "_id": 0,
            "vendedorId": "$_id",
            "tienda": "$datosVendedor.tiendaNombre",
            "totalIngresosGenerados": 1,
            "cantidadProductosVendidos": 1
        }}
    ]
    print_results(
        "TOP 5 VENDEDORES POR INGRESOS", 
        "¿Cuáles son los vendedores que más dinero generan y cuántos items han vendido?",
        db.ordenes.aggregate(pipeline_ventas_vendedores)
    )

    # CONSULTA 2: Valor del Inventario por Categoría
    # Permite a los administradores saber cuánto dinero hay "sentado" en 
    # inventario activo por cada categoría, multiplicando el precio por el stock de cada producto.
    pipeline_inventario = [
        {"$match": {"estado": "aprobado", "activo": True}}, # Solo productos activos y aprobados
        {"$group": {
            "_id": "$categoriaId",
            "valorTotalInventario": {"$sum": {"$multiply": ["$precio", "$stock"]}},
            "cantidadProductosDistintos": {"$sum": 1},
            "stockTotal": {"$sum": "$stock"}
        }},
        {"$sort": {"valorTotalInventario": -1}},
        {"$lookup": {
            "from": "categorias",
            "localField": "_id",
            "foreignField": "_id",
            "as": "categoriaInfo"
        }},
        {"$unwind": "$categoriaInfo"},
        {"$project": {
            "_id": 0,
            "categoria": "$categoriaInfo.nombre",
            "valorTotalInventario": 1,
            "cantidadProductosDistintos": 1,
            "stockTotal": 1
        }}
    ]
    print_results(
        "VALOR DE INVENTARIO POR CATEGORÍA", 
        "¿Cuál es el valor económico total del inventario disponible agrupado por categoría?",
        db.productos.aggregate(pipeline_inventario)
    )

    # CONSULTA 3: Top Productos con Mejores Reseñas (Mínimo 2 reseñas)
    # Muestra los productos más amados por los usuarios. Exige un conteo 
    # mínimo de reseñas para evitar sesgar el promedio con productos que solo tienen 1 reseña de 5 estrellas.
    pipeline_mejores_productos = [
        {"$group": {
            "_id": "$productoId",
            "calificacionPromedio": {"$avg": "$calificacion"},
            "totalResenas": {"$sum": 1}
        }},
        {"$match": {"totalResenas": {"$gte": 2}}}, # Filtro: al menos 2 reseñas
        {"$sort": {"calificacionPromedio": -1, "totalResenas": -1}}, # Ordena por calificación y luego volumen
        {"$limit": 5},
        {"$lookup": {
            "from": "productos",
            "localField": "_id",
            "foreignField": "_id",
            "as": "productoInfo"
        }},
        {"$unwind": "$productoInfo"},
        {"$project": {
            "_id": 0,
            "producto": "$productoInfo.nombre",
            "calificacionPromedio": {"$round": ["$calificacionPromedio", 2]},
            "totalResenas": 1,
            "precio": "$productoInfo.precio"
        }}
    ]
    print_results(
        "PRODUCTOS MEJOR CALIFICADOS", 
        "¿Cuáles son los productos con la calificación promedio más alta (que tengan al menos 2 reseñas)?",
        db.resenas.aggregate(pipeline_mejores_productos)
    )

    # CONSULTA 4: Usuarios con Carritos Abandonados de Alto Valor
    # Identifica oportunidades de remarketing directo. Busca usuarios que 
    # tienen dinero estimado en su carrito pero no han completado la orden reciente.
    pipeline_carritos = [
        {"$match": {
            "carrito.totalEstimado": {"$gt": 0},
            "carrito.items": {"$not": {"$size": 0}}
        }},
        {"$sort": {"carrito.totalEstimado": -1}}, # Ordenar por los carritos más caros
        {"$limit": 5},
        {"$project": {
            "_id": 0,
            "nombreCompleto": {"$concat": ["$nombre", " ", "$apellido"]},
            "email": 1,
            "totalEnCarrito": "$carrito.totalEstimado",
            "cantidadItems": {"$size": "$carrito.items"},
            "ultimaVezActualizado": "$carrito.fechaActualizacion"
        }}
    ]
    print_results(
        "CARRITOS ABANDONADOS DE ALTO VALOR", 
        "¿Qué usuarios tienen los carritos de compras con mayor valor monetario actualmente?",
        db.usuarios.aggregate(pipeline_carritos)
    )

    # CONSULTA 5: Cuello de botella en Soporte por Vendedor
    # Ayuda a la plataforma a auditar qué vendedores tienen la mayor cantidad 
    # de tickets/consultas abiertas o pendientes, lo que indica un posible mal servicio al cliente.
    pipeline_soporte = [
        {"$match": {"estado": {"$in": ["pendiente", "abierta"]}}}, # Solo consultas sin resolver
        {"$group": {
            "_id": "$vendedorId",
            "consultasSinResolver": {"$sum": 1}
        }},
        {"$sort": {"consultasSinResolver": -1}},
        {"$limit": 5},
        {"$lookup": {
            "from": "vendedores",
            "localField": "_id",
            "foreignField": "_id",
            "as": "vendedorInfo"
        }},
        # Left outer join simulado (por si hay consultas de plataforma general sin vendedor)
        {"$unwind": {"path": "$vendedorInfo", "preserveNullAndEmptyArrays": True}}, 
        {"$project": {
            "_id": 0,
            "vendedorId": "$_id",
            "tienda": {"$ifNull": ["$vendedorInfo.tiendaNombre", "Soporte General Plataforma"]},
            "consultasSinResolver": 1
        }}
    ]
    print_results(
        "RETRASOS EN SOPORTE POR VENDEDOR", 
        "¿Qué vendedores tienen la mayor cantidad de consultas pendientes o abiertas?",
        db.consultas.aggregate(pipeline_soporte)
    )

if __name__ == "__main__":
    try:
        run_queries()
    except Exception as e:
        logger.error(f"Error ejecutando las consultas: {e}")