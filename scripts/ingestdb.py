import logging
import os
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, ConnectionFailure, PyMongoError


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("mongodb_ingest")

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGO_URI or not DATABASE_NAME:
    raise ValueError("Las variables MONGO_URI y DATABASE_NAME deben estar definidas en el archivo .env")


try:
    from faker import Faker

    fake = Faker("es_CO")
except ModuleNotFoundError:
    fake = None


COLLECTIONS = [
    "normatividades",
    "categorias",
    "usuarios",
    "vendedores",
    "productos",
    "ordenes",
    "resenas",
    "transacciones",
    "embeddings",
    "consultas",
    "notificaciones",
]

DEPARTAMENTOS = ["Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Atlantico", "Bolivar"]
CIUDADES = ["Medellin", "Bogota", "Cali", "Bucaramanga", "Barranquilla", "Cartagena"]
NOMBRES = ["Camila", "Nicolas", "Valentina", "Santiago", "Laura", "Andres", "Manuela", "Sebastian"]
APELLIDOS = ["Ramirez", "Gomez", "Riascos", "Torres", "Moreno", "Castro", "Vargas", "Mejia"]
PRODUCTOS = [
    "Camiseta organica",
    "Mochila urbana",
    "Audifonos bluetooth",
    "Botella termica",
    "Cuaderno premium",
    "Lampara de escritorio",
    "Mouse ergonomico",
    "Teclado mecanico",
    "Silla auxiliar",
    "Soporte para laptop",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "tienda"


def unique_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def random_phone() -> str:
    return "+573" + "".join(str(random.randint(0, 9)) for _ in range(9))


def random_address(index: int) -> dict[str, str]:
    departamento = random.choice(DEPARTAMENTOS)
    ciudad = random.choice(CIUDADES)
    if fake:
        direccion = fake.street_address()
    else:
        direccion = f"Calle {random.randint(1, 120)} #{random.randint(1, 80)}-{random.randint(1, 99)}"

    return {
        "departamento": departamento,
        "ciudad": ciudad,
        "direccion": direccion,
        "codigoPostal": f"{110000 + index:06d}",
    }


def person_name(index: int) -> tuple[str, str]:
    if fake:
        return fake.first_name(), fake.last_name()
    return random.choice(NOMBRES), random.choice(APELLIDOS)


def text_paragraph(topic: str, index: int) -> str:
    return (
        f"{topic} generado para pruebas de comercio electronico. "
        f"Documento {index} con informacion descriptiva, condiciones de uso, "
        "atributos relevantes y contenido suficiente para busqueda textual y RAG."
    )


def common_fields(created_at: datetime | None = None) -> dict[str, Any]:
    date = created_at or now_utc()
    return {
        "schemaVersion": 1,
        "creadoEn": date,
        "actualizadoEn": date,
        "eliminadoEn": None,
    }


def make_embedding(size: int = 16) -> list[float]:
    return [round(random.uniform(-1, 1), 6) for _ in range(size)]


def insert_many(db, collection_name: str, docs: list[dict[str, Any]]) -> None:
    if not docs:
        return

    try:
        db[collection_name].insert_many(docs, ordered=False)
        logger.info("%s: %s documentos insertados", collection_name, len(docs))
    except BulkWriteError as exc:
        errors = exc.details.get("writeErrors", [])
        logger.error("%s: fallo insertando %s documentos", collection_name, len(errors))
        for error in errors[:3]:
            logger.error("Error ejemplo: %s", error.get("errmsg"))
        raise


def build_normatividades(batch: str) -> list[dict[str, Any]]:
    tipos = ["privacidad", "terminos", "devoluciones", "envios", "cookies", "dpa", "cumplimiento", "otro"]
    docs = []
    for index, tipo in enumerate(tipos, start=1):
        date = now_utc() - timedelta(days=index)
        docs.append(
            {
                "_id": ObjectId(),
                "tipo": tipo,
                "titulo": f"Politica de {tipo} lote {batch}",
                "version": f"1.{int(batch[-12:-6])}.{int(batch[-6:]) + index}",
                "contenido": text_paragraph(f"Normatividad de {tipo}", index),
                "fechaPublicacion": date,
                "activo": True,
                "idioma": "es",
                "fechaVigencia": date + timedelta(days=365),
                **common_fields(date),
            }
        )
    return docs


def build_categorias(batch: str, total: int = 12) -> list[dict[str, Any]]:
    docs = []
    nombres = [
        "Moda",
        "Tecnologia",
        "Hogar",
        "Papeleria",
        "Deportes",
        "Belleza",
        "Mascotas",
        "Juguetes",
        "Libros",
        "Oficina",
        "Cocina",
        "Salud",
    ]
    for index in range(total):
        date = now_utc() - timedelta(days=random.randint(1, 90))
        nombre = f"{nombres[index % len(nombres)]} {batch[-4:]}"
        docs.append(
            {
                "_id": ObjectId(),
                "nombre": nombre,
                "descripcion": text_paragraph(f"Categoria {nombre}", index + 1),
                "estado": "activa",
                **common_fields(date),
            }
        )
    return docs


def build_usuarios(batch: str, vendor_ids: list[ObjectId], total: int = 40) -> list[dict[str, Any]]:
    docs = []
    for index in range(total):
        date = now_utc() - timedelta(days=random.randint(1, 180))
        nombre, apellido = person_name(index)
        email_name = f"{nombre}.{apellido}.{batch}.{index}".lower()
        docs.append(
            {
                "_id": ObjectId(),
                "email": f"{slugify(email_name)}@example.com",
                "nombre": nombre,
                "apellido": apellido,
                "telefono": random_phone(),
                "direccion": random_address(index),
                "carrito": {
                    "items": [],
                    "totalEstimado": 0.0,
                    "fechaActualizacion": date,
                },
                "roles": ["cliente"],
                "vendedorId": random.choice(vendor_ids),
                "normativasAceptadas": [],
                **common_fields(date),
            }
        )
    return docs


def build_vendedores(batch: str, usuarios: list[dict[str, Any]], vendor_ids: list[ObjectId]) -> list[dict[str, Any]]:
    docs = []
    seller_users = usuarios[: len(vendor_ids)]
    for index, (usuario, vendor_id) in enumerate(zip(seller_users, vendor_ids), start=1):
        date = now_utc() - timedelta(days=random.randint(1, 150))
        tienda = f"Tienda {usuario['nombre']} {usuario['apellido']} {batch[-5:]} {index}"
        usuario["roles"] = ["cliente", "vendedor"]
        usuario["vendedorId"] = vendor_id
        docs.append(
            {
                "_id": vendor_id,
                "usuarioId": usuario["_id"],
                "tiendaNombre": tienda,
                "tiendaSlug": slugify(tienda),
                "tipoVendedor": random.choice(["persona_natural", "empresa", "marca"]),
                "comisionPorcentaje": round(random.uniform(5.0, 18.0), 2),
                "estado": "aprobado",
                **common_fields(date),
            }
        )
    return docs


def build_productos(
    batch: str,
    vendedores: list[dict[str, Any]],
    categorias: list[dict[str, Any]],
    total: int = 120,
) -> list[dict[str, Any]]:
    docs = []
    for index in range(total):
        date = now_utc() - timedelta(days=random.randint(1, 120))
        producto_base = random.choice(PRODUCTOS)
        nombre = f"{producto_base} {batch[-4:]}-{index + 1:03d}"
        image_count = 1 if index < 90 else 2
        docs.append(
            {
                "_id": ObjectId(),
                "vendedorId": random.choice(vendedores)["_id"],
                "categoriaId": random.choice(categorias)["_id"],
                "nombre": nombre,
                "descripcion": text_paragraph(f"Producto {nombre}", index + 1),
                "precio": round(random.uniform(15000, 850000), 2),
                "stock": random.randint(1, 250),
                "estado": "aprobado",
                "activo": True,
                "imagenesUrl": [
                    f"https://picsum.photos/seed/{batch}-{index}-{image}/900/700"
                    for image in range(1, image_count + 1)
                ],
                **common_fields(date),
            }
        )
    return docs


def build_ordenes(
    usuarios: list[dict[str, Any]],
    productos: list[dict[str, Any]],
    normatividades: list[dict[str, Any]],
    total: int = 50,
) -> list[dict[str, Any]]:
    docs = []
    for index in range(total):
        date = now_utc() - timedelta(days=random.randint(1, 60))
        usuario = random.choice(usuarios)
        selected_products = random.sample(productos, k=random.randint(1, 3))
        items = []
        for product in selected_products:
            cantidad = random.randint(1, 4)
            precio = float(product["precio"])
            subtotal = round(cantidad * precio, 2)
            items.append(
                {
                    "itemId": product["_id"],
                    "vendedor": product["vendedorId"],
                    "nombreProducto": product["nombre"],
                    "cantidad": cantidad,
                    "precioUnitario": precio,
                    "subtotal": subtotal,
                    "estadoVendedor": random.choice(["pendiente", "en_preparacion", "enviado", "entregado"]),
                    "tracking": f"TRK-{ObjectId()}",
                    "fechaEnvio": date + timedelta(days=1),
                    "fechaEntregaEstimada": date + timedelta(days=random.randint(3, 8)),
                }
            )
        total_productos = round(sum(item["subtotal"] for item in items), 2)
        envio = round(random.uniform(8000, 25000), 2)
        comision = round(total_productos * 0.08, 2)
        total_orden = round(total_productos + envio, 2)
        norma = random.choice(normatividades)
        docs.append(
            {
                "_id": ObjectId(),
                "usuarioId": usuario["_id"],
                "pago": {
                    "metodo": random.choice(["tarjeta_credito", "pse", "nequi", "transferencia"]),
                    "estado": random.choice(["aprobado", "pendiente", "rechazado"]),
                    "monto": total_orden,
                    "fechaPago": date,
                },
                "items": items,
                "resumen": {
                    "totalProductos": total_productos,
                    "costoEnvioTotal": envio,
                    "comisionPlataforma": comision,
                    "total": total_orden,
                },
                "normatividadAceptada": [
                    {
                        "normatividadId": norma["_id"],
                        "titulo": norma["titulo"],
                        "version": norma["version"],
                        "fechaAceptacion": date,
                    }
                ],
                "direccionEnvio": {
                    "direccion": usuario["direccion"]["direccion"],
                    "ciudad": usuario["direccion"]["ciudad"],
                    "departamento": usuario["direccion"]["departamento"],
                    "destinatario": f"{usuario['nombre']} {usuario['apellido']}",
                    "telefono": usuario["telefono"],
                },
                **common_fields(date),
            }
        )
    return docs


def build_transacciones(batch: str, ordenes: list[dict[str, Any]], total: int = 50) -> list[dict[str, Any]]:
    docs = []
    for index, orden in enumerate(ordenes[:total], start=1):
        date = orden["creadoEn"] + timedelta(minutes=random.randint(1, 180))
        by_vendor: dict[ObjectId, float] = {}
        for item in orden["items"]:
            by_vendor[item["vendedor"]] = by_vendor.get(item["vendedor"], 0.0) + float(item["subtotal"])
        detalle = []
        monto_plataforma = 0.0
        for vendor_id, bruto in by_vendor.items():
            comision = round(bruto * 0.08, 2)
            monto_plataforma += comision
            detalle.append(
                {
                    "vendedorId": vendor_id,
                    "montoBruto": round(bruto, 2),
                    "comision": comision,
                    "montoNeto": round(bruto - comision, 2),
                    "estadoPagoVendedor": random.choice(["pendiente", "pagado", "retenido"]),
                    "fechaPagoVendedor": date + timedelta(days=random.randint(1, 7)),
                }
            )
        monto_total = float(orden["resumen"]["total"])
        docs.append(
            {
                "_id": ObjectId(),
                "ordenId": orden["_id"],
                "usuarioId": orden["usuarioId"],
                "transactionId": f"TX-{batch}-{index:04d}",
                "tipo": "venta",
                "montoTotal": monto_total,
                "montoPlataforma": round(monto_plataforma, 2),
                "montoNetoVendedores": round(sum(item["montoNeto"] for item in detalle), 2),
                "detalleVendedores": detalle,
                "metodoPago": orden["pago"]["metodo"],
                "estado": orden["pago"]["estado"],
                "fechaTransaccion": date,
                **common_fields(date),
            }
        )
    return docs


def build_resenas(
    usuarios: list[dict[str, Any]],
    productos: list[dict[str, Any]],
    total: int = 80,
) -> list[dict[str, Any]]:
    docs = []
    for index in range(total):
        date = now_utc() - timedelta(days=random.randint(1, 45))
        usuario = random.choice(usuarios)
        producto = random.choice(productos)
        docs.append(
            {
                "_id": ObjectId(),
                "usuarioId": usuario["_id"],
                "productoId": producto["_id"],
                "vendedorId": producto["vendedorId"],
                "calificacion": random.randint(1, 5),
                "comentario": text_paragraph(f"Resena del producto {producto['nombre']}", index + 1),
                **common_fields(date),
            }
        )
    return docs


def build_embeddings(
    normatividades: list[dict[str, Any]],
    categorias: list[dict[str, Any]],
    productos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    docs = []
    for norma in normatividades:
        docs.append(
            {
                "_id": ObjectId(),
                "fuente": "normatividad",
                "normatividadId": norma["_id"],
                "productoId": None,
                "categoriaId": None,
                "texto": norma["contenido"],
                "embedding": make_embedding(),
                "activo": True,
                **common_fields(norma["creadoEn"]),
            }
        )
    for categoria in categorias:
        docs.append(
            {
                "_id": ObjectId(),
                "fuente": "categoria",
                "normatividadId": None,
                "productoId": None,
                "categoriaId": categoria["_id"],
                "texto": categoria["descripcion"],
                "embedding": make_embedding(),
                "activo": True,
                **common_fields(categoria["creadoEn"]),
            }
        )
    for producto in productos:
        docs.append(
            {
                "_id": ObjectId(),
                "fuente": "producto",
                "normatividadId": None,
                "productoId": producto["_id"],
                "categoriaId": None,
                "texto": f"{producto['nombre']}. {producto['descripcion']}",
                "embedding": make_embedding(),
                "activo": True,
                **common_fields(producto["creadoEn"]),
            }
        )
    return docs


def build_consultas(
    usuarios: list[dict[str, Any]],
    vendedores: list[dict[str, Any]],
    total: int = 40,
) -> list[dict[str, Any]]:
    docs = []
    for index in range(total):
        date = now_utc() - timedelta(days=random.randint(1, 30))
        docs.append(
            {
                "_id": ObjectId(),
                "usuarioId": random.choice(usuarios)["_id"],
                "asunto": f"Consulta sobre pedido {index + 1}",
                "mensaje": text_paragraph("Mensaje de soporte para vendedor", index + 1),
                "estado": random.choice(["pendiente", "abierta", "resuelta", "cerrada"]),
                "vendedorId": random.choice(vendedores)["_id"],
                **common_fields(date),
            }
        )
    return docs


def build_notificaciones(
    usuarios: list[dict[str, Any]],
    vendedores: list[dict[str, Any]],
    ordenes: list[dict[str, Any]],
    productos: list[dict[str, Any]],
    total: int = 80,
) -> list[dict[str, Any]]:
    docs = []
    tipos = [
        "nueva_orden",
        "orden_actualizada",
        "envio_iniciado",
        "paquete_entregado",
        "mensaje_vendedor",
        "promocion",
        "revision_producto",
        "otro",
    ]
    for index in range(total):
        date = now_utc() - timedelta(days=random.randint(1, 20))
        for_vendor = index % 3 == 0
        destinatario = random.choice(vendedores if for_vendor else usuarios)
        orden = random.choice(ordenes)
        producto = random.choice(productos)
        leida = random.choice([True, False])
        docs.append(
            {
                "_id": ObjectId(),
                "destinatarioId": destinatario["_id"],
                "destinatarioTipo": "vendedor" if for_vendor else "usuario",
                "tipo": random.choice(tipos),
                "titulo": f"Actualizacion de actividad {index + 1}",
                "mensaje": text_paragraph("Notificacion del sistema", index + 1),
                "leida": leida,
                "referencia": {
                    "tipo": random.choice(["orden", "producto", "otro"]),
                    "id": random.choice([orden["_id"], producto["_id"]]),
                },
                "datos": {
                    "numeroOrden": str(orden["_id"]),
                    "estadoOrden": orden["pago"]["estado"],
                    "cantidadProductos": len(orden["items"]),
                    "estadoAnterior": "pendiente",
                    "estadoNuevo": orden["pago"]["estado"],
                    "monto": float(orden["resumen"]["total"]),
                },
                "leidaEn": date + timedelta(hours=3) if leida else None,
                **common_fields(date),
            }
        )
    return docs


def ingest_database() -> None:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[DATABASE_NAME]
        batch = unique_suffix()
        logger.info("Conectado a MongoDB. Base de datos: %s", DATABASE_NAME)
        logger.info("Creando lote de datos: %s", batch)

        vendor_ids = [ObjectId() for _ in range(20)]
        normatividades = build_normatividades(batch)
        categorias = build_categorias(batch)
        usuarios = build_usuarios(batch, vendor_ids)
        vendedores = build_vendedores(batch, usuarios, vendor_ids)
        productos = build_productos(batch, vendedores, categorias)
        ordenes = build_ordenes(usuarios, productos, normatividades)
        transacciones = build_transacciones(batch, ordenes)
        resenas = build_resenas(usuarios, productos)
        embeddings = build_embeddings(normatividades, categorias, productos)
        consultas = build_consultas(usuarios, vendedores)
        notificaciones = build_notificaciones(usuarios, vendedores, ordenes, productos)

        insert_many(db, "normatividades", normatividades)
        insert_many(db, "categorias", categorias)
        insert_many(db, "usuarios", usuarios)
        insert_many(db, "vendedores", vendedores)
        insert_many(db, "productos", productos)
        insert_many(db, "ordenes", ordenes)
        insert_many(db, "transacciones", transacciones)
        insert_many(db, "resenas", resenas)
        insert_many(db, "embeddings", embeddings)
        insert_many(db, "consultas", consultas)
        insert_many(db, "notificaciones", notificaciones)

        text_docs = len(productos) + len(normatividades) + len(categorias) + len(resenas) + len(consultas)
        image_urls = sum(len(producto["imagenesUrl"]) for producto in productos)

        logger.info("Documentos de texto creados en el lote: %s", text_docs)
        logger.info("Imagenes asociadas creadas en el lote: %s", image_urls)
        logger.info("--- Conteo total actual por coleccion ---")
        for collection in COLLECTIONS:
            logger.info("%s: %s", collection, db[collection].count_documents({}))

    except ConnectionFailure as exc:
        logger.error("No se pudo conectar a MongoDB: %s", exc)
        raise
    except PyMongoError as exc:
        logger.error("Error durante la ingesta: %s", exc)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    ingest_database()
