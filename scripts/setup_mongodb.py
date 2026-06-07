import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid, OperationFailure, ConnectionFailure, PyMongoError

# ====================== CONFIGURACIÓN ======================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("mongodb_setup")

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGO_URI or not DATABASE_NAME:
    raise ValueError("Las variables MONGO_URI y DATABASE_NAME deben estar definidas en el archivo .env")

# Conexión al servidor MongoDB
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
client.admin.command("ping")
db = client[DATABASE_NAME]

logger.info("Conectado a la base de datos: %s", DATABASE_NAME)

# ====================== DEFINICIÓN DE VALIDADORES ======================

def get_validator(collection_name: str) -> dict:
    """Retorna el JSON Schema validator para cada colección."""

    base_common_properties = {
        "schemaVersion": {"bsonType": "int", "minimum": 1},
        "creadoEn": {"bsonType": "date"},
        "actualizadoEn": {"bsonType": "date"},
        "eliminadoEn": {"bsonType": ["date", "null"]}
    }

    validators = {
        "usuarios": {
            "bsonType": "object",
            "required": [
                "email", "nombre", "apellido", "telefono", "direccion", "carrito",
                "roles", "vendedorId", "normativasAceptadas", "schemaVersion", "creadoEn", "actualizadoEn"
            ],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "email": {
                    "bsonType": "string",
                    "description": "Correo electrónico único",
                    "pattern": "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$"
                },
                "nombre": {"bsonType": "string", "minLength": 2, "maxLength": 120},
                "apellido": {"bsonType": "string", "minLength": 2, "maxLength": 120},
                "telefono": {"bsonType": "string", "pattern": "^\\+?[1-9]\\d{7,14}$"},
                "direccion": {
                    "bsonType": "object",
                    "required": ["departamento", "ciudad", "direccion", "codigoPostal"],
                    "properties": {
                        "departamento": {"bsonType": "string"},
                        "ciudad": {"bsonType": "string"},
                        "direccion": {"bsonType": "string"},
                        "codigoPostal": {"bsonType": "string"}
                    }
                },
                "carrito": {
                    "bsonType": "object",
                    "required": ["items", "totalEstimado", "fechaActualizacion"],
                    "properties": {
                        "items": {
                            "bsonType": "array",
                            "items": {
                                "bsonType": "object",
                                "required": ["itemId"],
                                "properties": {
                                    "itemId": {"bsonType": "objectId"}
                                }
                            }
                        },
                        "totalEstimado": {"bsonType": "double", "minimum": 0},
                        "fechaActualizacion": {"bsonType": "date"}
                    }
                },
                "roles": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"}
                },
                "vendedorId": {"bsonType": "objectId"},
                "normativasAceptadas": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["normatividadId", "titulo", "version", "fechaAceptacion"],
                        "properties": {
                            "normatividadId": {"bsonType": "objectId"},
                            "titulo": {"bsonType": "string"},
                            "version": {"bsonType": "string"},
                            "fechaAceptacion": {"bsonType": "date"}
                        }
                    }
                },
                **base_common_properties
            }
        },
        "vendedores": {
            "bsonType": "object",
            "required": ["usuarioId", "tiendaNombre", "tiendaSlug", "tipoVendedor", "estado", "schemaVersion", "creadoEn", "actualizadoEn"],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "usuarioId": {"bsonType": "objectId"},
                "tiendaNombre": {"bsonType": "string", "minLength": 3, "maxLength": 140},
                "tiendaSlug": {"bsonType": "string", "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$"},
                "tipoVendedor": {"bsonType": "string", "enum": ["persona_natural", "empresa", "marca"]},
                "comisionPorcentaje": {"bsonType": "double", "minimum": 0},
                "estado": {"bsonType": "string", "enum": ["pendiente", "aprobado", "suspendido", "rechazado"]},
                **base_common_properties
            }
        },
        "productos": {
            "bsonType": "object",
            "required": [
                "vendedorId", "categoriaId", "nombre", "descripcion", "precio", "stock",
                "estado", "activo", "creadoEn", "imagenesUrl", "schemaVersion", "actualizadoEn"
            ],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "vendedorId": {"bsonType": "objectId"},
                "nombre": {"bsonType": "string", "minLength": 3, "maxLength": 180},
                "descripcion": {"bsonType": "string", "minLength": 10},
                "categoriaId": {"bsonType": "objectId"},
                "precio": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                "stock": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                "estado": {"bsonType": "string", "enum": ["borrador", "pendiente_aprobacion", "aprobado", "rechazado", "inactivo"]},
                "activo": {"bsonType": "bool"},
                "imagenesUrl": {
                    "bsonType": "array",
                    "minItems": 1,
                    "items": {"bsonType": "string"}
                },
                **base_common_properties
            }
        },
        "categorias": {
            "bsonType": "object",
            "required": ["nombre", "descripcion", "estado", "schemaVersion", "creadoEn", "actualizadoEn"],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "nombre": {"bsonType": "string"},
                "descripcion": {"bsonType": "string"},
                "estado": {"bsonType": "string"},
                **base_common_properties
            }
        },
        "ordenes": {
            "bsonType": "object",
            "required": [
                "usuarioId", "pago", "items", "resumen", "normatividadAceptada", "direccionEnvio",
                "schemaVersion", "creadoEn", "actualizadoEn"
            ],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "usuarioId": {"bsonType": "objectId"},
                "pago": {
                    "bsonType": "object",
                    "required": ["metodo", "estado", "monto", "fechaPago"],
                    "properties": {
                        "metodo": {"bsonType": "string"},
                        "estado": {"bsonType": "string"},
                        "monto": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                        "fechaPago": {"bsonType": "date"}
                    }
                },
                "items": {
                    "bsonType": "array",
                    "minItems": 1,
                    "items": {
                        "bsonType": "object",
                        "required": [
                            "itemId", "vendedor", "nombreProducto", "cantidad", "precioUnitario", "subtotal",
                            "estadoVendedor", "tracking", "fechaEnvio", "fechaEntregaEstimada"
                        ],
                        "properties": {
                            "itemId": {"bsonType": "objectId"},
                            "vendedor": {"bsonType": "objectId"},
                            "nombreProducto": {"bsonType": "string"},
                            "cantidad": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                            "precioUnitario": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                            "subtotal": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                            "estadoVendedor": {
                                "bsonType": "string",
                                "enum": ["pendiente", "en_preparacion", "enviado", "entregado", "cancelado"]
                            },
                            "tracking": {"bsonType": "string"},
                            "fechaEnvio": {"bsonType": "date"},
                            "fechaEntregaEstimada": {"bsonType": "date"}
                        }
                    }
                },
                "resumen": {
                    "bsonType": "object",
                    "required": ["totalProductos", "costoEnvioTotal", "comisionPlataforma", "total"],
                    "properties": {
                        "totalProductos": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                        "costoEnvioTotal": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                        "comisionPlataforma": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                        "total": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0}
                    }
                },
                "normatividadAceptada": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["normatividadId", "titulo", "version", "fechaAceptacion"],
                        "properties": {
                            "normatividadId": {"bsonType": "objectId"},
                            "titulo": {"bsonType": "string"},
                            "version": {"bsonType": "string"},
                            "fechaAceptacion": {"bsonType": "date"}
                        }
                    }
                },
                "direccionEnvio": {
                    "bsonType": "object",
                    "required": ["direccion", "ciudad", "departamento", "destinatario", "telefono"],
                    "properties": {
                        "direccion": {"bsonType": "string"},
                        "ciudad": {"bsonType": "string"},
                        "departamento": {"bsonType": "string"},
                        "destinatario": {"bsonType": "string"},
                        "telefono": {"bsonType": "string"}
                    }
                },
                **base_common_properties
            }
        },
        "resenas": {
            "bsonType": "object",
            "required": [
                "usuarioId", "productoId", "vendedorId", "calificacion", "comentario",
                "schemaVersion", "creadoEn", "actualizadoEn"
            ],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "usuarioId": {"bsonType": "objectId"},
                "productoId": {"bsonType": "objectId"},
                "vendedorId": {"bsonType": "objectId"},
                "calificacion": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 1, "maximum": 5},
                "comentario": {"bsonType": "string"},
                **base_common_properties
            }
        },
        "transacciones": {
            "bsonType": "object",
            "required": [
                "ordenId", "usuarioId", "transactionId", "tipo", "montoTotal", "montoPlataforma",
                "montoNetoVendedores", "detalleVendedores", "metodoPago", "estado", "fechaTransaccion",
                "schemaVersion", "creadoEn", "actualizadoEn"
            ],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "ordenId": {"bsonType": "objectId"},
                "usuarioId": {"bsonType": "objectId"},
                "transactionId": {"bsonType": "string"},
                "tipo": {"bsonType": "string"},
                "montoTotal": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                "montoPlataforma": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                "montoNetoVendedores": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                "detalleVendedores": {
                    "bsonType": "array",
                    "minItems": 1,
                    "items": {
                        "bsonType": "object",
                        "required": [
                            "vendedorId", "montoBruto", "comision", "montoNeto", "estadoPagoVendedor", "fechaPagoVendedor"
                        ],
                        "properties": {
                            "vendedorId": {"bsonType": "objectId"},
                            "montoBruto": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                            "comision": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                            "montoNeto": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                            "estadoPagoVendedor": {"bsonType": "string"},
                            "fechaPagoVendedor": {"bsonType": "date"}
                        }
                    }
                },
                "metodoPago": {"bsonType": "string"},
                "estado": {"bsonType": "string"},
                "fechaTransaccion": {"bsonType": "date"},
                **base_common_properties
            }
        },
        "normatividades": {
            "bsonType": "object",
            "required": [
                "tipo", "titulo", "version", "contenido", "fechaPublicacion", "activo",
                "schemaVersion", "creadoEn", "actualizadoEn"
            ],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "tipo": {
                    "bsonType": "string",
                    "enum": ["privacidad", "terminos", "devoluciones", "envios", "cookies", "dpa", "cumplimiento", "otro"]
                },
                "titulo": {"bsonType": "string", "minLength": 5, "maxLength": 200},
                "version": {"bsonType": "string", "pattern": "^v?[0-9]+\\.[0-9]+(?:\\.[0-9]+)?$"},
                "contenido": {"bsonType": "string", "minLength": 20},
                "fechaPublicacion": {"bsonType": "date"},
                "activo": {"bsonType": "bool"},
                "idioma": {"bsonType": "string", "enum": ["es", "en", "pt"]},
                "fechaVigencia": {"bsonType": ["date", "null"]},
                **base_common_properties
            }
        },
        "embeddings": {
            "bsonType": "object",
            "required": ["fuente", "texto", "embedding", "activo", "schemaVersion", "creadoEn", "actualizadoEn"],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "fuente": {"bsonType": "string", "enum": ["normatividad", "producto", "categoria"]},
                "normatividadId": {"bsonType": ["objectId", "null"]},
                "productoId": {"bsonType": ["objectId", "null"]},
                "categoriaId": {"bsonType": ["objectId", "null"]},
                "texto": {"bsonType": "string", "minLength": 10},
                "embedding": {
                    "bsonType": "array",
                    "minItems": 1,
                    "items": {"bsonType": "double"}
                },
                "activo": {"bsonType": "bool"},
                **base_common_properties
            }
        },
        "consultas": {
            "bsonType": "object",
            "required": ["usuarioId", "asunto", "mensaje", "estado", "vendedorId", "schemaVersion", "creadoEn", "actualizadoEn"],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "usuarioId": {"bsonType": "objectId"},
                "asunto": {"bsonType": "string", "minLength": 5, "maxLength": 160},
                "mensaje": {"bsonType": "string", "minLength": 10},
                "estado": {"bsonType": "string", "enum": ["pendiente", "abierta", "resuelta", "cerrada", "cancelada"]},
                "vendedorId": {"bsonType": "objectId"},
                **base_common_properties
            }
        },
        "notificaciones": {
            "bsonType": "object",
            "required": [
                "destinatarioId", "destinatarioTipo", "tipo", "titulo", "mensaje", "leida",
                "schemaVersion", "creadoEn", "actualizadoEn"
            ],
            "properties": {
                "_id": {"bsonType": "objectId"},
                "destinatarioId": {"bsonType": "objectId"},
                "destinatarioTipo": {
                    "bsonType": "string",
                    "enum": ["usuario", "vendedor"]
                },
                "tipo": {
                    "bsonType": "string",
                    "enum": [
                        "nueva_orden", "orden_actualizada", "envio_iniciado", "paquete_entregado",
                        "reembolso_procesado", "mensaje_vendedor", "promocion", "revision_producto", "otro"
                    ]
                },
                "titulo": {"bsonType": "string"},
                "mensaje": {"bsonType": "string"},
                "leida": {"bsonType": "bool"},
                "referencia": {
                    "bsonType": "object",
                    "properties": {
                        "tipo": {"bsonType": "string", "enum": ["orden", "producto", "consulta", "otro"]},
                        "id": {"bsonType": "objectId"}
                    }
                },
                "datos": {
                    "bsonType": "object",
                    "properties": {
                        "numeroOrden": {"bsonType": "string"},
                        "estadoOrden": {"bsonType": "string"},
                        "cantidadProductos": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                        "estadoAnterior": {"bsonType": "string"},
                        "estadoNuevo": {"bsonType": "string"},
                        "monto": {"bsonType": ["int", "long", "double", "decimal"], "minimum": 0},
                    }
                },
                "leidaEn": {"bsonType": ["date", "null"]},
                **base_common_properties
            }
        }
    }

    return {
        "$jsonSchema": validators.get(collection_name, {"bsonType": "object"})
    }


# ====================== CREACIÓN DE COLECCIONES CON VALIDATOR ======================
collections = [
    "usuarios", "vendedores", "productos", "categorias", "ordenes",
    "resenas", "transacciones", "normatividades", "embeddings", "consultas", "notificaciones"
]

for coll_name in collections:
    try:
        validator = get_validator(coll_name)
        db.create_collection(
            coll_name,
            validator=validator,
            validationAction="error",      # Rechaza documentos inválidos
            validationLevel="strict"
        )
        logger.info("Colección creada: %s", coll_name)
    except CollectionInvalid:
        # La colección ya existe → actualizamos el validator
        try:
            db.command({
                "collMod": coll_name,
                "validator": get_validator(coll_name),
                "validationAction": "error",
                "validationLevel": "strict"
            })
            logger.info("Validator actualizado en colección existente: %s", coll_name)
        except OperationFailure as exc:
            logger.error("Error al actualizar validator en %s: %s", coll_name, exc)
    except OperationFailure as exc:
        logger.error("Error en operación MongoDB para colección %s: %s", coll_name, exc)


# ====================== CREACIÓN DE ÍNDICES ======================
def create_indexes():
    logger.info("Creando índices...")

    # Usuarios
    db.usuarios.create_index("email", unique=True, name="idx_email_unique")
    db.usuarios.create_index("roles")
    db.usuarios.create_index("vendedorId")

    # Vendedores
    db.vendedores.create_index("usuarioId", unique=True, name="idx_usuarioId_unique")
    db.vendedores.create_index("tiendaSlug", unique=True, name="idx_tiendaSlug_unique")
    db.vendedores.create_index("estado")

    # Productos
    db.productos.create_index([("vendedorId", ASCENDING), ("activo", ASCENDING)], name="idx_vendedor_activo")
    db.productos.create_index("categoriaId")
    db.productos.create_index("precio")
    db.productos.create_index([("nombre", "text"), ("descripcion", "text")], name="idx_text_search")

    # Categorías
    db.categorias.create_index("estado")

    # Órdenes
    db.ordenes.create_index([("usuarioId", ASCENDING), ("creadoEn", DESCENDING)])
    db.ordenes.create_index("pago.estado")

    # Reseñas
    db.resenas.create_index([("productoId", ASCENDING), ("calificacion", DESCENDING)])
    db.resenas.create_index("vendedorId")

    # Transacciones
    db.transacciones.create_index("ordenId", unique=True, name="idx_transacciones_ordenId_unique")
    db.transacciones.create_index("transactionId", unique=True, name="idx_transacciones_transactionId_unique")
    db.transacciones.create_index("estado")
    db.transacciones.create_index([("usuarioId", ASCENDING), ("fechaTransaccion", DESCENDING)], name="idx_transacciones_usuario_fecha")
    db.transacciones.create_index("detalleVendedores.vendedorId")

    # Normatividades
    db.normatividades.create_index(
        [("tipo", ASCENDING), ("version", ASCENDING), ("idioma", ASCENDING)],
        unique=True,
        name="idx_normatividad_tipo_version_idioma_unique"
    )
    db.normatividades.create_index("activo")
    db.normatividades.create_index("fechaPublicacion")

    # Embeddings (importante para Vector Search)
    db.embeddings.create_index([("fuente", ASCENDING), ("activo", ASCENDING)])
    db.embeddings.create_index("normatividadId")
    db.embeddings.create_index("productoId")
    db.embeddings.create_index("categoriaId")
    db.embeddings.create_index([("fuente", ASCENDING), ("normatividadId", ASCENDING)], name="idx_embedding_fuente_normatividad")
    db.embeddings.create_index([("fuente", ASCENDING), ("productoId", ASCENDING)], name="idx_embedding_fuente_producto")
    db.embeddings.create_index([("fuente", ASCENDING), ("categoriaId", ASCENDING)], name="idx_embedding_fuente_categoria")
    
    # Nota: El índice Vector Search se recomienda crearlo desde Atlas UI o con db.command()

    # Consultas
    db.consultas.create_index([("usuarioId", ASCENDING), ("creadoEn", DESCENDING)])
    db.consultas.create_index("estado")
    db.consultas.create_index("vendedorId")

    # Notificaciones
    db.notificaciones.create_index(
        [("destinatarioTipo", ASCENDING), ("destinatarioId", ASCENDING), ("creadoEn", DESCENDING)],
        name="idx_notificaciones_destinatario_creadoEn"
    )
    db.notificaciones.create_index("destinatarioId")
    db.notificaciones.create_index("destinatarioTipo")
    db.notificaciones.create_index("tipo")
    db.notificaciones.create_index("leida")
    db.notificaciones.create_index("referencia.id")

    logger.info("Todos los índices creados (o ya existían)")

def main() -> None:
    try:
        create_indexes()
        logger.info("Setup completado exitosamente")
        logger.info("Base de datos: %s", DATABASE_NAME)
        logger.info("Colecciones creadas/actualizadas con validación e índices")
        logger.info(
            "Regla de validación cruzada recomendada en embeddings: "
            "si fuente=producto -> productoId obligatorio; "
            "si fuente=normatividad -> normatividadId obligatorio; "
            "si fuente=categoria -> categoriaId obligatorio"
        )
    except ConnectionFailure as exc:
        logger.error("No se pudo conectar a MongoDB: %s", exc)
        raise
    except OperationFailure as exc:
        logger.error("Operación MongoDB fallida durante setup: %s", exc)
        raise
    except PyMongoError as exc:
        logger.error("Error general de PyMongo durante setup: %s", exc)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()