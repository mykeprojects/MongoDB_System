import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

# Cargar variables desde el archivo .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("environment_config")

def verificar_conexion():
    uri = os.getenv("MONGO_URI")

    if not uri:
        raise ValueError("La variable MONGO_URI debe estar definida en el archivo .env")
    
    # Intentamos la conexión
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    
    try:
        # 1. El 'ping' es la prueba definitiva de conexión
        client.admin.command('ping')
        
        # 2. Obtenemos información del servidor de forma más compatible
        server_info = client.server_info()

        is_master_info = client.admin.command('isMaster')
        
        # El nombre del replica set suele tener el formato: atlas-<random>-shard-0
        replica_set = is_master_info.get('setName', 'N/A (Standalone)')

        logger.info("--- Verificación de Conexión ---")
        logger.info("Estado: Conexión establecida exitosamente")
        logger.info("Replica Set Name: %s", replica_set)
        logger.info("Nodo principal: %s", client.address[0])
        logger.info("Puerto: %s", client.address[1])
        logger.info("Versión de MongoDB: %s", server_info.get('version'))
        # 3. Listar bases de datos para confirmar permisos de lectura
        dbs = client.list_database_names()
        logger.info("Bases de datos accesibles: %s", ', '.join(dbs))
        logger.info("%s", "-" * 40)

    except ConnectionFailure as exc:
        logger.error("No se pudo contactar al servidor (Timeout): %s", exc)
        raise
    except OperationFailure as e:
        logger.error("Error de autenticación: usuario o contraseña incorrectos. Detalle: %s", e)
        raise
    finally:
        client.close()

if __name__ == "__main__":
    verificar_conexion()