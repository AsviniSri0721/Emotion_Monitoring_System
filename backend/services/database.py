import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Database connection pool
connection_pool = None
db_type = os.getenv('DB_TYPE', 'mysql').lower()

# Import appropriate database driver
use_mysql_connector = False

if db_type == 'mysql':
    try:
        import mysql.connector
        from mysql.connector import pooling
        use_mysql_connector = True
        logger.info("Using mysql-connector-python driver")
    except ImportError:
        try:
            import pymysql
            pymysql.install_as_MySQLdb()
            from MySQLdb import connect
            from MySQLdb.cursors import DictCursor
            logger.info("Using MySQL/MariaDB driver (pymysql)")
        except ImportError:
            logger.error("MySQL driver not found. Install: pip install pymysql or pip install mysql-connector-python")
            raise
else:
    try:
        import psycopg2
        from psycopg2 import pool
        logger.info("Using PostgreSQL driver")
    except ImportError:
        logger.error("PostgreSQL driver not found. Install: pip install psycopg2-binary")
        raise

def init_db():
    """Initialize database connection pool"""
    global connection_pool
    
    try:
        if db_type == 'mysql':
            if use_mysql_connector:
                connection_pool = mysql.connector.pooling.MySQLConnectionPool(
                    pool_name="emotion_pool",
                    pool_size=5,
                    pool_reset_session=True,
                    host=os.getenv('DB_HOST', 'localhost'),
                    database=os.getenv('DB_NAME', 'emotiondb'),
                    user=os.getenv('DB_USER', 'root'),
                    password=os.getenv('DB_PASSWORD', ''),
                    port=int(os.getenv('DB_PORT', '3306'))
                )
            else:
                # For pymysql, we'll create connections on demand
                connection_pool = 'pymysql'
        else:
            # PostgreSQL
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                20,
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'emotion_monitoring'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', ''),
                port=os.getenv('DB_PORT', '5432')
            )
        
        if connection_pool:
            logger.info(f"Database connection pool created successfully ({db_type})")
            
            # Test connection
            conn = get_connection()
            if conn:
                conn.close()
                logger.info("Database connection test successful")
        else:
            logger.error("Failed to create database connection pool")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def get_connection():
    """Get a connection from the pool"""
    try:
        if db_type == 'mysql':
            if use_mysql_connector:
                return connection_pool.get_connection()
            else:
                # Create new connection for pymysql
                return connect(
                    host=os.getenv('DB_HOST', 'localhost'),
                    db=os.getenv('DB_NAME', 'emotiondb'),
                    user=os.getenv('DB_USER', 'root'),
                    passwd=os.getenv('DB_PASSWORD', ''),
                    port=int(os.getenv('DB_PORT', '3306'))
                )
        else:
            return connection_pool.getconn()
    except Exception as e:
        logger.error(f"Error getting database connection: {str(e)}")
        return None

def return_connection(conn):
    """Return a connection to the pool"""
    try:
        if db_type == 'mysql':
            if use_mysql_connector:
                conn.close()
            else:
                conn.close()
        else:
            connection_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Error returning database connection: {str(e)}")

def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """Execute a database query"""
    conn = get_connection()
    if not conn:
        logger.error("No database connection available")
        raise Exception("Database connection failed")
    
    try:
        cursor = conn.cursor()
        
        # Log query for debugging (remove in production)
        logger.debug(f"Executing query: {query[:100]}...")
        if params:
            logger.debug(f"With params: {params}")
        
        cursor.execute(query, params)
        
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = cursor.rowcount
        
        cursor.close()
        return result
        
    except Exception as e:
        conn.rollback()
        error_msg = str(e)
        logger.error(f"Database query error: {error_msg}")
        logger.error(f"Query: {query[:200]}")
        if params:
            logger.error(f"Params: {params}")
        raise Exception(f"Database error: {error_msg}")
    finally:
        return_connection(conn)

