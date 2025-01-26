# 4. Add health check with database/redis connection status
# health_check.py
from flask import jsonify
import redis
import psycopg2
from datetime import datetime

def check_database_connection(db_url):
    try:
        conn = psycopg2.connect(db_url)
        conn.close()
        return True
    except:
        return False

def check_redis_connection(redis_url):
    try:
        r = redis.from_url(redis_url)
        r.ping()
        return True
    except:
        return False

def get_health_status():
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'database': check_database_connection(os.getenv('DATABASE_URL')),
            'redis': check_redis_connection(os.getenv('REDIS_URL')),
            'api': True
        }
    }
    
    if not all(health_data['components'].values()):
        health_data['status'] = 'degraded'
        
    return health_data