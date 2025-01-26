import os
import logging
from typing import Dict, Any, Optional
from flask import Flask
from market_app import create_app as market_create_app  # Updated import
from flask_compress import Compress
from flask_talisman import Talisman
from werkzeug.middleware.proxy_fix import ProxyFix
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-please-change')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'production')
    DEBUG = False
    TESTING = False
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max-limit
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = False
    
    # Rate limiting
    RATELIMIT_DEFAULT = "200 per day"
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL')
    
    # Custom settings
    API_VERSION = '1.0'
    MARKET_DATA_CACHE_TTL = 300  # 5 minutes
    
    # Sentry configuration
    SENTRY_DSN = os.environ.get('SENTRY_DSN')
    
    # Content Security Policy
    CSP = {
        'default-src': "'self'",
        'img-src': "'self' data: https:",
        'script-src': "'self'",
        'style-src': "'self' 'unsafe-inline'",
    }

def configure_sentry(app: Flask) -> None:
    """Configure Sentry error tracking"""
    if app.config['SENTRY_DSN']:
        sentry_sdk.init(
            dsn=app.config['SENTRY_DSN'],
            integrations=[FlaskIntegration()],
            traces_sample_rate=1.0,
            environment=app.config['FLASK_ENV'],
            enable_tracing=True,
            profiles_sample_rate=1.0,
        )

def configure_security(app: Flask) -> None:
    """Configure security middleware"""
    # Enable HTTPS-only
    Talisman(
        app,
        force_https=True,
        strict_transport_security=True,
        session_cookie_secure=True,
        content_security_policy=app.config['CSP']
    )
    
    # Configure for proxy headers
    app.wsgi_app = ProxyFix(
        app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
    )

def configure_compression(app: Flask) -> None:
    """Configure response compression"""
    Compress(app)

def configure_before_request(app: Flask) -> None:
    """Configure before_request handlers"""
    @app.before_request
    def before_request():
        # You can add request preprocessing here
        pass

def configure_after_request(app: Flask) -> None:
    """Configure after_request handlers"""
    @app.after_request
    def after_request(response):
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response

def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application for Heroku"""
    
    # Initialize configuration
    if config is None:
        config = {}
    
    base_config = {
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'development_secret_key'),
        'DEBUG': bool(os.environ.get('FLASK_DEBUG', False)),
        'TESTING': False
    }
    base_config.update(config)
    
    try:
        # Create app instance using the imported function
        app = market_create_app(base_config)
        
        # Load configuration
        app.config.from_object(Config)
        
        # Configure components
        configure_sentry(app)
        configure_security(app)
        configure_compression(app)
        configure_before_request(app)
        configure_after_request(app)
        
        # Log successful initialization
        logger.info(
            f"Application initialized successfully in {app.config['FLASK_ENV']} mode"
        )
        
        return app
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

def configure_workers() -> Dict[str, Any]:
    """Configure Gunicorn workers based on environment"""
    web_concurrency = os.environ.get('WEB_CONCURRENCY', 3)
    return {
        'workers': int(web_concurrency),
        'worker_class': 'gevent',
        'threads': 2,
        'timeout': 30,
        'keepalive': 2,
        'max_requests': 1000,
        'max_requests_jitter': 50
    }

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = bool(os.environ.get('FLASK_DEBUG', True))
    
    # Create and run app
    app = create_app()
    
    # Configure host based on environment
    host = '0.0.0.0' if not debug else 'localhost'
    
    try:
        logger.info(f"Starting development server on port {port}")
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start development server: {str(e)}")
        raise