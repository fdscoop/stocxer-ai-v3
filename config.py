# 3. Enhanced configuration management
# config.py
import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-please-change')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'production')
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    DATABASE_URL = os.environ.get('DATABASE_URL')
    REDIS_URL = os.environ.get('REDIS_URL')
    
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    
class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True