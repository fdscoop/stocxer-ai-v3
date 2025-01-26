import os
from market_app import create_app as create_market_app

def create_app():
    """Create and configure the Flask application for Heroku"""
    config = {
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'development_secret_key'),
        'DEBUG': False,
        'TESTING': False
    }
    return create_market_app(config)

# For local development
if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)