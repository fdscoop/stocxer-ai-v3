from flask import Flask, request, jsonify
import os
import logging
from market_analysis import (
    MarketAnalyzer, 
    IndexOptionsAnalyzer, 
    OptionsGreeksCalculator, 
    OptionsStrategyGenerator,
    APIResponse,
    validate_market_data,
    validate_strategy_request
)
from datetime import datetime
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # Initialize components
    market_analyzer = MarketAnalyzer({})  # Pass an empty dict as initial market data
    options_analyzer = IndexOptionsAnalyzer(OptionsGreeksCalculator())
    strategy_generator = OptionsStrategyGenerator()
    
    def validate_request(f):
        """Request validation decorator"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                if not request.is_json:
                    return jsonify(APIResponse(
                        status='error',
                        error='Content-Type must be application/json'
                    ).to_dict()), 400
                
                data = request.get_json()
                if not data:
                    return jsonify(APIResponse(
                        status='error',
                        error='Request body cannot be empty'
                    ).to_dict()), 400
                
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Request validation error: {str(e)}")
                return jsonify(APIResponse(
                    status='error',
                    error='Invalid request format'
                ).to_dict()), 400
        return decorated_function

    def log_request(f):
        """Request logging decorator"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            request_id = str(int(time.time() * 1000))
            logger.info(f"Request {request_id} started: {request.path}")
            
            start_time = time.time()
            response = f(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(
                f"Request {request_id} completed: {request.path} "
                f"Duration: {duration:.2f}s Status: {response[1]}"
            )
            return response
        return decorated_function

    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify(APIResponse(
            status='success',
            message='Service is healthy'
        ).to_dict()), 200

    @app.route('/api/v1/analyze', methods=['POST'])
    @validate_request
    @log_request
    def analyze_market():
        """Market analysis endpoint"""
        try:
            payload = request.get_json()
            
            # Validate market data
            if not validate_market_data(payload):
                return jsonify(APIResponse(
                    status='error',
                    error='Invalid market data format'
                ).to_dict()), 400

            # Perform market analysis
            market_structure = market_analyzer.analyze_market_structure(payload)
            technical_analysis = market_analyzer.analyze_technical_indicators(payload)
            
            # Analyze options
            options_data = options_analyzer.analyze_options_chain(
                payload.get('current_market', {}).get('index', {}).get('ltp', 0),
                payload.get('options', {}),
                payload.get('futures', {}),
                payload.get('current_market', {}).get('vix', {}).get('ltp', 0)
            )
            
            # Generate trading strategy
            trading_strategy = strategy_generator.generate_trading_strategy(
                payload,
                options_data.get('optimal_options', {}),
                technical_analysis
            )
            
            # Prepare response
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'market_structure': market_structure,
                'technical_analysis': technical_analysis,
                'options_analysis': options_data,
                'trading_strategy': trading_strategy
            }
            
            return jsonify(APIResponse(
                status='success',
                data=analysis_results
            ).to_dict()), 200

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return jsonify(APIResponse(
                status='error',
                error='Analysis processing error'
            ).to_dict()), 500

    @app.route('/api/v1/strategy', methods=['POST'])
    @validate_request
    @log_request
    def generate_strategy():
        """Strategy generation endpoint"""
        try:
            payload = request.get_json()
            
            # Validate strategy request
            if not validate_strategy_request(payload):
                return jsonify(APIResponse(
                    status='error',
                    error='Invalid strategy request format'
                ).to_dict()), 400

            # Generate trading strategy
            strategy = strategy_generator.generate_trading_strategy(
                payload.get('market_data', {}),
                payload.get('optimal_options', {}),
                payload.get('technical_analysis', {})
            )
            
            return jsonify(APIResponse(
                status='success',
                data=strategy
            ).to_dict()), 200

        except Exception as e:
            logger.error(f"Strategy generation error: {str(e)}")
            return jsonify(APIResponse(
                status='error',
                error='Strategy generation error'
            ).to_dict()), 500

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return jsonify(APIResponse(
            status='error',
            error='Resource not found'
        ).to_dict()), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors"""
        return jsonify(APIResponse(
            status='error',
            error='Method not allowed'
        ).to_dict()), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 errors"""
        return jsonify(APIResponse(
            status='error',
            error='Internal server error'
        ).to_dict()), 500

    return app