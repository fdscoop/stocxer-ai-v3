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

class MarketAnalysisService:
    """Service class to handle market analysis components"""
    def __init__(self):
        self.options_analyzer = IndexOptionsAnalyzer(OptionsGreeksCalculator())
        self.strategy_generator = OptionsStrategyGenerator()

    def analyze_market(self, payload: dict) -> dict:
        """Process market analysis"""
        try:
            logger.info("Starting market analysis with payload structure")
            logger.info(f"Payload keys: {list(payload.keys())}")
            
            # Extract market data
            current_market = payload.get('current_market', {})
            
            # Extract specific components
            index_data = current_market.get('index', {})
            vix_data = current_market.get('vix', {})
            futures_data = current_market.get('futures', {})
            options_data = current_market.get('options', {})
            
            logger.info("Market data validation:")
            logger.info(f"Index data present: {bool(index_data)}")
            logger.info(f"VIX data present: {bool(vix_data)}")
            logger.info(f"Futures data present: {bool(futures_data)}")
            logger.info(f"Options data present: {bool(options_data)}")

            # Validate required fields
            if not all([index_data, vix_data, options_data]):
                raise ValueError("Missing required market data components")

            # Extract and convert LTP values
            try:
                index_ltp = float(index_data.get('ltp', 0))
                vix_ltp = float(vix_data.get('ltp', 0))
                logger.info(f"LTP values - Index: {index_ltp}, VIX: {vix_ltp}")
            except (TypeError, ValueError) as e:
                logger.error(f"Error converting LTP values: {e}")
                raise ValueError(f"Invalid LTP values: index={index_data.get('ltp')}, vix={vix_data.get('ltp')}")

            # Create analyzer instance
            logger.info("Creating MarketAnalyzer instance")
            market_analyzer = MarketAnalyzer(payload)
            
            # Perform market analysis
            logger.info("Starting market structure analysis")
            market_structure = market_analyzer.analyze_market_structure(payload)
            
            logger.info("Starting technical analysis")
            technical_analysis = market_analyzer.analyze_technical_indicators(payload)
            
            logger.info("Processing options chain")
            logger.info(f"Processing options chain - Parameters: index_ltp={index_ltp}, vix_ltp={vix_ltp}")
            
            # Process options chain with correct number of arguments
            options_chain = self.options_analyzer.analyze_options_chain(
                index_ltp,
                options_data,
                vix_ltp
            )
            
            logger.info("Generating trading strategy")
            trading_strategy = self.strategy_generator.generate_trading_strategy(
                payload,
                options_chain.get('optimal_options', {}),
                technical_analysis
            )
            
            # Prepare analysis results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'market_structure': market_structure,
                'technical_analysis': technical_analysis,
                'options_analysis': options_chain,
                'trading_strategy': trading_strategy,
                'market_metrics': payload.get('market_metrics', {}),
                'summary': {
                    'index_price': index_ltp,
                    'vix_level': vix_ltp,
                    'trend': market_structure.get('trend_analysis', {}).get('overall', {}).get('direction', 'Neutral'),
                    'pcr_volume': payload.get('market_metrics', {}).get('volume_pcr', 0),
                    'pcr_oi': payload.get('market_metrics', {}).get('oi_pcr', 0)
                }
            }
            
            logger.info("Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            logger.error("Error details:", exc_info=True)
            raise

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # Initialize service
    analysis_service = MarketAnalysisService()
    
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

    @app.route('/')
    def index():
        """Root endpoint with API information"""
        return jsonify(APIResponse(
            status='success',
            data={
                'api_version': '1.0',
                'description': 'StocXer AI Market Analysis API',
                'endpoints': {
                    'health': {
                        'path': '/health',
                        'method': 'GET',
                        'description': 'Health check endpoint'
                    },
                    'analyze': {
                        'path': '/api/v1/analyze',
                        'method': 'POST',
                        'description': 'Market analysis endpoint'
                    },
                    'strategy': {
                        'path': '/api/v1/strategy',
                        'method': 'POST',
                        'description': 'Strategy generation endpoint'
                    }
                }
            }
        ).to_dict()), 200

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
            logger.info("Received analysis request")
            
            # Perform analysis
            analysis_results = analysis_service.analyze_market(payload)
            
            return jsonify(APIResponse(
                status='success',
                data=analysis_results
            ).to_dict()), 200

        except ValueError as e:
            # Handle validation errors
            logger.error(f"Validation error: {str(e)}")
            return jsonify(APIResponse(
                status='error',
                error='Validation error',
                details=str(e)
            ).to_dict()), 400
            
        except Exception as e:
            # Handle other errors
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return jsonify(APIResponse(
                status='error',
                error='Analysis processing error',
                details=str(e)
            ).to_dict()), 500

    @app.route('/api/v1/strategy', methods=['POST'])
    @validate_request
    @log_request
    def generate_strategy():
        """Strategy generation endpoint"""
        try:
            payload = request.get_json()
            logger.info("Received strategy request")
            
            # Generate strategy
            strategy = analysis_service.strategy_generator.generate_trading_strategy(
                payload,
                payload.get('trading_strategy', {}).get('primary_strategy', {}),
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
                error='Strategy generation error',
                details=str(e)
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