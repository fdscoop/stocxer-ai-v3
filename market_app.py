from flask import Flask, request, jsonify
import os
import logging
from datetime import datetime
import time
from functools import wraps
from typing import Dict, Any

from market_analysis import (
    MarketAnalyzer,
    MarketAnalysisService,
    OptionsDataAnalyzer,
    OptionsGreeksCalculator,
    MarketDataConfig,
    APIResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_market_data(payload: Dict[str, Any]) -> bool:
    try:
        # Retain existing base structure validation
        required_components = [
            'current_market',
            'historical_data',
            'market_metrics',
            'options_structure',
            'summary'
        ]
        if not all(comp in payload for comp in required_components):
            logger.error(f"Missing required components in payload")
            return False

        # Existing specific validations (keep these)
        current_market = payload['current_market']
        required_market_fields = ['index', 'vix', 'futures', 'options']
        if not all(field in current_market for field in required_market_fields):
            logger.error(f"Missing required fields in current_market")
            return False

        # Existing historical data validation (keep this)
        historical_data = payload['historical_data']
        required_historical = ['index', 'vix', 'futures']
        if not all(data in historical_data for data in required_historical):
            logger.error(f"Missing required historical data")
            return False

        # Existing options structure base validation (keep this)
        options_structure = payload['options_structure']
        if not all(key in options_structure for key in ['options', 'byExpiry']):
            logger.error(f"Invalid options structure")
            return False

        # Existing market metrics validation (keep this)
        market_metrics = payload['market_metrics']
        required_metrics = ['volume_pcr', 'oi_pcr', 'total_volumes', 'total_oi']
        if not all(metric in market_metrics for metric in required_metrics):
            logger.error(f"Missing required market metrics")
            return False

        # NEW: Enhanced Options Structure Validation
        if 'byExpiry' not in options_structure:
            logger.error("Missing 'byExpiry' in options structure")
            return False
        
        for expiry, expiry_data in options_structure['byExpiry'].items():
            # Validate expiry data structure
            if not isinstance(expiry_data, dict):
                logger.error(f"Invalid expiry data for {expiry}")
                return False
            
            # Ensure both 'calls' and 'puts' keys exist
            calls = expiry_data.get('calls', {})
            puts = expiry_data.get('puts', {})
            
            # Log warnings for incomplete data
            if not calls and not puts:
                logger.warning(f"No option data found for expiry {expiry}")
            
            # Validate individual strike details
            for option_type, options in [('calls', calls), ('puts', puts)]:
                for strike, strike_data in options.items():
                    required_keys = ['symbol', 'strikePrice', 'optionType', 'exchange']
                    if not all(key in strike_data for key in required_keys):
                        logger.error(f"Incomplete {option_type} data for strike {strike}")
                        return False

        return True
    
    except Exception as e:
        logger.error(f"Comprehensive payload validation error: {e}")
        return False

class MarketAnalysisAPI:
    """API Service for market analysis"""
    
    def __init__(self):
        self.config = MarketDataConfig()
        self.analysis_service = MarketAnalysisService()

    def analyze_market(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process market analysis request"""
        try:
            logger.info("Starting market analysis")
            
            # Validate full payload structure
            if not validate_market_data(payload):
                raise ValueError("Invalid market data format")
            
            # Extract all required components
            current_market = payload['current_market']
            historical_data = payload['historical_data']
            market_metrics = payload['market_metrics']
            options_structure = payload['options_structure']
            
            # Perform analysis using all components
            analysis_results = self.analysis_service.analyze_market({
                'current_market': current_market,
                'historical_data': historical_data,
                'market_metrics': market_metrics,
                'options_structure': options_structure
            })
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            raise

    def generate_strategy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process strategy generation request"""
        try:
            logger.info("Starting strategy generation")
            
            if not validate_market_data(payload):
                raise ValueError("Invalid market data format")
            
            # Extract required components
            technical_analysis = payload.get('technical_analysis', {})
            market_structure = payload.get('market_structure', {})
            
            # Generate strategy
            strategy = self.analysis_service.generate_trading_strategy(
                technical_analysis,
                market_structure,
                payload
            )
            
            logger.info("Strategy generation completed successfully")
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy generation error: {str(e)}", exc_info=True)
            raise

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # Initialize API service
    api_service = MarketAnalysisAPI()
    
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
                'description': 'Market Analysis API',
                'endpoints': {
                    'health': '/health',
                    'analyze': '/api/v1/analyze',
                    'strategy': '/api/v1/strategy'
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
            
            # Perform analysis
            analysis_results = api_service.analyze_market(payload)
            
            return jsonify(APIResponse(
                status='success',
                data=analysis_results
            ).to_dict()), 200
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify(APIResponse(
                status='error',
                error=str(e)
            ).to_dict()), 400
        except Exception as e:
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
            
            # Generate strategy
            strategy = api_service.generate_strategy(payload)
            
            return jsonify(APIResponse(
                status='success',
                data=strategy
            ).to_dict()), 200
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify(APIResponse(
                status='error',
                error=str(e)
            ).to_dict()), 400
        except Exception as e:
            logger.error(f"Strategy generation error: {str(e)}", exc_info=True)
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = bool(os.environ.get('FLASK_DEBUG', True))
    
    app = create_app()
    app.run(host='0.0.0.0', port=port, debug=debug)