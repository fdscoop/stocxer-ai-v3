from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketDataConfig:
    """Configuration parameters for market data analysis"""
    risk_free_rate: float = 0.07
    vix_threshold: float = 20.0
    default_expiry_days: int = 30
    
class MarketDataValidator:
    """Standalone validator class for market data validation"""
    
    @staticmethod
    def validate_options_data(options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean options market data"""
        try:
            if not options_data:
                logger.warning("Empty options data received")
                return {}

            # Validate option chain structure
            chain_structure = options_data.get('option_chain_structure', {})
            active_calls = chain_structure.get('active_calls', 0)
            active_puts = chain_structure.get('active_puts', 0)

            if active_calls == 0 or active_puts == 0:
                calls = options_data.get('calls', [])
                puts = options_data.get('puts', [])
                
                # Recalculate active options
                active_calls = sum(1 for c in calls if c.get('openInterest', 0) > 0)
                active_puts = sum(1 for p in puts if p.get('openInterest', 0) > 0)
                
                options_data['option_chain_structure'] = {
                    'active_calls': active_calls,
                    'active_puts': active_puts,
                    'total_strikes': len(set(
                        [c.get('strikePrice', 0) for c in calls] +
                        [p.get('strikePrice', 0) for p in puts]
                    ))
                }

            return options_data
        except Exception as e:
            logger.error(f"Options data validation error: {e}")
            return {}

class MarketAnalyzer:
    """Primary class for market analysis"""
    
    def __init__(self, market_data: Dict[str, Any], config: Optional[MarketDataConfig] = None):
        self.market_data = market_data
        self.config = config or MarketDataConfig()
        self.current_market = market_data.get('current_market', {})
        self.historical_data = market_data.get('historical_data', {})
        self.options_data = market_data.get('options', {})

    def analyze_market_structure(self, payload):  # Add payload parameter
        try:
            current_market = payload.get('current_market', {}).get('index', {})
            vix_data = payload.get('current_market', {}).get('vix', {})
            return {
                'price_levels': self._analyze_price_levels(current_market),
                'trend_analysis': self._analyze_trend(current_market),
                'volatility': self._analyze_volatility(current_market, vix_data)
            }
        except Exception as e:
            logger.error(f"Market structure analysis error: {e}")
            return {}
    def analyze_technical_indicators(self, payload) -> Dict[str, Any]:
        """Analyze technical indicators using price data"""
        try:
            index_history = payload.get('historical_data', {}).get('index', [])
            
            if not index_history:
                return {'error': 'Insufficient historical data'}

            df = self._prepare_price_dataframe(index_history)
            
            # Calculate indicators
            sma_20 = df['close'].rolling(window=20).mean()
            sma_50 = df['close'].rolling(window=50).mean()
            rsi = self._calculate_rsi(df['close'])
            
            return {
                'moving_averages': self._calculate_moving_averages(df, sma_20, sma_50),
                'momentum_indicators': self._calculate_momentum_indicators(df, rsi),
                'candlestick_analysis': self._analyze_candlestick_patterns(df)
            }
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return {'error': str(e)}

    def _prepare_price_dataframe(self, index_history: List[Dict]) -> pd.DataFrame:
        """Prepare price data for analysis"""
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime(entry['timestamp']),
            'open': float(entry['price_data']['open']),
            'high': float(entry['price_data']['high']),
            'low': float(entry['price_data']['low']),
            'close': float(entry['price_data']['close'])
        } for entry in index_history])
        
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].interpolate(method='linear')
        return df

    def _calculate_rsi(self, price_series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = price_series.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze candlestick patterns"""
        patterns = {}
        
        # Bullish engulfing
        bullish_engulfing = (df['close'] > df['open'].shift(1)) & \
                           (df['open'] < df['close'].shift(1))
        patterns['bullish_engulfing'] = int(bullish_engulfing.sum())
        
        # Bearish engulfing
        bearish_engulfing = (df['close'] < df['open'].shift(1)) & \
                           (df['open'] > df['close'].shift(1))
        patterns['bearish_engulfing'] = int(bearish_engulfing.sum())
        
        return patterns

    def _calculate_moving_averages(self, 
                                 df: pd.DataFrame, 
                                 sma_20: pd.Series, 
                                 sma_50: pd.Series) -> Dict[str, float]:
        """Calculate moving average indicators"""
        last_close = float(df['close'].iloc[-1])
        last_sma_20 = float(sma_20.iloc[-1])
        last_sma_50 = float(sma_50.iloc[-1])
        
        return {
            'sma_20': last_sma_20,
            'sma_50': last_sma_50,
            'price_vs_sma20': last_close - last_sma_20,
            'price_vs_sma50': last_close - last_sma_50
        }

    def _calculate_momentum_indicators(self, 
                                    df: pd.DataFrame, 
                                    rsi: pd.Series) -> Dict[str, Any]:
        """Calculate momentum-based indicators"""
        momentum = df['close'].pct_change(periods=10)
        last_momentum = float(momentum.iloc[-1])
        last_rsi = float(rsi.iloc[-1])
        
        trend_strength = 'Strong' if abs(last_momentum) > 0.05 else 'Moderate'
        trend_direction = 'Bullish' if last_momentum > 0 else 'Bearish'
        
        return {
            '10_day_momentum': last_momentum,
            'rsi': last_rsi,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction
        }

    def _analyze_price_levels(self, current_market: Dict[str, Any]) -> Dict[str, float]:
        """Analyze current price levels"""
        return {
            'current': current_market.get('ltp', 0),
            'high': current_market.get('high', 0),
            'low': current_market.get('low', 0),
            'open': current_market.get('open', 0),
            'prev_close': current_market.get('close', 0)
        }

    def _analyze_trend(self, current_market: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price trends"""
        ltp = current_market.get('ltp', 0)
        open_price = current_market.get('open', 0)
        
        return {
            'intraday': {
                'change': ltp - open_price,
                'change_percent': current_market.get('percentChange', 0),
                'direction': 'Bullish' if ltp >= open_price else 'Bearish'
            },
            'overall': {
                'net_change': current_market.get('netChange', 0),
                'net_change_percent': current_market.get('percentChange', 0)
            }
        }

    def _analyze_volatility(self, 
                          current_market: Dict[str, Any], 
                          vix_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market volatility"""
        high = current_market.get('high', 0)
        low = current_market.get('low', 0)
        
        return {
            'market_range': {
                'day_high': high,
                'day_low': low,
                'range_percent': abs(high - low) / low * 100 if low else 0
            },
            'vix_current': vix_data.get('ltp', 0),
            'vix_change': vix_data.get('netChange', 0),
            'vix_percent_change': vix_data.get('percentChange', 0)
        }


from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.stats import norm
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class OptionParameters:
    """Parameters for option calculations"""
    spot_price: float
    strike_price: float
    time_to_expiry: float
    volatility: float
    option_type: str = 'call'
    risk_free_rate: float = 0.07

    def validate(self) -> bool:
        """Validate option parameters"""
        try:
            return all([
                self.spot_price > 0,
                self.strike_price > 0,
                self.time_to_expiry >= 0,
                self.volatility > 0,
                self.option_type in ['call', 'put'],
                self.risk_free_rate >= 0
            ])
        except Exception:
            return False

@dataclass
class Greeks:
    """Container for option Greeks values"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    @classmethod
    def create_expired(cls, option_type: str) -> 'Greeks':
        """Create Greeks for expired options"""
        return cls(
            delta=1.0 if option_type == 'call' else -1.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0
        )

class OptionsGreeksCalculator:
    """Calculate and analyze options Greeks"""

    def __init__(self, risk_free_rate: float = 0.07):
        self.risk_free_rate = risk_free_rate

    def calculate_greeks(self, params: OptionParameters) -> Greeks:
        """Calculate all Greeks for an option"""
        try:
            if not params.validate():
                raise ValueError("Invalid option parameters")

            if params.time_to_expiry <= 0:
                return Greeks.create_expired(params.option_type)

            d1, d2 = self._calculate_d1_d2(params)
            
            delta = self._calculate_delta(d1, params.option_type)
            gamma = self._calculate_gamma(d1, params)
            theta = self._calculate_theta(d1, d2, params)
            vega = self._calculate_vega(d1, params)
            rho = self._calculate_rho(d2, params)

            return Greeks(delta, gamma, theta, vega, rho)

        except Exception as e:
            logger.error(f"Greeks calculation error: {e}")
            return Greeks.create_expired(params.option_type)

    def _calculate_d1_d2(self, params: OptionParameters) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters for Black-Scholes"""
        d1 = (np.log(params.spot_price / params.strike_price) +
              (self.risk_free_rate + 0.5 * params.volatility ** 2) * 
              params.time_to_expiry) / \
             (params.volatility * np.sqrt(params.time_to_expiry))
        
        d2 = d1 - params.volatility * np.sqrt(params.time_to_expiry)
        return d1, d2

    def _calculate_delta(self, d1: float, option_type: str) -> float:
        """Calculate option delta"""
        return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)

    def _calculate_gamma(self, d1: float, params: OptionParameters) -> float:
        """Calculate option gamma"""
        return norm.pdf(d1) / (params.spot_price * params.volatility * 
                             np.sqrt(params.time_to_expiry))

    def _calculate_theta(self, d1: float, d2: float, params: OptionParameters) -> float:
        """Calculate option theta"""
        first_term = -params.spot_price * norm.pdf(d1) * params.volatility / \
                    (2 * np.sqrt(params.time_to_expiry))
        
        second_term = self.risk_free_rate * params.strike_price * \
                     np.exp(-self.risk_free_rate * params.time_to_expiry)
        
        if params.option_type == 'call':
            return first_term - second_term * norm.cdf(d2)
        else:
            return first_term + second_term * norm.cdf(-d2)

    def _calculate_vega(self, d1: float, params: OptionParameters) -> float:
        """Calculate option vega"""
        return params.spot_price * np.sqrt(params.time_to_expiry) * \
               norm.pdf(d1) / 100

    def _calculate_rho(self, d2: float, params: OptionParameters) -> float:
        """Calculate option rho"""
        factor = 1 if params.option_type == 'call' else -1
        return factor * params.strike_price * params.time_to_expiry * \
               np.exp(-self.risk_free_rate * params.time_to_expiry) * \
               norm.cdf(factor * d2) / 100

class IndexOptionsAnalyzer:
    """Analyze index options with futures data integration"""

    def __init__(self, greeks_calculator: OptionsGreeksCalculator):
        self.greeks_calculator = greeks_calculator

    def analyze_options_chain(self, 
                            current_price: float,
                            options_chain: Dict[str, List[Dict]],
                            futures_data: Dict[str, Any],
                            vix: float) -> Dict[str, Any]:
        """Analyze full options chain with Greeks and market data"""
        try:
            # Generate optimal options
            optimal_options = self._select_optimal_options(
                current_price, options_chain, futures_data, vix
            )

            # Calculate activity metrics
            activity_metrics = self._calculate_activity_metrics(
                options_chain, futures_data
            )

            return {
                'optimal_options': optimal_options,
                'activity_metrics': activity_metrics,
                'market_metrics': self._calculate_market_metrics(activity_metrics),
                'risk_metrics': self._calculate_risk_metrics(optimal_options, vix)
            }

        except Exception as e:
            logger.error(f"Options chain analysis error: {e}")
            return {}

    def _select_optimal_options(self,
                              current_price: float,
                              options_chain: Dict[str, List[Dict]],
                              futures_data: Dict[str, Any],
                              vix: float) -> Dict[str, List[Dict]]:
        """Select optimal options based on multiple criteria"""
        try:
            base_strikes = self._generate_base_strikes(current_price)
            selected_options = {'calls': [], 'puts': []}

            for strike in base_strikes:
                for option_type in ['calls', 'puts']:
                    params = OptionParameters(
                        spot_price=current_price,
                        strike_price=strike,
                        time_to_expiry=7/365,  # 1 week expiry
                        volatility=vix/100,
                        option_type=option_type[:-1]
                    )

                    greeks = self.greeks_calculator.calculate_greeks(params)
                    theo_price = self._calculate_theoretical_price(params)
                    entry_zones = self._calculate_entry_zones(
                        theo_price, greeks, current_price, vix
                    )

                    selected_options[option_type].append({
                        'strike_price': strike,
                        'theoretical_price': theo_price,
                        'greeks': greeks.__dict__,
                        'entry_zones': entry_zones,
                        'expiry': '2025-02-01',
                        'volume': futures_data.get('volume', 0) * 0.1,
                        'openInterest': futures_data.get('oi', 0) * 0.1
                    })

            return selected_options

        except Exception as e:
            logger.error(f"Optimal options selection error: {e}")
            return {'calls': [], 'puts': []}

    def _generate_base_strikes(self, current_price: float) -> List[float]:
        """Generate base strike prices around current price"""
        return [round(current_price * (1 + x/100)) for x in range(-2, 3)]

    def _calculate_theoretical_price(self, params: OptionParameters) -> float:
        """Calculate theoretical option price"""
        greeks = self.greeks_calculator.calculate_greeks(params)
        
        if params.option_type == 'call':
            intrinsic = max(params.spot_price - params.strike_price, 0)
        else:
            intrinsic = max(params.strike_price - params.spot_price, 0)
            
        time_value = greeks.theta * params.time_to_expiry
        return intrinsic + time_value

    def _calculate_entry_zones(self,
                             theo_price: float,
                             greeks: Greeks,
                             current_price: float,
                             vix: float) -> Dict[str, Any]:
        """Calculate entry and exit zones for options"""
        volatility_factor = vix/20
        delta = abs(greeks.delta)

        return {
            'entry': {
                'low': theo_price * (1 - 0.03 * volatility_factor),
                'high': theo_price * (1 + 0.02 * volatility_factor)
            },
            'exit': {
                'stop_loss': theo_price * (1 - 0.08 * volatility_factor),
                'target': theo_price * (1 + 0.12 * volatility_factor)
            },
            'greeks_limits': {
                'delta_min': -0.7 if delta < 0 else 0.3,
                'delta_max': -0.3 if delta < 0 else 0.7,
                'theta_limit': -0.15 * theo_price,
                'gamma_limit': 0.03
            }
        }

    def _calculate_activity_metrics(self,
                                  options_chain: Dict[str, List[Dict]],
                                  futures_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate options activity metrics"""
        call_oi = sum(opt.get('openInterest', 0) for opt in options_chain['calls'])
        put_oi = sum(opt.get('openInterest', 0) for opt in options_chain['puts'])
        
        return {
            'call_oi': call_oi,
            'put_oi': put_oi,
            'futures_volume': futures_data.get('volume', 0)
        }

    def _calculate_market_metrics(self, 
                                activity_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate market-wide metrics"""
        call_oi = activity_metrics['call_oi']
        put_oi = activity_metrics['put_oi']
        
        return {
            'put_call_ratio': put_oi / call_oi if call_oi > 0 else 0,
            'total_oi': call_oi + put_oi,
            'market_sentiment': self._determine_sentiment(put_oi, call_oi)
        }

    def _calculate_risk_metrics(self,
                              optimal_options: Dict[str, List[Dict]],
                              vix: float) -> Dict[str, Any]:
        """Calculate risk metrics for optimal options"""
        return {
            'volatility_regime': 'High' if vix > 20 else 'Normal',
            'position_risk': self._calculate_position_risk(optimal_options),
            'market_risk': self._calculate_market_risk(vix)
        }

    def _determine_sentiment(self, put_oi: float, call_oi: float) -> str:
        """Determine market sentiment from options data"""
        if call_oi == 0:
            return 'Neutral'
            
        pcr = put_oi / call_oi
        if pcr > 1.5:
            return 'Bearish'
        elif pcr < 0.7:
            return 'Bullish'
        return 'Neutral'

    def _calculate_position_risk(self,
                               optimal_options: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate position-specific risk metrics"""
        max_delta = max(
            abs(opt['greeks']['delta'])
            for options in optimal_options.values()
            for opt in options
        )
        
        return {
            'max_delta': max_delta,
            'risk_rating': 'High' if max_delta > 0.7 else 'Medium' if max_delta > 0.4 else 'Low'
        }

    def _calculate_market_risk(self, vix: float) -> Dict[str, Any]:
        """Calculate market-wide risk metrics"""
        return {
            'vix_level': vix,
            'risk_environment': 'High' if vix > 20 else 'Normal' if vix > 15 else 'Low',
            'position_sizing_factor': max(0.5, 1 - ((vix - 15) / 100))
        }
    
import numpy as np
from scipy.stats import norm
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class OptionParameters:
    spot_price: float
    strike_price: float 
    time_to_expiry: float
    volatility: float
    option_type: str
    risk_free_rate: float = 0.05

@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float

class OptionsGreeksCalculator:
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def calculate_greeks(self, params: OptionParameters) -> Greeks:
        if params.time_to_expiry <= 0:
            return Greeks(0, 0, 0, 0)

        d1, d2 = self._calculate_d1_d2(params)
        greeks = Greeks(
            delta=self._calculate_delta(d1, params.option_type),
            gamma=self._calculate_gamma(d1, params),
            theta=self._calculate_theta(d1, d2, params),
            vega=self._calculate_vega(d1, params)
        )
        return greeks

    def _calculate_d1_d2(self, params: OptionParameters) -> Tuple[float, float]:
        d1 = (np.log(params.spot_price / params.strike_price) + 
              (self.risk_free_rate + 0.5 * params.volatility ** 2) * params.time_to_expiry) / \
             (params.volatility * np.sqrt(params.time_to_expiry))
        d2 = d1 - params.volatility * np.sqrt(params.time_to_expiry)
        return d1, d2

    def _calculate_delta(self, d1: float, option_type: str) -> float:
        if option_type == 'call':
            return norm.cdf(d1) 
        else:
            return -norm.cdf(-d1)

    def _calculate_gamma(self, d1: float, params: OptionParameters) -> float:  
        return norm.pdf(d1) / (params.spot_price * params.volatility * np.sqrt(params.time_to_expiry))

    def _calculate_theta(self, d1: float, d2: float, params: OptionParameters) -> float:
        if params.option_type == 'call':
            return -params.spot_price * norm.pdf(d1) * params.volatility / (2 * np.sqrt(params.time_to_expiry)) - \
                   self.risk_free_rate * params.strike_price * np.exp(-self.risk_free_rate * params.time_to_expiry) * norm.cdf(d2)  
        else:
            return -params.spot_price * norm.pdf(d1) * params.volatility / (2 * np.sqrt(params.time_to_expiry)) + \
                   self.risk_free_rate * params.strike_price * np.exp(-self.risk_free_rate * params.time_to_expiry) * norm.cdf(-d2)
    
    def _calculate_vega(self, d1: float, params: OptionParameters) -> float:
        return params.spot_price * np.sqrt(params.time_to_expiry) * norm.pdf(d1) / 100
        
class IndexOptionsAnalyzer:

    def __init__(self, greeks_calculator: OptionsGreeksCalculator):
        self.greeks_calculator = greeks_calculator

    def analyze_options_chain(
        self, 
        current_price: float, 
        options_chain: Dict[str, List[Dict]], 
        vix: float
    ) -> Dict[str, Any]:
        
        optimal_options = self._select_optimal_options(current_price, options_chain, vix)
        
        # Calculate Greeks for each option
        for expiry, options in optimal_options.items():
            for option_type in ['calls', 'puts']:
                for option in options[option_type]:
                    greeks = self.greeks_calculator.calculate_greeks(OptionParameters(
                        spot_price=current_price,
                        strike_price=option['strikePrice'],
                        time_to_expiry=self._get_days_to_expiry(expiry) / 365,
                        volatility=vix/100,
                        option_type=option_type[:-1]
                    ))
                    option['greeks'] = greeks.__dict__
        
        return {
            'optimal_options': optimal_options,
            'vega': sum(opt['greeks']['vega'] for exp in optimal_options.values() for opt in exp['calls'] + exp['puts'])
        }
    
    def _select_optimal_options(
        self, 
        current_price: float,
        options_chain: Dict[str, List[Dict]],
        vix: float  
    ) -> Dict[str, List[Dict]]:
        
        selected_options = {}

        for expiry in options_chain.keys():
            
            calls = options_chain[expiry]['calls']
            puts = options_chain[expiry]['puts']
            
            atm_call = min(calls, key=lambda x: abs(x['strikePrice'] - current_price))
            atm_put = min(puts, key=lambda x: abs(x['strikePrice'] - current_price))

            selected_options[expiry] = {
                'calls': [atm_call],
                'puts': [atm_put]  
            }
            
            # Select additional strikes based on delta  
            for option in calls + puts:
                greeks = self.greeks_calculator.calculate_greeks(OptionParameters(
                    spot_price=current_price,
                    strike_price=option['strikePrice'],  
                    time_to_expiry=self._get_days_to_expiry(expiry) / 365,
                    volatility=vix/100,
                    option_type='call' if option in calls else 'put'
                ))
                
                if 0.4 <= abs(greeks.delta) <= 0.6:
                    if option in calls:
                        selected_options[expiry]['calls'].append(option)
                    else:  
                        selected_options[expiry]['puts'].append(option)

        return selected_options

    def _get_days_to_expiry(self, expiry_date: str) -> int:
        # Calculate days between now and expiry  
        pass

class OptionsStrategyGenerator:
    def __init__(self, vix_threshold: float = 20):  
        self.vix_threshold = vix_threshold

    def generate_trading_strategy(
        self,
        market_data: Dict[str, Any],
        options_data: Dict[str, Any],
        vix: float
    ) -> Dict[str, Any]:
        
        market_condition = self._extract_market_condition(market_data)
        
        primary_strategy = self._select_primary_strategy(
            market_condition, 
            options_data['optimal_options'],
            vix  
        )
        
        position_sizing = self._calculate_position_sizing(
            primary_strategy['strategy_type'],
            vix
        )
        
        risk_parameters = self._generate_risk_parameters(
            primary_strategy['greeks'],  
            options_data['vega']
        )
        
        return {
            'primary_strategy': primary_strategy,
            'position_sizing': position_sizing.__dict__,
            'risk_parameters': risk_parameters.__dict__,  
        }
        
    def _select_primary_strategy(
        self,
        market_condition: Dict[str, Any],  
        optimal_options: Dict[str, List[Dict]],
        vix: float
    ) -> Dict[str, Any]:
        
        if vix > self.vix_threshold:
            return self._create_delta_neutral_strategy(optimal_options)
        
        if market_condition['trend'] == 'Bullish':
            return self._create_bullish_strategy(optimal_options)
        elif market_condition['trend'] == 'Bearish': 
            return self._create_bearish_strategy(optimal_options)
        else:
            return self._create_range_bound_strategy(optimal_options)
            
    def _calculate_position_sizing(self, strategy_type: str, vix: float) -> PositionSizing:
        base_lots = self.base_position_size
        
        if vix > self.vix_threshold:
            base_lots //= 2  
        
        return PositionSizing(
            base_lots=base_lots,
            volatility_adjustment=min(1, self.vix_threshold / vix)
        )
    
    def _generate_risk_parameters(
        self, 
        strategy_greeks: Dict[str, float],
        strategy_vega: float  
    ) -> RiskParameters:
        
        # Adjust risk based on Greeks
        stop_loss = min(0.2, 2 * abs(strategy_greeks['theta']))
        profit_target = min(0.5, 3 * abs(strategy_greeks['delta']))

        return RiskParameters(
            stop_loss=stop_loss,  
            profit_target=profit_target,
            max_vega=strategy_vega / 10
        )

    def _extract_market_condition(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        trend = market_data['market_structure']['trend_analysis']['overall']
        momentum = market_data['technical_analysis']['momentum_indicators']
        
        return {
            'trend': 'Bullish' if trend['net_change_percent'] > 0 else 'Bearish',
            'momentum': momentum['trend_direction'],
            'strength': momentum['trend_strength']
        }
            
    def _create_bullish_strategy(
        self, 
        optimal_options: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        
        # Select call options with delta close to 0.5
        candidate_options = [
            opt for exp, strikes in optimal_options.items() 
            for opt in strikes['calls']
            if 0.45 <= opt['greeks']['delta'] <= 0.55
        ]
        
        if not candidate_options:
            return self._create_wait_strategy() 
        
        selected_option = max(candidate_options, key=lambda x: x['greeks']['delta'])
        
        return {
            'strategy_type': 'LONG_CALL',
            'leg': selected_option,
            'greeks': selected_option['greeks']
        }
        
    def _create_bearish_strategy(
        self,
        optimal_options: Dict[str, List[Dict]]  
    ) -> Dict[str, Any]:
        
        # Select put options with delta close to -0.5  
        candidate_options = [
            opt for exp, strikes in optimal_options.items()
            for opt in strikes['puts'] 
            if -0.55 <= opt['greeks']['delta'] <= -0.45
        ]
        
        if not candidate_options:
            return self._create_wait_strategy()
        
        selected_option = min(candidate_options, key=lambda x: x['greeks']['delta'])
        
        return {
            'strategy_type': 'LONG_PUT',  
            'leg': selected_option,
            'greeks': selected_option['greeks']
        }
        
    def _create_delta_neutral_strategy(
        self,
        optimal_options: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        
        # Select ATM call and put  
        candidate_calls = [
            opt for exp, strikes in optimal_options.items()
            for opt in strikes['calls']
        ]
        candidate_puts = [  
            opt for exp, strikes in optimal_options.items()
            for opt in strikes['puts']  
        ]
        
        if not candidate_calls or not candidate_puts:
            return self._create_wait_strategy() 
        
        atm_call = min(candidate_calls, key=lambda x: abs(x['greeks']['delta'] - 0.5)) 
        atm_put = min(candidate_puts, key=lambda x: abs(x['greeks']['delta'] + 0.5))
        
        return {
            'strategy_type': 'SHORT_STRANGLE',
            'legs': {  
                'call': atm_call,
                'put': atm_put
            },
            'greeks': {
                'delta': round(atm_call['greeks']['delta'] + atm_put['greeks']['delta'], 2),  
                'theta': round(atm_call['greeks']['theta'] + atm_put['greeks']['theta'], 2),
                'vega': round(atm_call['greeks']['vega'] + atm_put['greeks']['vega'], 2)
            }
        }
        
    def _create_range_bound_strategy(
        self,
        optimal_options: Dict[str, List[Dict]]  
    ) -> Dict[str, Any]:
        
        # Select slightly OTM calls and puts  
        candidate_calls = [
            opt for exp, strikes in optimal_options.items()  
            for opt in strikes['calls'] if opt['greeks']['delta'] < 0.5
        ]
        candidate_puts = [
            opt for exp, strikes in optimal_options.items()
            for opt in strikes['puts'] if opt['greeks']['delta'] > -0.5  
        ]
        
        if not candidate_calls or not candidate_puts:
            return self._create_wait_strategy()
            
        otm_call = max(candidate_calls, key=lambda x: x['greeks']['delta'])
        otm_put = min(candidate_puts, key=lambda x: x['greeks']['delta'])  

        return {
            'strategy_type': 'SHORT_STRANGLE', 
            'legs': {
                'call': otm_call, 
                'put': otm_put  
            },
            'greeks': {
                'delta': round(otm_call['greeks']['delta'] + otm_put['greeks']['delta'], 2),
                'theta': round(otm_call['greeks']['theta'] + otm_put['greeks']['theta'], 2), 
                'vega': round(otm_call['greeks']['vega'] + otm_put['greeks']['vega'], 2)
            } 
        }

    def _create_wait_strategy(self) -> Dict[str, Any]:  
        return {
            'strategy_type': 'WAIT',
            'reason': 'No suitable options found'  
        }

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Enumeration of available strategy types"""
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    WAIT = "WAIT"

@dataclass
class MarketCondition:
    """Market condition parameters"""
    trend: str
    rsi: float
    vix: float
    put_call_ratio: float
    
    @property
    def volatility_regime(self) -> str:
        """Determine volatility regime"""
        if self.vix > 25:
            return "High"
        elif self.vix > 15:
            return "Normal"
        return "Low"

@dataclass
class PositionSizing:
    """Position sizing parameters"""
    base_lots: int
    max_positions: int
    scaling_rules: str
    volatility_adjustment: float

@dataclass
class RiskParameters:
    """Risk management parameters"""
    position_loss_limit: float
    daily_loss_limit: float
    profit_targets: Dict[str, float]
    stop_loss: Dict[str, float]

class OptionsStrategyGenerator:
    """Generate trading strategies with comprehensive risk management"""

    def __init__(self, vix_threshold: float = 20.0):
        self.vix_threshold = vix_threshold
        self.base_position_size = 75  # Base lot size

    def generate_trading_strategy(self,
                                market_data: Dict[str, Any],
                                optimal_options: Dict[str, List[Dict]],
                                technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive options trading strategy"""
        try:
            # Extract market conditions
            market_condition = self._extract_market_condition(market_data, technical_analysis)
            
            # Generate primary strategy
            primary_strategy = self._select_primary_strategy(
                market_condition,
                optimal_options
            )
            
            # Generate hedge strategy if needed
            hedge_strategy = self._generate_hedge_strategy(
                primary_strategy,
                optimal_options,
                market_condition.volatility_regime == "High"
            )
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(
                primary_strategy.get('strategy_type'),
                market_condition.vix
            )
            
            # Generate risk parameters
            risk_parameters = self._generate_risk_parameters(market_condition.vix)
            
            return {
                'primary_strategy': primary_strategy,
                'hedge_strategy': hedge_strategy,
                'position_sizing': position_sizing.__dict__,
                'risk_parameters': risk_parameters.__dict__,
                'execution_guidelines': self._generate_execution_guidelines(
                    primary_strategy.get('strategy_type'),
                    market_condition.volatility_regime == "High"
                )
            }

        except Exception as e:
            logger.error(f"Strategy generation error: {e}")
            return self._generate_default_strategy()

    def _extract_market_condition(self,
                              market_data: Dict[str, Any],
                              technical_analysis: Dict[str, Any]) -> MarketCondition:
        """Extract market conditions from data"""
        # Extract VIX from market_data (which is now the full payload)
        vix = market_data.get('current_market', {}).get('vix', {}).get('ltp', 0)
        
        # Get technical indicators and options data
        momentum = technical_analysis.get('momentum_indicators', {})
        options = market_data.get('options_analysis', {})
        
        return MarketCondition(
            trend=momentum.get('trend_direction', 'Neutral'),
            rsi=momentum.get('rsi', 50),
            vix=vix,
            put_call_ratio=options.get('put_call_ratio', 1.0)
        )

    def _select_primary_strategy(self,
                               market_condition: MarketCondition,
                               optimal_options: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Select primary trading strategy based on market conditions"""
        try:
            strategies = []
            
            # Get ATM options
            atm_call = optimal_options.get('calls', [{}])[0]
            atm_put = optimal_options.get('puts', [{}])[0]
            
            # Strong Bearish Conditions
            if market_condition.trend == 'Bearish' and market_condition.rsi < 30:
                if market_condition.volatility_regime == "High":
                    strategies.append(self._create_bear_put_spread(atm_put, optimal_options))
                else:
                    strategies.append(self._create_long_put(atm_put))
            
            # Strong Bullish Conditions
            elif market_condition.trend == 'Bullish' and market_condition.rsi > 70:
                if market_condition.volatility_regime == "High":
                    strategies.append(self._create_bull_call_spread(atm_call, optimal_options))
                else:
                    strategies.append(self._create_long_call(atm_call))
            
            # Neutral Conditions with High Volatility
            elif market_condition.volatility_regime == "High":
                strategies.append(self._create_iron_condor(optimal_options, atm_call, atm_put))
            
            # Default Wait Strategy
            else:
                strategies.append(self._create_wait_strategy())

            return max(strategies, key=lambda x: 
                      {'high': 3, 'medium': 2, 'low': 1}[x['confidence']])

        except Exception as e:
            logger.error(f"Strategy selection error: {e}")
            return self._create_wait_strategy()

    def _create_bear_put_spread(self,
                               atm_put: Dict[str, Any],
                               optimal_options: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Create bear put spread strategy"""
        return {
            'strategy_type': StrategyType.BEAR_PUT_SPREAD.value,
            'primary_leg': atm_put,
            'secondary_leg': optimal_options.get('puts', [{}])[-1],
            'rationale': 'Strong bearish trend with oversold RSI in high volatility',
            'confidence': 'high'
        }

    def _create_long_put(self, atm_put: Dict[str, Any]) -> Dict[str, Any]:
        """Create long put strategy"""
        return {
            'strategy_type': StrategyType.LONG_PUT.value,
            'primary_leg': atm_put,
            'rationale': 'Strong bearish trend with oversold RSI in low volatility',
            'confidence': 'high'
        }

    def _create_bull_call_spread(self,
                                atm_call: Dict[str, Any],
                                optimal_options: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Create bull call spread strategy"""
        return {
            'strategy_type': StrategyType.BULL_CALL_SPREAD.value,
            'primary_leg': atm_call,
            'secondary_leg': optimal_options.get('calls', [{}])[-1],
            'rationale': 'Strong bullish trend with overbought RSI in high volatility',
            'confidence': 'high'
        }

    def _create_long_call(self, atm_call: Dict[str, Any]) -> Dict[str, Any]:
        """Create long call strategy"""
        return {
            'strategy_type': StrategyType.LONG_CALL.value,
            'primary_leg': atm_call,
            'rationale': 'Strong bullish trend with overbought RSI in low volatility',
            'confidence': 'high'
        }

    def _create_iron_condor(self,
                           optimal_options: Dict[str, List[Dict]],
                           atm_call: Dict[str, Any],
                           atm_put: Dict[str, Any]) -> Dict[str, Any]:
        """Create iron condor strategy"""
        return {
            'strategy_type': StrategyType.IRON_CONDOR.value,
            'call_spread': {
                'long': optimal_options.get('calls', [{}])[-1],
                'short': atm_call
            },
            'put_spread': {
                'long': optimal_options.get('puts', [{}])[-1],
                'short': atm_put
            },
            'rationale': 'High volatility with neutral trend',
            'confidence': 'medium'
        }

    def _create_wait_strategy(self) -> Dict[str, Any]:
        """Create wait strategy"""
        return {
            'strategy_type': StrategyType.WAIT.value,
            'rationale': 'Market conditions unclear or not favorable',
            'confidence': 'low'
        }

    def _generate_hedge_strategy(self,
                               primary_strategy: Dict[str, Any],
                               optimal_options: Dict[str, List[Dict]],
                               is_high_volatility: bool) -> Dict[str, Any]:
        """Generate hedge recommendations based on primary strategy"""
        strategy_type = primary_strategy.get('strategy_type')
        
        if strategy_type in [StrategyType.LONG_CALL.value, StrategyType.BULL_CALL_SPREAD.value]:
            return {
                'hedge_type': 'PROTECTIVE_PUT' if not is_high_volatility else 'PUT_SPREAD',
                'option': optimal_options.get('puts', [{}])[0],
                'sizing': '30-40% of primary position',
                'entry_timing': 'Enter hedge when delta of primary position > 0.7'
            }
        elif strategy_type in [StrategyType.LONG_PUT.value, StrategyType.BEAR_PUT_SPREAD.value]:
            return {
                'hedge_type': 'COVERED_CALL' if not is_high_volatility else 'CALL_SPREAD',
                'option': optimal_options.get('calls', [{}])[0],
                'sizing': '30-40% of primary position',
                'entry_timing': 'Enter hedge when delta of primary position < -0.7'
            }
        
        return {
            'hedge_type': 'NONE',
            'rationale': 'Primary strategy already delta-neutral'
        }

    def _calculate_position_sizing(self,
                                 strategy_type: str,
                                 vix: float) -> PositionSizing:
        """Calculate position sizing with volatility adjustment"""
        try:
            volatility_factor = max(0.5, 1 - ((vix - self.vix_threshold) / 100))
            
            position_sizes = {
                StrategyType.LONG_CALL.value: self.base_position_size,
                StrategyType.LONG_PUT.value: self.base_position_size,
                StrategyType.BULL_CALL_SPREAD.value: self.base_position_size * 1.5,
                StrategyType.BEAR_PUT_SPREAD.value: self.base_position_size * 1.5,
                StrategyType.IRON_CONDOR.value: self.base_position_size * 0.75,
                StrategyType.CALENDAR_SPREAD.value: self.base_position_size,
                StrategyType.WAIT.value: 0
            }
            
            base_lots = position_sizes.get(strategy_type, self.base_position_size)
            
            return PositionSizing(
                base_lots=int(base_lots * volatility_factor),
                max_positions=1 if strategy_type in [StrategyType.IRON_CONDOR.value,
                                                   StrategyType.CALENDAR_SPREAD.value] else 2,
                scaling_rules=self._get_scaling_rules(strategy_type),
                volatility_adjustment=volatility_factor
            )
            
        except Exception as e:
            logger.error(f"Position sizing calculation error: {e}")
            return PositionSizing(0, 0, "Error in calculation", 1.0)

    def _generate_risk_parameters(self, vix: float) -> RiskParameters:
        """Generate risk management parameters"""
        volatility_factor = vix / self.vix_threshold
        
        return RiskParameters(
            position_loss_limit=min(15 * volatility_factor, 25),
            daily_loss_limit=min(5 * volatility_factor, 10),
            profit_targets={
                'first_target': 20 * volatility_factor,
                'final_target': 35 * volatility_factor
            },
            stop_loss={
                'initial': 10 * volatility_factor,
                'trailing': 15 * volatility_factor
            }
        )

    def _get_scaling_rules(self, strategy_type: str) -> str:
        """Get specific scaling rules for each strategy type"""
        rules = {
            StrategyType.LONG_CALL.value: 'Scale in 2-3 parts on dips',
            StrategyType.LONG_PUT.value: 'Scale in 2-3 parts on rallies',
            StrategyType.BULL_CALL_SPREAD.value: 'Enter full spread position at once',
            StrategyType.BEAR_PUT_SPREAD.value: 'Enter full spread position at once',
            StrategyType.IRON_CONDOR.value: 'Enter all legs simultaneously',
            StrategyType.CALENDAR_SPREAD.value: 'Enter full position at once',
            StrategyType.WAIT.value: 'No scaling needed'
        }
        return rules.get(strategy_type, 'Enter full position at once')

    def _generate_execution_guidelines(self,
                                    strategy_type: str,
                                    is_high_volatility: bool) -> Dict[str, Any]:
        """Generate detailed execution guidelines"""
        return {
            'entry_rules': {
                'price_conditions': self._get_price_conditions(strategy_type),
                'timing_rules': self._get_timing_rules(is_high_volatility),
                'volume_conditions': self._get_volume_conditions(strategy_type)
            },
            'exit_rules': {
                'profit_taking': self._get_profit_rules(strategy_type),
                'loss_prevention': self._get_loss_rules(strategy_type),
                'greek_limits': self._get_greek_limits(strategy_type)
            },
            'trade_management': {
                'position_review_frequency': 'Hourly' if is_high_volatility else 'Daily',
                'adjustment_triggers': self._get_adjustment_triggers(),
                'rollover_rules': self._get_rollover_rules(strategy_type)
            }
        }

    def _get_price_conditions(self, strategy_type: str) -> List[str]:
        """Get price-based entry conditions"""
        conditions = {
            StrategyType.LONG_PUT.value: [
                'Enter at resistance levels',
                'Wait for price rejection',
                'Check for increased put volumes',
                'Verify bearish price action',
                'Monitor overhead resistance'
            ],
            StrategyType.LONG_CALL.value: [
                'Enter at support levels',
                'Wait for price bounce',
                'Check for increased call volumes',
                'Verify bullish price action',
                'Monitor underlying support'
            ],
            StrategyType.BULL_CALL_SPREAD.value: [
                'Enter when price crosses above MA20',
                'Verify support holds',
                'Monitor call-put ratio',
                'Check for positive momentum',
                'Confirm trend strength'
            ],
            StrategyType.BEAR_PUT_SPREAD.value: [
                'Enter when price crosses below MA20',
                'Verify resistance holds',
                'Monitor put-call ratio',
                'Check for negative momentum',
                'Confirm trend weakness'
            ],
            StrategyType.IRON_CONDOR.value: [
                'Enter when price is between major support/resistance',
                'Verify range-bound conditions',
                'Check for moderate volatility',
                'Monitor price channels',
                'Confirm low directional bias'
            ],
            StrategyType.CALENDAR_SPREAD.value: [
                'Enter at key technical levels',
                'Check for low implied volatility',
                'Verify neutral price action',
                'Monitor term structure',
                'Check for stable volatility'
            ]
        }
        return conditions.get(strategy_type, ['Default price conditions'])

    def _get_timing_rules(self, is_high_volatility: bool) -> Dict[str, Any]:
        """Get timing-based rules for entry"""
        base_rules = {
            'time_of_day': 'Avoid first and last 30 minutes of session',
            'expiry_selection': '7-15 days for directional, 20-30 days for non-directional',
            'volatility_consideration': 'Wait for VIX stabilization in high volatility'
        }
        
        if is_high_volatility:
            base_rules.update({
                'entry_timing': 'Stagger entries over 2-3 trades',
                'session_preference': 'Mid-session entries preferred',
                'volume_threshold': 'Wait for above-average volume',
                'volatility_check': 'Monitor implied volatility term structure',
                'momentum_confirmation': 'Wait for momentum stabilization'
            })
        else:
            base_rules.update({
                'entry_timing': 'Single entry on signal confirmation',
                'session_preference': 'Early session entries preferred',
                'volume_threshold': 'Normal volume sufficient',
                'volatility_check': 'Standard volatility checks',
                'momentum_confirmation': 'Regular momentum confirmation'
            })
        
        return base_rules

    def _get_volume_conditions(self, strategy_type: str) -> Dict[str, Any]:
        """Get volume-based conditions for entry"""
        base_conditions = {
            'minimum_option_volume': 100 if strategy_type in [
                StrategyType.IRON_CONDOR.value,
                StrategyType.CALENDAR_SPREAD.value
            ] else 50,
            'open_interest_threshold': 500,
            'volume_spike_threshold': '200% of average',
            'liquidity_checks': [
                'Bid-ask spread < 5%',
                'Multiple market makers present',
                'Consistent quote presence'
            ]
        }
        
        # Strategy-specific volume conditions
        if strategy_type in [StrategyType.IRON_CONDOR.value, StrategyType.CALENDAR_SPREAD.value]:
            base_conditions.update({
                'leg_volume_requirements': {
                    'minimum_leg_volume': 50,
                    'balanced_volume_ratio': 'Within 30% across legs',
                    'market_maker_presence': 'Required for all legs'
                },
                'rolling_volume_check': 'Monitor 5-day average volume',
                'spread_width_adjustment': 'Based on average daily volume'
            })
        else:
            base_conditions.update({
                'directional_volume_check': 'Monitor volume trend direction',
                'relative_volume_analysis': 'Compare to sector volume',
                'options_flow_monitoring': 'Track institutional order flow'
            })
        
        return base_conditions

    def _get_adjustment_triggers(self) -> List[str]:
        """Get position adjustment triggers"""
        return [
            'Delta neutrality breach > 20%',
            'Implied volatility change > 20%',
            'Technical trend reversal',
            'Time decay acceleration',
            'Hedge position delta change',
            'Support/resistance breach',
            'Volatility regime change',
            'Earnings/event risk',
            'Sector correlation shift',
            'Market sentiment change'
        ]

    def _get_rollover_rules(self, strategy_type: str) -> Dict[str, Any]:
        """Get position rollover rules"""
        base_rules = {
            'days_to_expiry': 5,
            'value_remaining': '25% of premium',
            'roll_conditions': [
                'Minimum credit received',
                'New strike within 1 standard deviation',
                'Sufficient liquidity in new expiry',
                'Implied volatility comparison favorable',
                'Technical alignment maintained'
            ]
        }
        
        if strategy_type in [StrategyType.IRON_CONDOR.value, 
                            StrategyType.CALENDAR_SPREAD.value]:
            base_rules.update({
                'roll_width': 'Maintain original wing width',
                'credit_requirement': '20% of margin',
                'volatility_skew_check': True,
                'term_structure_analysis': {
                    'check_forward_volatility': True,
                    'minimum_term_premium': '2%',
                    'skew_analysis_required': True
                },
                'liquidity_requirements': {
                    'minimum_open_interest': 1000,
                    'maximum_bid_ask_spread': '5%',
                    'market_maker_presence': 'Required'
                }
            })
        
        return base_rules

    def _get_profit_rules(self, strategy_type: str) -> Dict[str, Any]:
        """Get profit-taking rules"""
        base_rules = {
            StrategyType.LONG_CALL.value: {
                'first_target': '30% of position at 1.5x risk',
                'final_target': 'Trail remainder with 25% stop',
                'scaling_rules': 'Scale out in 3 parts',
                'profit_lock': 'Lock 50% at 2x risk'
            },
            StrategyType.LONG_PUT.value: {
                'first_target': '30% of position at 1.5x risk',
                'final_target': 'Trail remainder with 25% stop',
                'scaling_rules': 'Scale out in 3 parts',
                'profit_lock': 'Lock 50% at 2x risk'
            },
            StrategyType.BULL_CALL_SPREAD.value: {
                'first_target': '50% of max profit',
                'final_target': '80% of max profit',
                'time_based_exit': '75% of time decay',
                'delta_based_exit': 'When delta reaches 0.85'
            },
            StrategyType.BEAR_PUT_SPREAD.value: {
                'first_target': '50% of max profit',
                'final_target': '80% of max profit',
                'time_based_exit': '75% of time decay',
                'delta_based_exit': 'When delta reaches -0.85'
            },
            StrategyType.IRON_CONDOR.value: {
                'first_target': '40% of max profit',
                'final_target': '70% of max profit',
                'wing_adjustment': 'Adjust at 25% profit',
                'time_decay_target': '60% of theta decay'
            }
        }
        
        return base_rules.get(strategy_type, {
            'first_target': '30% of risk',
            'final_target': '50% of risk',
            'default_scaling': 'Scale out at 25%, 50%, 75% targets',
            'time_based_exit': 'Exit at 75% of time value decay'
        })

    def _get_loss_rules(self, strategy_type: str) -> Dict[str, Any]:
        """Get loss prevention rules"""
        base_rules = {
            'max_loss_single_trade': '2% of trading capital',
            'max_daily_loss': '5% of trading capital',
            'position_stop_loss': self._get_position_stop_loss(strategy_type),
            'time_stop': self._get_time_stop(strategy_type),
            'volatility_based_stops': {
                'vix_increase_threshold': '30%',
                'implied_volatility_stop': '50% increase',
                'skew_change_stop': '25% shift'
            }
        }
        
        if strategy_type in [StrategyType.IRON_CONDOR.value, 
                            StrategyType.CALENDAR_SPREAD.value]:
            base_rules.update({
                'individual_leg_loss': '25% of max profit',
                'gamma_risk_threshold': '0.02',
                'vega_risk_threshold': '0.15',
                'wing_adjustment_stops': {
                    'delta_threshold': '0.30',
                    'gamma_threshold': '0.02',
                    'theta_decay_minimum': '0.10'
                }
            })
        
        return base_rules

    def _get_greek_limits(self, strategy_type: str) -> Dict[str, float]:
        """Get Greek-based risk limits"""
        base_limits = {
            'delta_limit': 0.30 if strategy_type in [
                StrategyType.IRON_CONDOR.value,
                StrategyType.CALENDAR_SPREAD.value
            ] else 0.70,
            'gamma_limit': 0.02,
            'theta_limit': -0.15,
            'vega_limit': 0.25
        }
        
        # Add strategy-specific adjustments
        if strategy_type == StrategyType.CALENDAR_SPREAD.value:
            base_limits.update({
                'calendar_vega_limit': 0.30,
                'calendar_theta_ratio': 2.0
            })
        elif strategy_type == StrategyType.IRON_CONDOR.value:
            base_limits.update({
                'wing_delta_limit': 0.15,
                'total_gamma_limit': 0.03
            })
        
        return base_limits
    def _get_position_stop_loss(self, strategy_type: str) -> str:
        """Get position-specific stop loss rules"""
        stops = {
            StrategyType.LONG_CALL.value: '30% of premium',
            StrategyType.LONG_PUT.value: '30% of premium',
            StrategyType.BULL_CALL_SPREAD.value: '60% of max loss',
            StrategyType.BEAR_PUT_SPREAD.value: '60% of max loss',
            StrategyType.IRON_CONDOR.value: '2x max profit',
            StrategyType.CALENDAR_SPREAD.value: '40% of premium'
        }
        return stops.get(strategy_type, '50% of risk')

    def _get_time_stop(self, strategy_type: str) -> str:
        """Get time-based stop loss rules"""
        time_stops = {
            StrategyType.LONG_CALL.value: '2 days against trend',
            StrategyType.LONG_PUT.value: '2 days against trend',
            StrategyType.BULL_CALL_SPREAD.value: '5 days without progress',
            StrategyType.BEAR_PUT_SPREAD.value: '5 days without progress',
            StrategyType.IRON_CONDOR.value: '50% of time value decay',
            StrategyType.CALENDAR_SPREAD.value: 'Front month expiry'
        }
        return time_stops.get(strategy_type, '5 days without progress')





from flask import Flask, request, jsonify
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
from dataclasses import dataclass
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response structure"""
    status: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        response = {'status': self.status}
        if self.data is not None:
            response['data'] = self.data
        if self.message is not None:
            response['message'] = self.message
        if self.error is not None:
            response['error'] = self.error
        return response

def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # Initialize components
    market_analyzer = MarketAnalyzer()
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
    def health_check() -> Tuple[Dict[str, Any], int]:
        """Health check endpoint"""
        return jsonify(APIResponse(
            status='success',
            message='Service is healthy'
        ).to_dict()), 200

    @app.route('/api/v1/analyze', methods=['POST'])
    @validate_request
    @log_request
    def analyze_market() -> Tuple[Dict[str, Any], int]:
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
    def generate_strategy() -> Tuple[Dict[str, Any], int]:
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
    def not_found(error) -> Tuple[Dict[str, Any], int]:
        """Handle 404 errors"""
        return jsonify(APIResponse(
            status='error',
            error='Resource not found'
        ).to_dict()), 404

    @app.errorhandler(405)
    def method_not_allowed(error) -> Tuple[Dict[str, Any], int]:
        """Handle 405 errors"""
        return jsonify(APIResponse(
            status='error',
            error='Method not allowed'
        ).to_dict()), 405

    @app.errorhandler(500)
    def internal_server_error(error) -> Tuple[Dict[str, Any], int]:
        """Handle 500 errors"""
        return jsonify(APIResponse(
            status='error',
            error='Internal server error'
        ).to_dict()), 500

    return app

def validate_market_data(data: Dict[str, Any]) -> bool:
    try:
        required_keys = ['historical_data', 'current_market']
        if not all(key in data for key in required_keys):
            logger.warning("Missing required keys")
            return False

        index_data = data.get('historical_data', {}).get('index', [])
        if not index_data or not all(key in index_data[0].get('price_data', {})
                                   for key in ['open', 'high', 'low', 'close']):
            logger.warning("Invalid price data structure")
            return False

        return True
    except Exception as e:
        logger.error(f"Payload validation error: {e}")
        return False

def validate_strategy_request(data: Dict[str, Any]) -> bool:
    """Validate strategy generation request"""
    try:
        # Check for essential keys
        if not data or not isinstance(data, dict):
            logger.warning("Invalid or empty request data")
            return False

        # Ensure market structure and other key data are present
        if not data.get('market_structure') or not data.get('technical_analysis'):
            logger.warning("Missing market structure or technical analysis data")
            return False

        return True
    except Exception as e:
        logger.error(f"Strategy request validation error: {str(e)}")
        return False