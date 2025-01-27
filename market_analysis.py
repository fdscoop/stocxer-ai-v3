from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from dataclasses import dataclass
from scipy.stats import norm
import talib as ta

logger = logging.getLogger(__name__)

@dataclass
class MarketDataConfig:
    """Configuration parameters for Indian market analysis"""
    risk_free_rate: float = 0.07
    vix_threshold: float = 15.0  # Adjusted for Indian markets
    default_expiry_days: int = 30
    nifty_lot_size: int = 50
    min_option_volume: int = 100
    min_oi_threshold: int = 1000

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
    """Calculate and analyze options Greeks for Indian markets"""

    def __init__(self, risk_free_rate: float = 0.07):
        self.risk_free_rate = risk_free_rate

    def calculate_option_greeks(self, 
                              option_data: Dict[str, Any],
                              spot_price: float,
                              days_to_expiry: int,
                              vix_value: float,
                              option_type: str) -> Dict[str, float]:
        """Calculate Greeks using actual market data"""
        try:
            # Extract required data
            strike_price = float(option_data.get('strike', 0)) / 100  # Convert to actual strike price
            current_price = float(option_data.get('ltp', 0))
            
            # Calculate implied volatility if not provided
            volatility = vix_value / 100  # Use VIX as volatility estimate
            
            params = OptionParameters(
                spot_price=spot_price,
                strike_price=strike_price,
                time_to_expiry=days_to_expiry / 365,
                volatility=volatility,
                option_type=option_type.lower(),
                risk_free_rate=self.risk_free_rate
            )

            if not params.validate():
                return self._create_default_greeks()

            d1, d2 = self._calculate_d1_d2(params)
            
            greeks = {
                'delta': self._calculate_delta(d1, params.option_type),
                'gamma': self._calculate_gamma(d1, params),
                'theta': self._calculate_theta(d1, d2, params),
                'vega': self._calculate_vega(d1, params),
                'rho': self._calculate_rho(d2, params)
            }
            
            # Add risk analysis
            greeks['risk_metrics'] = self._analyze_greeks_risk(greeks)
            
            return greeks

        except Exception as e:
            logger.error(f"Greeks calculation error for {option_type} option: {e}")
            return self._create_default_greeks()

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
        if option_type == 'call':
            return round(norm.cdf(d1), 4)
        return round(-norm.cdf(-d1), 4)

    def _calculate_gamma(self, d1: float, params: OptionParameters) -> float:
        """Calculate option gamma"""
        gamma = norm.pdf(d1) / (params.spot_price * params.volatility * 
                             np.sqrt(params.time_to_expiry))
        return round(gamma, 4)

    def _calculate_theta(self, d1: float, d2: float, params: OptionParameters) -> float:
        """Calculate option theta"""
        first_term = -(params.spot_price * norm.pdf(d1) * params.volatility) / \
                    (2 * np.sqrt(params.time_to_expiry))
        
        second_term = self.risk_free_rate * params.strike_price * \
                     np.exp(-self.risk_free_rate * params.time_to_expiry)
        
        if params.option_type == 'call':
            theta = first_term - second_term * norm.cdf(d2)
        else:
            theta = first_term + second_term * norm.cdf(-d2)
            
        return round(theta / 365, 4)  # Daily theta

    def _calculate_vega(self, d1: float, params: OptionParameters) -> float:
        """Calculate option vega"""
        vega = params.spot_price * np.sqrt(params.time_to_expiry) * \
               norm.pdf(d1) / 100
        return round(vega, 4)

    def _calculate_rho(self, d2: float, params: OptionParameters) -> float:
        """Calculate option rho"""
        factor = 1 if params.option_type == 'call' else -1
        rho = factor * params.strike_price * params.time_to_expiry * \
              np.exp(-self.risk_free_rate * params.time_to_expiry) * \
              norm.cdf(factor * d2) / 100
        return round(rho, 4)

    def _create_default_greeks(self) -> Dict[str, float]:
        """Create default Greeks values"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'risk_metrics': {
                'delta_risk': 'Unknown',
                'gamma_risk': 'Unknown',
                'theta_risk': 'Unknown',
                'vega_risk': 'Unknown'
            }
        }

    def _analyze_greeks_risk(self, greeks: Dict[str, float]) -> Dict[str, str]:
        """Analyze risk based on Greeks values"""
        return {
            'delta_risk': self._analyze_delta_risk(greeks['delta']),
            'gamma_risk': self._analyze_gamma_risk(greeks['gamma']),
            'theta_risk': self._analyze_theta_risk(greeks['theta']),
            'vega_risk': self._analyze_vega_risk(greeks['vega'])
        }

    def _analyze_delta_risk(self, delta: float) -> str:
        """Analyze risk based on delta"""
        abs_delta = abs(delta)
        if abs_delta > 0.8:
            return 'Very High Directional Risk'
        elif abs_delta > 0.6:
            return 'High Directional Risk'
        elif abs_delta > 0.3:
            return 'Moderate Directional Risk'
        return 'Low Directional Risk'

    def _analyze_gamma_risk(self, gamma: float) -> str:
        """Analyze risk based on gamma"""
        if gamma > 0.1:
            return 'High Gamma Risk'
        elif gamma > 0.05:
            return 'Moderate Gamma Risk'
        return 'Low Gamma Risk'

    def _analyze_theta_risk(self, theta: float) -> str:
        """Analyze risk based on theta"""
        abs_theta = abs(theta)
        if abs_theta > 0.1:
            return 'High Time Decay Risk'
        elif abs_theta > 0.05:
            return 'Moderate Time Decay Risk'
        return 'Low Time Decay Risk'

    def _analyze_vega_risk(self, vega: float) -> str:
        """Analyze risk based on vega"""
        if vega > 0.2:
            return 'High Volatility Risk'
        elif vega > 0.1:
            return 'Moderate Volatility Risk'
        return 'Low Volatility Risk'
    

class MarketAnalyzer:
    """Primary class for Indian market analysis"""
    
    def __init__(self, market_data: Dict[str, Any], config: Optional[MarketDataConfig] = None):
        self.market_data = market_data
        self.config = config or MarketDataConfig()
        self.current_market = market_data.get('current_market', {})
        self.historical_data = market_data.get('historical_data', {})
        self.market_metrics = market_data.get('market_metrics', {})

    def analyze_market_structure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market structure using historical and current data"""
        try:
            # Prepare dataframes from historical data
            df_index = self._prepare_price_dataframe(self.historical_data.get('index', []))
            df_futures = self._prepare_price_dataframe(self.historical_data.get('futures', []))
            df_vix = self._prepare_price_dataframe(self.historical_data.get('vix', []))

            # Current market analysis
            current_analysis = self._analyze_current_market(
                self.current_market,
                df_index,
                df_vix
            )

            # Technical analysis
            technical_analysis = self._analyze_technical_indicators(df_index)

            # Volume and OI analysis using futures data
            volume_analysis = self._analyze_futures_data(df_futures)

            # Market metrics analysis (PCR, etc.)
            metrics_analysis = self._analyze_market_metrics(self.market_metrics)

            return {
                'current_market': current_analysis,
                'technical_analysis': technical_analysis,
                'volume_analysis': volume_analysis,
                'metrics_analysis': metrics_analysis,
                'market_strength': self._determine_market_strength(
                    current_analysis,
                    technical_analysis,
                    metrics_analysis
                )
            }

        except Exception as e:
            logger.error(f"Market structure analysis error: {e}")
            return {}

    def _prepare_price_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Prepare price data for analysis"""
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime(entry['timestamp']),
            'open': float(entry['price_data']['open']),
            'high': float(entry['price_data']['high']),
            'low': float(entry['price_data']['low']),
            'close': float(entry['price_data']['close']),
            'volume': float(entry.get('volume', 0))
        } for entry in data])
        
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].astype(float)
        return df

    def _analyze_current_market(self,
                              current_market: Dict[str, Any],
                              df_index: pd.DataFrame,
                              df_vix: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions"""
        index_data = current_market.get('index', {})
        vix_data = current_market.get('vix', {})
        
        try:
            ltp = float(index_data.get('ltp', 0))
            vix = float(vix_data.get('ltp', 0))
            
            return {
                'index': {
                    'ltp': ltp,
                    'change': float(index_data.get('netChange', 0)),
                    'change_percent': float(index_data.get('percentChange', 0)),
                    'day_high': float(index_data.get('high', 0)),
                    'day_low': float(index_data.get('low', 0))
                },
                'vix': {
                    'value': vix,
                    'change': float(vix_data.get('netChange', 0)),
                    'regime': self._determine_volatility_regime(vix)
                },
                'trend': {
                    'intraday': self._determine_intraday_trend(index_data),
                    'short_term': self._determine_short_term_trend(df_index),
                    'volatility_trend': self._analyze_volatility_trend(df_vix)
                }
            }
        except Exception as e:
            logger.error(f"Current market analysis error: {e}")
            return {}

    def _analyze_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        try:
            # Trend Indicators
            df['ema_20'] = ta.EMA(df['close'], timeperiod=20)
            df['ema_50'] = ta.EMA(df['close'], timeperiod=50)
            df['ema_200'] = ta.EMA(df['close'], timeperiod=200)
            
            # Momentum Indicators
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            macd, signal, hist = ta.MACD(df['close'], 
                                       fastperiod=12, 
                                       slowperiod=26, 
                                       signalperiod=9)
            
            # Volatility Indicators
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20)
            
            # Supertrend (custom calculation for Indian markets)
            df['supertrend'] = self._calculate_supertrend(df)
            
            latest = df.iloc[-1]
            latest_macd = macd.iloc[-1]
            latest_signal = signal.iloc[-1]
            
            return {
                'trend_indicators': {
                    'ema_20': latest['ema_20'],
                    'ema_50': latest['ema_50'],
                    'ema_200': latest['ema_200'],
                    'supertrend': latest['supertrend'],
                    'trend_direction': self._determine_trend_direction(latest)
                },
                'momentum_indicators': {
                    'rsi': latest['rsi'],
                    'macd': {
                        'macd_line': latest_macd,
                        'signal_line': latest_signal,
                        'histogram': latest_macd - latest_signal
                    }
                },
                'volatility_indicators': {
                    'atr': latest['atr'],
                    'bollinger_bands': {
                        'upper': upper.iloc[-1],
                        'middle': middle.iloc[-1],
                        'lower': lower.iloc[-1]
                    }
                },
                'support_resistance': self._calculate_support_resistance(df)
            }
            
        except Exception as e:
            logger.error(f"Technical indicators calculation error: {e}")
            return {}

    def _calculate_supertrend(self, df: pd.DataFrame, 
                            period: int = 10, 
                            multiplier: float = 3.0) -> pd.Series:
        """Calculate Supertrend indicator (popular in Indian markets)"""
        try:
            atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            
            # Calculate basic upper and lower bands
            basic_upper = (df['high'] + df['low']) / 2 + multiplier * atr
            basic_lower = (df['high'] + df['low']) / 2 - multiplier * atr
            
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            for i in range(period, len(df)):
                if df['close'].iloc[i] > basic_upper.iloc[i-1]:
                    direction.iloc[i] = 1
                elif df['close'].iloc[i] < basic_lower.iloc[i-1]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
                    
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = basic_lower.iloc[i]
                else:
                    supertrend.iloc[i] = basic_upper.iloc[i]
                    
            return supertrend
            
        except Exception as e:
            logger.error(f"Supertrend calculation error: {e}")
            return pd.Series(index=df.index)

    def _analyze_futures_data(self, df_futures: pd.DataFrame) -> Dict[str, Any]:
        """Analyze futures data for volume and OI trends"""
        try:
            # Calculate volume and OI metrics
            df_futures['volume_sma'] = df_futures['volume'].rolling(window=10).mean()
            df_futures['volume_ratio'] = df_futures['volume'] / df_futures['volume_sma']
            
            latest = df_futures.iloc[-1]
            
            return {
                'volume_analysis': {
                    'current_volume': latest['volume'],
                    'volume_sma': latest['volume_sma'],
                    'volume_ratio': latest['volume_ratio'],
                    'volume_trend': self._determine_volume_trend(latest['volume_ratio'])
                },
                'basis_analysis': self._analyze_futures_basis(
                    self.current_market.get('index', {}).get('ltp', 0),
                    self.current_market.get('futures', {}).get('ltp', 0)
                )
            }
            
        except Exception as e:
            logger.error(f"Futures data analysis error: {e}")
            return {}

    def _analyze_market_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market-wide metrics including PCR"""
        try:
            pcr_volume = metrics.get('volume_pcr', 0)
            pcr_oi = metrics.get('oi_pcr', 0)
            
            total_volumes = metrics.get('total_volumes', {})
            total_oi = metrics.get('total_oi', {})
            
            return {
                'put_call_ratios': {
                    'volume_pcr': pcr_volume,
                    'oi_pcr': pcr_oi,
                    'pcr_interpretation': self._interpret_pcr(pcr_volume, pcr_oi)
                },
                'volume_analysis': {
                    'total_call_volume': total_volumes.get('calls', 0),
                    'total_put_volume': total_volumes.get('puts', 0),
                    'volume_skew': self._calculate_volume_skew(total_volumes)
                },
                'oi_analysis': {
                    'total_call_oi': total_oi.get('calls', 0),
                    'total_put_oi': total_oi.get('puts', 0),
                    'oi_skew': self._calculate_oi_skew(total_oi)
                }
            }
            
        except Exception as e:
            logger.error(f"Market metrics analysis error: {e}")
            return {}

    def _determine_volatility_regime(self, vix: float) -> str:
        """Determine volatility regime based on VIX"""
        if vix > 20:
            return 'High Volatility'
        elif vix > 15:
            return 'Moderate Volatility'
        return 'Low Volatility'

    def _determine_market_strength(self,
                                 current_analysis: Dict[str, Any],
                                 technical_analysis: Dict[str, Any],
                                 metrics_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall market strength"""
        try:
            # Extract key indicators
            trend = technical_analysis.get('trend_indicators', {})
            momentum = technical_analysis.get('momentum_indicators', {})
            pcr = metrics_analysis.get('put_call_ratios', {})
            
            # Calculate strength score
            trend_score = self._calculate_trend_score(trend)
            momentum_score = self._calculate_momentum_score(momentum)
            sentiment_score = self._calculate_sentiment_score(pcr)
            
            total_score = (trend_score + momentum_score + sentiment_score) / 3
            
            return {
                'strength_score': total_score,
                'market_condition': self._interpret_market_condition(total_score),
                'contributing_factors': {
                    'trend_score': trend_score,
                    'momentum_score': momentum_score,
                    'sentiment_score': sentiment_score
                }
            }
            
        except Exception as e:
            logger.error(f"Market strength determination error: {e}")
            return {}

    def _calculate_trend_score(self, trend: Dict[str, Any]) -> float:
        """Calculate trend strength score"""
        try:
            ema_20 = float(trend.get('ema_20', 0))
            ema_50 = float(trend.get('ema_50', 0))
            ema_200 = float(trend.get('ema_200', 0))
            
            if ema_20 > ema_50 > ema_200:
                return 1.0
            elif ema_20 < ema_50 < ema_200:
                return 0.0
            else:
                return 0.5
                
        except Exception:
            return 0.5

    def _calculate_momentum_score(self, momentum: Dict[str, Any]) -> float:
        """Calculate momentum strength score"""
        try:
            rsi = float(momentum.get('rsi', 50))
            macd = momentum.get('macd', {})
            macd_hist = float(macd.get('histogram', 0))
            
            rsi_score = (rsi - 50) / 50  # Normalized RSI score
            macd_score = 1 if macd_hist > 0 else -1
            
            return (rsi_score + macd_score) / 2
            
        except Exception:
            return 0.0

    def _calculate_sentiment_score(self, pcr: Dict[str, Any]) -> float:
        """Calculate sentiment score based on PCR"""
        try:
            volume_pcr = float(pcr.get('volume_pcr', 1))
            oi_pcr = float(pcr.get('oi_pcr', 1))
            
            # Normalize PCR scores (1.0 is neutral)
            volume_score = 1 - (volume_pcr - 1)
            oi_score = 1 - (oi_pcr - 1)
            
            return (volume_score + oi_score) / 2
            
        except Exception:
            return 0.5

    def _interpret_market_condition(self, strength_score: float) -> str:
        """Interpret market condition based on strength score"""
        if strength_score > 0.7:
            return 'Strong Bullish'
        elif strength_score > 0.5:
            return 'Moderately Bullish'
        elif strength_score > 0.3:
            return 'Moderately Bearish'
        else:
            return 'Strong Bearish'

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            # Use last 20 days for level calculation
            recent_df = df.tail(20)
            
            # Find pivot points
            pivot = (recent_df['high'].iloc[-1] + recent_df['low'].iloc[-1] + 
                    recent_df['close'].iloc[-1]) / 3
            
            r1 = 2 * pivot - recent_df['low'].iloc[-1]
            s1 = 2 * pivot - recent_df['high'].iloc[-1]
            r2 = pivot + (recent_df['high'].iloc[-1] - recent_df['low'].iloc[-1])
            s2 = pivot - (recent_df['high'].iloc[-1] - recent_df['low'].iloc[-1])
            
            return {
                'pivot': round(pivot, 2),
                'r1': round(r1, 2),
                'r2': round(r2, 2),
                's1': round(s1, 2),
                's2': round(s2, 2)
            }
        except Exception as e:
            logger.error(f"Support/Resistance calculation error: {e}")
            return {}

    def _analyze_futures_basis(self, spot_price: float, futures_price: float) -> Dict[str, Any]:
        """Analyze futures basis and its implications"""
        try:
            basis = futures_price - spot_price
            basis_percent = (basis / spot_price) * 100
            
            return {
                'basis': round(basis, 2),
                'basis_percent': round(basis_percent, 2),
                'interpretation': self._interpret_basis(basis_percent)
            }
        except Exception as e:
            logger.error(f"Futures basis analysis error: {e}")
            return {}

    def _interpret_basis(self, basis_percent: float) -> str:
        """Interpret futures basis"""
        if basis_percent > 0.5:
            return 'Strong Contango'
        elif basis_percent > 0.2:
            return 'Moderate Contango'
        elif basis_percent < -0.5:
            return 'Strong Backwardation'
        elif basis_percent < -0.2:
            return 'Moderate Backwardation'
        return 'Normal Basis'

    def _interpret_pcr(self, volume_pcr: float, oi_pcr: float) -> str:
        """Interpret Put-Call Ratio"""
        avg_pcr = (volume_pcr + oi_pcr) / 2
        
        if avg_pcr > 1.5:
            return 'Extremely Bearish'
        elif avg_pcr > 1.2:
            return 'Moderately Bearish'
        elif avg_pcr < 0.7:
            return 'Extremely Bullish'
        elif avg_pcr < 0.8:
            return 'Moderately Bullish'
        return 'Neutral'

class MarketAnalysisService:
    """Service class for comprehensive market analysis"""
    
    def __init__(self):
        self.config = MarketDataConfig()
        self.greeks_calculator = OptionsGreeksCalculator()
    
    def analyze_market(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process comprehensive market analysis for Indian markets"""
        try:
            logger.info("Starting market analysis")
            
            # Initialize analyzers
            market_analyzer = MarketAnalyzer(payload, self.config)
            options_analyzer = OptionsDataAnalyzer(self.config)
            
            # Extract key data points
            current_market = payload.get('current_market', {})
            options_structure = payload.get('options_structure', {})
            market_metrics = payload.get('market_metrics', {})
            
            # Market structure analysis
            market_structure = market_analyzer.analyze_market_structure(payload)
            
            # Options analysis
            options_analysis = self._analyze_options(
                current_market,
                options_structure,
                market_metrics
            )
            
            # Generate trading opportunities
            trading_opportunities = self._generate_trading_opportunities(
                market_structure,
                options_analysis,
                market_metrics
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'market_structure': market_structure,
                'options_analysis': options_analysis,
                'trading_opportunities': trading_opportunities,
                'summary': self._create_summary(
                    market_structure,
                    options_analysis,
                    trading_opportunities,
                    market_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            raise

    def _analyze_options(self,
                        current_market: Dict[str, Any],
                        options_structure: Dict[str, Any],
                        market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze options data"""
        try:
            index_ltp = float(current_market.get('index', {}).get('ltp', 0))
            vix_value = float(current_market.get('vix', {}).get('ltp', 0))
            
            expiry_analysis = {}
            available_expiries = options_structure.get('options', {}).get('byExpiry', {})
            
            for expiry, expiry_data in available_expiries.items():
                expiry_analysis[expiry] = self._analyze_single_expiry(
                    expiry,
                    expiry_data,
                    index_ltp,
                    vix_value
                )
            
            return {
                'expiry_analysis': expiry_analysis,
                'overall_metrics': self._calculate_overall_metrics(
                    expiry_analysis,
                    market_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Options analysis error: {e}")
            return {}
    
    def _calculate_overall_metrics(self,
                             expiry_analysis: Dict[str, Any],
                             market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall options metrics"""
        try:
            return {
                'put_call_ratios': {
                    'volume': market_metrics.get('volume_pcr', 0),
                    'oi': market_metrics.get('oi_pcr', 0)
                },
                'volumes': {
                    'total_call_volume': market_metrics.get('total_volumes', {}).get('calls', 0),
                    'total_put_volume': market_metrics.get('total_volumes', {}).get('puts', 0)
                },
                'open_interest': {
                    'total_call_oi': market_metrics.get('total_oi', {}).get('calls', 0),
                    'total_put_oi': market_metrics.get('total_oi', {}).get('puts', 0)
                }
            }

    
    def _analyze_single_option(self,
                             option_data: Dict[str, Any],
                             option_type: str,
                             index_ltp: float,
                             days_to_expiry: int,
                             vix_value: float) -> Dict[str, Any]:
        """Analyze individual option contract"""
        try:
            if not option_data:
                return {}
                
            # Calculate Greeks
            greeks = self.greeks_calculator.calculate_option_greeks(
                option_data,
                index_ltp,
                days_to_expiry,
                vix_value,
                option_type
            )
            
            return {
                'price': float(option_data.get('ltp', 0)),
                'volume': int(option_data.get('tradeVolume', 0)),
                'oi': int(option_data.get('opnInterest', 0)),
                'greeks': greeks,
                'depth': self._analyze_market_depth(
                    option_data.get('depth', {})
                )
            }
            
        except Exception as e:
            logger.error(f"Single option analysis error: {e}")
            return {}

    def _analyze_market_depth(self, depth: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market depth data"""
        try:
            buy_orders = depth.get('buy', [])
            sell_orders = depth.get('sell', [])
            
            return {
                'bid_ask_spread': self._calculate_spread(buy_orders, sell_orders),
                'buy_sell_ratio': self._calculate_depth_ratio(buy_orders, sell_orders),
                'liquidity_score': self._calculate_liquidity_score(buy_orders, sell_orders)
            }
        except Exception as e:
            logger.error(f"Market depth analysis error: {e}")
            return {}

    def _calculate_days_to_expiry(self, expiry: str) -> int:
        """Calculate days to expiry"""
        try:
            expiry_date = datetime.strptime(expiry, '%d%b%Y')
            current_date = datetime.now()
            return (expiry_date - current_date).days
        except Exception:
            return 0

    def _generate_trading_opportunities(self,
                                     market_structure: Dict[str, Any],
                                     options_analysis: Dict[str, Any],
                                     market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading opportunities"""
        try:
            trend = market_structure.get('market_strength', {}).get('market_condition', 'Neutral')
            
            opportunities = {
                'directional': self._find_directional_trades(
                    trend,
                    options_analysis,
                    market_metrics
                ),
                'non_directional': self._find_non_directional_trades(
                    options_analysis,
                    market_metrics
                ),
                'hedging': self._find_hedging_opportunities(
                    trend,
                    options_analysis
                )
            }
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Trading opportunities generation error: {e}")
            return {}

    def _create_summary(self,
                       market_structure: Dict[str, Any],
                       options_analysis: Dict[str, Any],
                       trading_opportunities: Dict[str, Any],
                       market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis summary"""
        try:
            return {
                'market_condition': market_structure.get('market_strength', {}).get('market_condition'),
                'optimal_strategies': self._get_optimal_strategies(trading_opportunities),
                'risk_metrics': {
                    'current_vix': options_analysis.get('vix_value', 0),
                    'pcr_volume': market_metrics.get('volume_pcr', 0),
                    'pcr_oi': market_metrics.get('oi_pcr', 0)
                },
                'key_levels': market_structure.get('technical_analysis', {}).get('support_resistance', {})
            }
        except Exception as e:
            logger.error(f"Summary creation error: {e}")
            return {}

class OptionsDataAnalyzer:
    def analyze_options_chain(self, current_price, options_structure, vix):
        try:
            # Robust input validation
            if not isinstance(options_structure, dict):
                raise ValueError("Invalid options structure")
            
            # Safe data extraction with type checking
            options_chain = options_structure.get('options', {}).get('byExpiry', {})
            
            if not options_chain:
                logger.warning("No options data available")
                return {}
            
            # Process with explicit type handling
            expiry_analysis = {}
            for expiry, expiry_data in options_chain.items():
                if isinstance(expiry_data, dict):
                    expiry_analysis[expiry] = self._analyze_single_expiry(
                        expiry, expiry_data, current_price, vix
                    )
            
            return expiry_analysis
        
        except Exception as e:
            logger.error(f"Options chain analysis error: {e}", exc_info=True)
            return {}
        
    def _analyze_single_expiry(self, 
                                expiry: str,
                                expiry_data: Dict[str, Any],
                                current_price: float,
                                market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensively analyze options for a specific expiration date.
        
        Args:
            expiry: Expiration date string
            expiry_data: Dictionary containing options data
            current_price: Current market price
            market_metrics: Market-wide metrics
        
        Returns:
            Comprehensive analysis of options for the specified expiry
        """
        try:
            # Validate input data structure
            if not isinstance(expiry_data, dict):
                logger.warning(f"Invalid expiry data structure for {expiry}")
                return {}

            # Safely extract calls and puts
            calls = expiry_data.get('calls', {})
            puts = expiry_data.get('puts', {})

            # Identify unique strikes
            call_strikes = set(calls.keys())
            put_strikes = set(puts.keys())
            all_strikes = call_strikes.union(put_strikes)

            # Comprehensive strikes analysis
            strikes_analysis = {}
            for strike in all_strikes:
                call_data = calls.get(strike, {})
                put_data = puts.get(strike, {})

                # Skip if no data for the strike
                if not call_data and not put_data:
                    logger.info(f"No data for strike {strike} in expiry {expiry}")
                    continue

                strike_analysis = self._analyze_strike(
                    strike,
                    call_data,
                    put_data,
                    current_price
                )
                
                # Only add if analysis successful
                if strike_analysis:
                    strikes_analysis[strike] = strike_analysis

            # Compute expiry-level metrics
            expiry_metrics = self._calculate_expiry_metrics(
                calls, puts, market_metrics
            )

            # Identify optimal strikes
            optimal_strikes = self._select_optimal_strikes(
                strikes_analysis,
                current_price
            )

            return {
                'strikes_analysis': strikes_analysis,
                'expiry_metrics': expiry_metrics,
                'optimal_strikes': optimal_strikes,
                'total_strikes': len(strikes_analysis),
                'expiry_date': expiry
            }
        
        except Exception as e:
            logger.error(
                f"Comprehensive expiry analysis error for {expiry}: {e}", 
                exc_info=True
            )
            return {
                'error': str(e),
                'expiry': expiry,
                'error_type': type(e).__name__
            }

    def _analyze_strike(self,
                    strike: str,
                    call_data: Dict[str, Any],
                    put_data: Dict[str, Any],
                    current_price: float) -> Dict[str, Any]:
        """
        Analyze individual strike price options.
        
        Args:
            strike: Strike price as a string
            call_data: Call option data dictionary
            put_data: Put option data dictionary
            current_price: Current market price
        
        Returns:
            Comprehensive analysis of the strike's options
        """
        try:
            # Validate and convert strike price
            try:
                strike_price = float(strike)
            except (TypeError, ValueError):
                logger.warning(f"Invalid strike price: {strike}")
                return {}

            # Validate current price
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                logger.warning(f"Invalid current price: {current_price}")
                return {}

            # Comprehensive strike analysis
            return {
                'strike_price': strike_price,
                'moneyness': self._calculate_moneyness(strike_price, current_price),
                'call': self._process_option_data(call_data, 'CALL'),
                'put': self._process_option_data(put_data, 'PUT'),
                'spread_analysis': self._analyze_strike_spread(call_data, put_data)
            }
        
        except Exception as e:
            logger.error(f"Strike analysis error for strike {strike}: {e}")
            return {}

    def _process_option_data(self, option_data: Dict[str, Any], option_type: str) -> Dict[str, Any]:
        """
        Process and enhance individual option data.
        
        Args:
            option_data: Raw option data dictionary
            option_type: Type of option (CALL/PUT)
        
        Returns:
            Processed and enriched option data
        """
        if not option_data:
            return {}

        return {
            'ltp': float(option_data.get('ltp', 0)),
            'volume': int(option_data.get('tradeVolume', 0)),
            'open_interest': int(option_data.get('opnInterest', 0)),
            'depth': option_data.get('depth', {}),
            'option_type': option_type,
            'implied_volatility': option_data.get('impliedVolatility', None)
        }

    def _analyze_strike_spread(self, call_data: Dict[str, Any], put_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze spread between call and put options.
        
        Args:
            call_data: Call option data
            put_data: Put option data
        
        Returns:
            Spread analysis metrics
        """
        call_ltp = float(call_data.get('ltp', 0))
        put_ltp = float(put_data.get('ltp', 0))

        return {
            'call_ltp': call_ltp,
            'put_ltp': put_ltp,
            'spread': abs(call_ltp - put_ltp),
            'spread_percentage': (abs(call_ltp - put_ltp) / ((call_ltp + put_ltp) / 2)) * 100 if call_ltp and put_ltp else None
        }
        
    def _calculate_moneyness(self, strike: float, current_price: float) -> str:
        """Calculate option moneyness"""
        diff_percent = ((strike - current_price) / current_price) * 100
        
        if abs(diff_percent) <= 0.5:
            return 'ATM'
        elif diff_percent > 0:
            return 'OTM' if diff_percent <= 2 else 'FAR_OTM'
        else:
            return 'ITM' if diff_percent >= -2 else 'FAR_ITM'

    # Add these methods to MarketAnalysisService class:

    def _find_directional_trades(self,
                               trend: str,
                               options_analysis: Dict[str, Any],
                               market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Find directional trading opportunities"""
        try:
            optimal_expiry = options_analysis.get('optimal_expiry', {})
            expiry = optimal_expiry.get('selected_expiry')
            
            if not expiry:
                return {}
                
            expiry_data = options_analysis.get('expiry_analysis', {}).get(expiry, {})
            strikes = expiry_data.get('strikes_analysis', {})
            
            opportunities = {
                'bullish': self._find_bullish_trades(strikes, trend),
                'bearish': self._find_bearish_trades(strikes, trend)
            }
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Directional trades identification error: {e}")
            return {}

    def _find_bullish_trades(self,
                           strikes: Dict[str, Any],
                           trend: str) -> List[Dict[str, Any]]:
        """Find bullish trading opportunities"""
        opportunities = []
        
        for strike, analysis in strikes.items():
            call_data = analysis.get('call', {})
            greeks = call_data.get('greeks', {})
            
            if self._is_suitable_bullish_trade(greeks, analysis.get('moneyness')):
                opportunities.append({
                    'strategy': 'LONG_CALL',
                    'strike': strike,
                    'premium': call_data.get('ltp', 0),
                    'greeks': greeks,
                    'confidence': self._calculate_trade_confidence(
                        trend, greeks, analysis.get('moneyness')
                    )
                })
        
        return sorted(opportunities, key=lambda x: x['confidence'], reverse=True)

    def _find_bearish_trades(self,
                           strikes: Dict[str, Any],
                           trend: str) -> List[Dict[str, Any]]:
        """Find bearish trading opportunities"""
        opportunities = []
        
        for strike, analysis in strikes.items():
            put_data = analysis.get('put', {})
            greeks = put_data.get('greeks', {})
            
            if self._is_suitable_bearish_trade(greeks, analysis.get('moneyness')):
                opportunities.append({
                    'strategy': 'LONG_PUT',
                    'strike': strike,
                    'premium': put_data.get('ltp', 0),
                    'greeks': greeks,
                    'confidence': self._calculate_trade_confidence(
                        trend, greeks, analysis.get('moneyness')
                    )
                })
        
        return sorted(opportunities, key=lambda x: x['confidence'], reverse=True)

    def _calculate_trade_confidence(self,
                                 trend: str,
                                 greeks: Dict[str, Any],
                                 moneyness: str) -> float:
        """Calculate confidence score for a trade"""
        try:
            trend_alignment = 1.0 if trend in ['Strong Bullish', 'Moderately Bullish'] else 0.5
            greeks_score = self._calculate_greeks_score(greeks)
            moneyness_score = self._calculate_moneyness_score(moneyness)
            
            return (trend_alignment + greeks_score + moneyness_score) / 3
            
        except Exception:
            return 0.0

    def _calculate_greeks_score(self, greeks: Dict[str, Any]) -> float:
        """Calculate score based on Greeks values"""
        try:
            delta = abs(float(greeks.get('delta', 0)))
            gamma = float(greeks.get('gamma', 0))
            theta = abs(float(greeks.get('theta', 0)))
            
            delta_score = 1.0 if 0.3 <= delta <= 0.7 else 0.5
            gamma_score = 1.0 if gamma <= 0.05 else 0.5
            theta_score = 1.0 if theta <= 0.1 else 0.5
            
            return (delta_score + gamma_score + theta_score) / 3
            
        except Exception:
            return 0.0

    def _find_non_directional_trades(self,
                                   options_analysis: Dict[str, Any],
                                   market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Find non-directional trading opportunities"""
        try:
            optimal_expiry = options_analysis.get('optimal_expiry', {})
            expiry = optimal_expiry.get('selected_expiry')
            expiry_data = options_analysis.get('expiry_analysis', {}).get(expiry, {})
            
            return {
                'straddle': self._find_straddle_opportunities(expiry_data),
                'strangle': self._find_strangle_opportunities(expiry_data),
                'iron_condor': self._find_iron_condor_opportunities(expiry_data)
            }
            
        except Exception as e:
            logger.error(f"Non-directional trades identification error: {e}")
            return {}

    def _find_hedging_opportunities(self,
                                  trend: str,
                                  options_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Find hedging opportunities"""
        try:
            optimal_expiry = options_analysis.get('optimal_expiry', {})
            expiry = optimal_expiry.get('selected_expiry')
            expiry_data = options_analysis.get('expiry_analysis', {}).get(expiry, {})
            
            return {
                'protective_puts': self._find_protective_puts(expiry_data),
                'covered_calls': self._find_covered_calls(expiry_data, trend)
            }
            
        except Exception as e:
            logger.error(f"Hedging opportunities identification error: {e}")
            return {}

    def _get_optimal_strategies(self, 
                              trading_opportunities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimal trading strategies based on opportunities"""
        try:
            strategies = []
            
            # Add directional strategies
            directional = trading_opportunities.get('directional', {})
            for strategy in ['bullish', 'bearish']:
                opportunities = directional.get(strategy, [])
                if opportunities:
                    strategies.extend(
                        self._filter_best_opportunities(opportunities, 2)
                    )
            
            # Add non-directional strategies
            non_directional = trading_opportunities.get('non_directional', {})
            for strategy_type, opportunities in non_directional.items():
                if opportunities:
                    strategies.extend(
                        self._filter_best_opportunities(opportunities, 1)
                    )
            
            return sorted(strategies, key=lambda x: x.get('confidence', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Optimal strategies selection error: {e}")
            return []

    def _filter_best_opportunities(self,
                                 opportunities: List[Dict[str, Any]],
                                 limit: int) -> List[Dict[str, Any]]:
        """Filter best opportunities based on confidence"""
        return sorted(
            opportunities,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )[:limit]