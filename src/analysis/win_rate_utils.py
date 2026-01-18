"""
Utilities for win rate calculations, including Bayesian correction.
"""
from typing import List, Tuple

def calculate_bayesian_win_rate(wins: int, total: int, confidence: float = 10.0, global_mean: float = 50.0) -> float:
    """
    Calculate the Bayesian averaged win rate.
    
    Formula: (W * R + C * M) / (R + C)
    Where:
        W = Actual win rate (wins / total * 100)
        R = Number of matches (total)
        C = Confidence constant (matches required to trust trend)
        M = Global mean win rate (usually 50.0)
    
    Args:
        wins: Number of wins
        total: Total number of matches
        confidence: Number of matches for confidence (C)
        global_mean: Global mean win rate (M) in percent (0-100)
        
    Returns:
        float: Bayesian win rate percentage (0-100)
    """
    if total == 0:
        return global_mean
        
    actual_win_rate = (wins / total) * 100
    
    # Bayesian Average Formula
    # (actual_win_rate * total + global_mean * confidence) / (total + confidence)
    bayesian_rate = (actual_win_rate * total + global_mean * confidence) / (total + confidence)
    
    return bayesian_rate

def get_display_win_rate(wins: int, total: int, config_data: dict) -> float:
    """
    Get the win rate to display based on configuration.
    
    Args:
        wins: Number of wins
        total: Total matches
        config_data: Configuration dictionary containing 'analytics' settings
        
    Returns:
        float: Win rate percentage
    """
    analytics_config = config_data.get('analytics', {})
    mode = analytics_config.get('win_rate_mode', 'raw') # 'raw' or 'bayesian'
    
    if mode == 'bayesian' and total > 0:
        confidence = analytics_config.get('bayesian_confidence', 10.0)
        mean = analytics_config.get('bayesian_mean', 50.0)
        return calculate_bayesian_win_rate(wins, total, confidence, mean)
    
    # Default to raw
    if total == 0:
        if mode == 'bayesian':
            return analytics_config.get('bayesian_mean', 50.0)
        return 0.0
    return (wins / total) * 100


def calculate_win_probability(
    red_win_rates: List[float], 
    blue_win_rates: List[float],
    red_matches: List[int] = None,
    blue_matches: List[int] = None
) -> Tuple[float, float]:
    """
    Calculate the probability of the blue team winning and a confidence score.
    
    Args:
        red_win_rates: List of win rates for red team (0-100)
        blue_win_rates: List of win rates for blue team (0-100)
        red_matches: Optional list of match counts per player
        blue_matches: Optional list of match counts per player
        
    Returns:
        Tuple[float, float]: (Blue win probability 0-100, Confidence 0-100)
    """
    import math

    def to_logit(p):
        p = max(0.01, min(0.99, p / 100.0))
        return math.log(p / (1.0 - p))

    def from_logit(l):
        return 1.0 / (1.0 + math.exp(-l))

    # Calculate average logit for each team
    blue_strength = sum(to_logit(wr) for wr in blue_win_rates) / len(blue_win_rates)
    red_strength = sum(to_logit(wr) for wr in red_win_rates) / len(red_win_rates)

    prob = from_logit(blue_strength - red_strength) * 100
    
    # Calculate confidence
    # We consider 200 total matches across all 10 players as "High Confidence" (100%)
    total_matches = 0
    if red_matches: total_matches += sum(red_matches)
    if blue_matches: total_matches += sum(blue_matches)
    
    confidence = min(100.0, (total_matches / 200.0) * 100.0)
    
    return prob, confidence
