"""
Utilities for win rate calculations, including Bayesian correction.
"""

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
        return 0.0
    return (wins / total) * 100
