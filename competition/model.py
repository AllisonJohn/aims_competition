"""Baseline submission: Benchmark-level predictions with item difficulty adjustment."""

# Binary benchmark pass rates (from your statistics)
BINARY_BENCHMARK_RATES = {
    'afrimedqa': 0.6417,
    'agentdojo': 0.3612,
    'ai2d_test': 0.7293,
    'androidworld': 0.8764,
    'bfcl': 0.6071,
    'cybench': 0.0781,
    'hle': 0.3149,
    'livecodebench': 0.4830,
    'matharena': 0.6399,
    'mathvista_mini': 0.2604,
    'mmbench_v11': 0.8065,
    'mmlupro': 0.4990,
    'rewardbench': 0.7058,
    'swebench': 0.5147,
}

# Continuous benchmark normalized scores
CONTINUOUS_BENCHMARK_RATES = {
    'mtbench': 0.5748,      # 5.748 / 10
    'ultrafeedback': 0.7444, # 3.722 / 5
}

# Combined lookup
BENCHMARK_RATES = {**BINARY_BENCHMARK_RATES, **CONTINUOUS_BENCHMARK_RATES}

# Default rate for unknown benchmarks
DEFAULT_RATE = 0.55


def estimate_item_difficulty(item_text: str) -> float:
    """
    Estimate relative difficulty adjustment based on item text.
    Returns multiplier: 1.0 = average, <1.0 = harder, >1.0 = easier
    """
    
    if not item_text:
        return 1.0
    
    difficulty_multiplier = 1.0
    
    # Length-based heuristics
    text_length = len(item_text)
    if text_length > 1000:
        difficulty_multiplier *= 0.90  # Very long questions are harder
    elif text_length > 500:
        difficulty_multiplier *= 0.95
    elif text_length < 100:
        difficulty_multiplier *= 1.05  # Very short questions are easier
    
    # Content-based heuristics
    text_lower = item_text.lower()
    
    # Mathematical content
    if any(term in item_text for term in ['∫', '∑', '∂', '√', '≤', '≥', '∈']):
        difficulty_multiplier *= 0.92  # Math symbols = harder
    
    # Reasoning indicators
    reasoning_words = ['prove', 'derive', 'explain why', 'justify', 'demonstrate']
    if any(word in text_lower for word in reasoning_words):
        difficulty_multiplier *= 0.93  # Reasoning = harder
    
    # Code content
    if any(keyword in text_lower for keyword in ['def ', 'function', 'class ', 'import ']):
        difficulty_multiplier *= 0.94  # Code = harder
    
    # Multi-part questions
    if item_text.count('\n') > 10 or text_lower.count('part ') > 1:
        difficulty_multiplier *= 0.92  # Multi-part = harder
    
    # Simple factual questions (likely easier)
    simple_patterns = ['what is', 'who is', 'when did', 'where is']
    if any(pattern in text_lower for pattern in simple_patterns):
        if text_length < 200:  # Only if also short
            difficulty_multiplier *= 1.08
    
    return difficulty_multiplier


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """
    Predict probability that subject answers item correctly.
    
    Strategy:
    1. Use benchmark-level base rate (known from training)
    2. Adjust based on estimated item difficulty (inferred from text)
    3. Optionally calibrate using revealed labels
    
    Args:
        input: Dict with keys 'benchmark', 'condition', 'subject_content', 'item_content'
        labeled: Optional list of revealed labeled examples
    
    Returns:
        Float in [0, 1] representing predicted probability
    """
    
    # Get benchmark
    benchmark = input.get('benchmark', '')
    
    # Look up base rate for this benchmark
    base_rate = BENCHMARK_RATES.get(benchmark, DEFAULT_RATE)
    
    # Adjust for estimated item difficulty
    item_text = input.get('item_content', '')
    difficulty_adjustment = estimate_item_difficulty(item_text)
    
    prediction = base_rate * difficulty_adjustment
    
    # Optional: Use labeled data for calibration
    if labeled and len(labeled) > 0:
        # Simple calibration: blend with observed rate from labeled examples
        labeled_rate = sum(ex.get('label', 0) for ex in labeled) / len(labeled)
        
        # Weighted blend: trust our prediction more, but adjust based on labels
        prediction = 0.8 * prediction + 0.2 * labeled_rate
    
    # Clamp to valid probability range (avoid 0.0 and 1.0 exactly)
    return max(0.01, min(0.99, prediction))