# demo_all_attacks.py

def demo_timing_attack(secret="hunter2xyz"):
    """Pure timing attack demo"""
    from timing_model import build_timing_model, collect_timing_data
    from timing_plot import plot_timing_pgm_structure, plot_timing_posteriors
    
    # Show structure
    plot_timing_pgm_structure()
    
    # Run attack
    timings = collect_timing_data(secret)
    model = build_timing_model()
    posteriors = run_timing_inference(model, timings)
    
    # Show results
    plot_timing_posteriors(posteriors)
    return posteriors

def demo_wordlike_attack(partial="veg", target="vegetable"):
    """Character prediction demo"""
    from wordlike_model import build_wordlike_model, run_wordlike_inference
    from wordlike_plot import plot_wordlike_pgm_structure, plot_wordlike_posteriors
    
    # Show structure  
    plot_wordlike_pgm_structure()
    
    # Run prediction
    model = build_wordlike_model()
    observed = {f"G{i+1}": CHARSET_INDEX[c] for i, c in enumerate(partial)}
    posteriors = run_wordlike_inference(model, observed)
    
    # Show results
    plot_wordlike_posteriors(posteriors, observed)
    return posteriors

def demo_hybrid_attack(secret="vegetable"):
    """Combined attack demo"""
    from hybrid_model import build_hybrid_model, run_hybrid_inference
    from hybrid_plot import plot_hybrid_pgm_structure, plot_hybrid_posteriors
    
    # Show structure
    plot_hybrid_pgm_structure()
    
    # Simulate partial knowledge + timing
    # ... implementation ...
    
    # Show superior results
    plot_hybrid_posteriors(posteriors)
    return posteriors