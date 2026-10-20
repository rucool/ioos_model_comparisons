import sys
try:
    from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
