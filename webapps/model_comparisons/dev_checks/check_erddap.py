import time
from datetime import datetime, timedelta
from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time
from ioos_model_comparisons.regions import regions

t0 = time.time()
argo = get_argo_floats_by_time(regions['Mid Atlantic Bight']['extent'], datetime.now()-timedelta(days=14), datetime.now())
print(f"Argo took {time.time()-t0:.2f}s")
print(argo.head() if not argo.empty else "No Argo")

t0 = time.time()
gliders = get_active_gliders(regions['Mid Atlantic Bight']['extent'])
print(f"Gliders took {time.time()-t0:.2f}s")
print(gliders)
