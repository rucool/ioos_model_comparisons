import pandas as pd
from pathlib import Path

root_dir = Path.home() / "Documents"
data_dir = root_dir / "data"
impact_calc = data_dir / "impact_metrics" / "calculated"

gdata = "ng645-20210613T0000_calculated_glider_data.pkl"
rdata = "ng645-20210613T0000_rtofs_0day_offset_data_computed.pkl"
sname = "ng645-20210613T0000_0day_offset_combined.pkl"
# rdata = "ng645-20210613T0000_rtofs_1day_offset_data_computed.pkl"
# sname = "ng645-20210613T0000_1day_offset_combined.pkl"
# rdata = "ng645-20210613T0000_rtofs_2day_offset_data_computed.pkl"
# sname = "ng645-20210613T0000_2day_offset_combined.pkl"
# rdata = "ng645-20210613T0000_rtofs_3day_offset_data_computed.pkl"
# sname = "ng645-20210613T0000_3day_offset_combined.pkl"

glider = pd.read_pickle(impact_calc / gdata)
rtofs = pd.read_pickle(impact_calc / rdata)

glider["source"] = "ng645"
rtofs["source"] = "rtofs"

df = pd.concat([glider, rtofs])
df.to_pickle(impact_calc / "merged" / sname)