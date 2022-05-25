import pandas as pd
from hurricanes.src import active_argo_floats
import datetime as dt

extent = [-99.040131, -37.089444, 7.514353, 35.693125]

t0 = dt.datetime(2021, 8, 1)
t1 = dt.datetime(2021, 10, 1)

floats = active_argo_floats(bbox=extent, time_start=t0, time_end=t1)
# floats.to_csv('/Users/mikesmith/Documents/argo-erddap-hurricane-season2021.csv', index=False)
# floats = pd.read_csv('/Users/mikesmith/Documents/argo-erddap-hurricane-season2021.csv')

# df = pd.read_csv('/Users/mikesmith/Documents/whoi_argo.csv', usecols=['wmo', 'lon', 'lat'])
df = floats
df['sampling_times'] = None

grouped_floats = floats.groupby('platform_number')

temp_df = pd.DataFrame()

for group in grouped_floats:
    if group[0] in df.platform_number.unique():
        float_df = pd.DataFrame()
        time_grouped = group[1].groupby('time (UTC)')
        print(f'{group[0]} - {len(time_grouped)} unique times')
        df.loc[df['platform_number'] == group[0], 'sampling_times'] = len(time_grouped)
        for tgroup in time_grouped:
            temp = dict(
                wmo=int(tgroup[1]['platform_number'].min()),
                time=pd.Timestamp(tgroup[0]),
                lon=tgroup[1]['longitude (degrees_east)'].min(),
                lat=tgroup[1]['latitude (degrees_north)'].min(),
            )
            float_df = float_df.append(temp, ignore_index=True)
        float_df['surfacing_interval'] = float_df['time'].diff()
        temp_df = temp_df.append(float_df, ignore_index=True)

temp_df['wmo'] = temp_df['wmo'].astype(int)
tdf = temp_df[temp_df.surfacing_interval < '4 days']
# tdf.to_csv('/Users/mikesmith/Documents/who_argo.csv', index=False)
tdf.to_csv('/Users/mikesmith/Documents/whoi_argo_complete_profiles.csv', index=False)
