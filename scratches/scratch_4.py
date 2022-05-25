

# Make a big ol' plot to compare the wind products AT THE GLIDER LOCATION
fig = plt.figure(figsize=(15,10), constrained_layout=True)
xticks=(pd.date_range(start_date, end_date-timedelta(hours=23), freq='2D'))

# Break it up into sections
gs = fig.add_gridspec(3,1)

# Get colors for lines
N=3
colors = cmo.phase(np.linspace(0.6,0.95,N))

################################################################################
# Plot wind speed
ax = fig.add_subplot(gs[0,:])



# ax.plot(buoydf['time (UTC)'],buoydf['wspd (m s-1)'], c=colors[0], label='42040: 4.1m', lw=3)
ax.plot(hrrr_time,hrrr_windSpeed, c=colors[1], label='HRRR: 10m')
ax.plot(nam_time,nam_windSpeed, c=colors[2], label='NAM: 10m')

ax.legend()
ax.set_title('Wind speed',fontweight='bold')

# Set the xticks and labels
ax.xaxis.set_major_formatter(myFmt)
ax.set_xticks(xticks)
ax.tick_params(labelbottom = False, bottom = False) # Remove bottom tick marks and labels
ax.grid(True)

################################################################################

################################################################################
# Plot U wind
ax = fig.add_subplot(gs[1,:])

# ax.plot(buoydf['time (UTC)'],buoydf['wspu (m s-1)'], c=colors[0], label='42040: 4.1m', lw=3)
ax.plot(hrrr_time,hrrr_EWwind, c=colors[1], label='HRRR: 10m')
ax.plot(nam_time,nam_EWwind, c=colors[2], label='NAM: 10m')

ax.legend()
ax.set_title('E-W Wind speed',fontweight='bold')

# Set the xticks and labels
ax.xaxis.set_major_formatter(myFmt)
ax.set_xticks(xticks)
ax.tick_params(labelbottom = False, bottom = False) # Remove bottom tick marks and labels
ax.grid(True)
################################################################################

################################################################################
# Plot V wind
ax = fig.add_subplot(gs[2,:])

# ax.plot(buoydf['time (UTC)'],buoydf['wspv (m s-1)'], c=colors[0], label='42040: 4.1m', lw=3)
ax.plot(hrrr_time,hrrr_NWwind, c=colors[1], label='HRRR: 10m')
ax.plot(nam_time,nam_NWwind, c=colors[2], label='NAM: 10m')

ax.legend()
ax.set_title('N-S Wind speed',fontweight='bold')

# Set the xticks and labels
ax.xaxis.set_major_formatter(myFmt)
ax.set_xticks(xticks)
ax.grid(True)
################################################################################
plt.suptitle('Winds at ng645 location', fontweight='bold')