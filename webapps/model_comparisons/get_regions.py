import re
import json

with open("/Users/mikesmith/Documents/github/ioos_model_comparisons/ioos_model_comparisons/regions.py", "r") as f:
    content = f.read()

# split by `key = `
blocks = content.split("key = ")

region_info = {}

for block in blocks[1:]:
    # get the name
    name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', block)
    if not name_match:
        continue
    name = name_match.group(1)
    
    variables = []
    depths = set()
    
    # check for sea_water_temperature
    if 'sea_water_temperature = [' in block:
        variables.append('temperature')
        depth_matches = re.findall(r'dict\(depth=(\d+)', block.split('sea_water_temperature = [')[1].split(']')[0])
        for d in depth_matches: depths.add(f"{d}m")
        
    # check for salinity
    if 'salinity = [' in block:
        variables.append('salinity')
        depth_matches = re.findall(r'dict\(depth=(\d+)', block.split('salinity = [')[1].split(']')[0])
        for d in depth_matches: depths.add(f"{d}m")
        
    if 'sea_surface_height =' in block:
        variables.append('ssh')
        
    if 'ocean_heat_content =' in block:
        variables.append('ocean_heat_content')
        
    if 'currents =' in block:
        # check if bool=True
        curr_block = block.split('currents =')[1]
        if 'bool=True' in curr_block or 'bool = True' in curr_block:
            variables.append('currents')
            # find depths = [0, 150]
            depth_match = re.search(r'depths\s*=\s*\[([\d,\s]+)\]', curr_block)
            if depth_match:
                ds = depth_match.group(1).split(',')
                for d in ds:
                    d = d.strip()
                    if d: depths.add(f"{d}m")
            elif 'depths' not in curr_block.split('figure')[0]:
                pass # defaults
                
    depths_sorted = sorted(list(depths), key=lambda x: int(x[:-1]))
    
    region_info[name] = {
        "variables": variables,
        "depths": depths_sorted
    }

print(json.dumps(region_info, indent=4))
