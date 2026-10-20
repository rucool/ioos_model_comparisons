import requests
import re

def get_latest_fvon_date(region):
    region_key = region.lower().replace(" ", "_")
    if region_key == "fiji":
        region_key = "south_pacific"
    url = f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/fvon/{region_key}/last_14_days/"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})T', r.text)
        if not dates:
            return None
        return sorted(set(dates), reverse=True)[0]
    except Exception as e:
        print("Error:", e)
        return None

print(get_latest_fvon_date("Bahamas"))
print(get_latest_fvon_date("Fiji"))
