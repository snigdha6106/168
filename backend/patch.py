import re
with open('main.py', 'r') as f:
    content = f.read()

import_patch = """import asyncio
import math

def clean_float(val):
    if val is None or math.isnan(val):
        return None
    return float(val)
"""
content = content.replace("import asyncio", import_patch)

payload_old = """                # 4. Construct payload
                payload = {
                    "timestamp": row['timestamp'],
                    "gnss_active": gnss_active,
                    "truth": {
                        "lat": row['true_lat'],
                        "lon": row['true_lon'],
                        "velocity": row['true_velocity']
                    },
                    "measured": {
                        "lat": row['gnss_lat'] if gnss_active else None,
                        "lon": row['gnss_lon'] if gnss_active else None,
                    },
                    "estimated": {
                        "lat": fusion_lat,
                        "lon": fusion_lon,
                        "velocity": pred_v,
                        "mode": "GNSS ACTIVE" if gnss_active else "DEAD RECKONING"
                    }
                }"""
                
payload_new = """                # 4. Construct payload
                payload = {
                    "timestamp": clean_float(row['timestamp']),
                    "gnss_active": gnss_active,
                    "truth": {
                        "lat": clean_float(row['true_lat']),
                        "lon": clean_float(row['true_lon']),
                        "velocity": clean_float(row['true_velocity'])
                    },
                    "measured": {
                        "lat": clean_float(row['gnss_lat']) if gnss_active else None,
                        "lon": clean_float(row['gnss_lon']) if gnss_active else None,
                    },
                    "estimated": {
                        "lat": clean_float(fusion_lat),
                        "lon": clean_float(fusion_lon),
                        "velocity": clean_float(pred_v),
                        "mode": "GNSS ACTIVE" if gnss_active else "DEAD RECKONING"
                    }
                }"""
content = content.replace(payload_old, payload_new)
# Since I changed the string earlier to GNSS+INS Fusion, let's just do a regex replace or just replace the specific dictionary.
