import json
from collections import defaultdict

# --- Paths ---
simplified_json_path = "./benchmark/traces/simplified_trace.json"

# --- Load simplified JSON ---
with open(simplified_json_path, "r") as f:
    data = json.load(f)

# --- Aggregate durations by function name ---
durations_by_name = defaultdict(float)

for ev in data.get("traceEvents", []):
    name = ev.get("name", "unknown")
    dur = ev.get("dur", 0)
    durations_by_name[name] += dur

# --- Sort by total duration descending ---
sorted_durations = sorted(durations_by_name.items(), key=lambda x: x[1], reverse=True)

# --- Display top functions ---
print("Top time-consuming functions (µs):")
for name, total_dur in sorted_durations[:200]:  # top 200
    print(f"{name:50s} {total_dur:12.2f}")

# --- Optional: total time ---
total_time = sum(durations_by_name.values())
print(f"\nTotal traced time: {total_time:.2f} µs")
