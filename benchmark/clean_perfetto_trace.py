import json

# --- Paths to modify ---
input_json_path = "./benchmark/traces/perfetto_trace.json"   # JSON exported from Perfetto
output_json_path = "./benchmark/traces/simplified_trace.json"

# --- Load the full JSON ---
with open(input_json_path, "r") as f:
    data = json.load(f)

simplified_events = []

# --- Filter only "X" events (complete events with duration) ---
for ev in data.get("traceEvents", []):
    if ev.get("ph") == "X":  # "X" = Complete event with duration
        simplified_events.append({
            "name": ev.get("name"),
            "ts": ev.get("ts"),
            "dur": ev.get("dur"),
            "pid": ev.get("pid"),
            "tid": ev.get("tid")
        })

# --- Create simplified JSON ---
simplified_json = {
    "traceEvents": simplified_events
}

# --- Save simplified JSON ---
with open(output_json_path, "w") as f:
    json.dump(simplified_json, f, indent=2)

print(f"Saved {len(simplified_events)} events to {output_json_path}")
