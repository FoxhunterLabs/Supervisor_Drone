# ================================================================
# Supervisor Drone HUD v2 — Heavy Equipment Site Oversight
# Cinematic construction-site style UI, human-gated proposals only
#
# Run with:
# streamlit run app.py
#
# All positions, zones, and signals are SYNTHETIC.
# No real-world control or actuation is performed.
# ================================================================

import math
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# ------------------------------------------------
# PAGE CONFIG

# ------------------------------------------------
st.set_page_config(
page_title="Supervisor Drone HUD — Heavy Equipment Safety",
layout="wide",
)

# ------------------------------------------------
# SITE / SIM CONFIG
# ------------------------------------------------
SITE_WIDTH = 100.0 # meters (X)
SITE_HEIGHT = 80.0 # meters (Y)
NUM_WORKERS = 8
NUM_MACHINES = 4
MAX_HISTORY = 300

# Background image (synthetic aerial view).
# In this environment we load from the uploaded file path.
# In a real repo, replace with your own asset path.
BACKGROUND_IMG_PATH = "/mnt/data/A0A0FBF0-9206-41B1-B530-C500C5C891BB.png"

# ================================================================
# GEOMETRY / ZONES
# ================================================================
def get_zones() -> List[Dict[str, Any]]:
"""

Polygon zones across a synthetic quarry / construction site.
Roughly mirrors common patterns: haul road, staging, no-go, etc.
"""
return [
{
"name": "Blast / Cutting Zone",
"type": "NO_GO",
"severity": 1.0,
"poly": [(20, 15), (55, 15), (55, 32), (20, 32)],
},
{
"name": "Haul Road",
"type": "EQUIP_ONLY",
"severity": 0.8,
"poly": [(0, 30), (100, 30), (100, 48), (0, 48)],
},
{
"name": "Staging & Laydown",
"type": "HIGH_TRAFFIC",
"severity": 0.6,
"poly": [(60, 8), (95, 8), (95, 28), (60, 28)],
},
{
"name": "Spotter Required Zone",
"type": "SPOTTER_REQUIRED",
"severity": 0.7,

"poly": [(18, 52), (58, 52), (58, 76), (18, 76)],
},
{
"name": "Pedestrian Corridor",
"type": "PEDESTRIAN_ONLY",
"severity": 0.6,
"poly": [(70, 32), (92, 32), (92, 76), (70, 76)],
},
]

def point_in_poly(x: float, y: float, poly: List[tuple]) -> bool:
"""Ray-casting point-in-polygon test."""
inside = False
n = len(poly)
px, py = x, y
for i in range(n):
x1, y1 = poly[i]
x2, y2 = poly[(i + 1) % n]
if ((y1 > py) != (y2 > py)) and (
px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
):
inside = not inside
return inside

def clamp(v: float, lo: float, hi: float) -> float:
return max(lo, min(hi, v))

# ================================================================
# AUDIT / EVENTS
# ================================================================
def sha256_entry(data: Dict[str, Any]) -> str:
s = json.dumps(data, sort_keys=True)
return hashlib.sha256(s.encode("utf-8")).hexdigest()

def log_audit_event(kind: str, payload: Dict[str, Any]) -> None:
prev_hash = (
st.session_state.audit_log[-1]["hash"]
if st.session_state.audit_log
else "0" * 64
)
entry = {
"timestamp": datetime.utcnow().isoformat() + "Z",
"tick": st.session_state.tick,
"kind": kind,
"payload": payload,
"prev_hash": prev_hash,
}
entry["hash"] = sha256_entry(entry)

st.session_state.audit_log.append(entry)
st.session_state.audit_log = st.session_state.audit_log[-200:]

def log_event(level: str, msg: str, extras: Optional[Dict[str, Any]] = None) -> None:
payload: Dict[str, Any] = {
"timestamp": datetime.utcnow().strftime("%H:%M:%S"),
"level": level,
"msg": msg,
}
if extras:
payload.update(extras)
st.session_state.events.append(payload)
st.session_state.events = st.session_state.events[-200:]

# ================================================================
# STATE INIT
# ================================================================
def init_state() -> None:
st.session_state.tick = 0
st.session_state.history: List[Dict[str, Any]] = []
st.session_state.audit_log: List[Dict[str, Any]] = []
st.session_state.events: List[Dict[str, Any]] = []
st.session_state.proposals: List[Dict[str, Any]] = []
st.session_state.human_gate_open = False

st.session_state.zones = get_zones()

# Drone starts mid-site, medium altitude / zoom
st.session_state.drone = {
"x": SITE_WIDTH / 2,
"y": SITE_HEIGHT / 2,
"alt": 35.0, # meters
"zoom": 0.5, # 0–1 (1 = tight)
}

# Workers & machines seeded randomly
rng = np.random.default_rng(42)
st.session_state.workers = [
{
"id": f"W{i+1}",
"x": float(rng.uniform(5, SITE_WIDTH - 5)),
"y": float(rng.uniform(5, SITE_HEIGHT - 5)),
}
for i in range(NUM_WORKERS)
]
st.session_state.machines = [
{
"id": f"M{i+1}",
"x": float(rng.uniform(10, SITE_WIDTH - 10)),
"y": float(rng.uniform(10, SITE_HEIGHT - 10)),
}

for i in range(NUM_MACHINES)
]
log_audit_event("INIT", {"msg": "Simulation initialized"})

if "tick" not in st.session_state:
init_state()

# ================================================================
# CLARITY / RISK ENGINE
# ================================================================
def compute_fov_bounds(drone: Dict[str, Any]) -> Dict[str, float]:
"""
Approximate camera footprint as axis-aligned rect.
Larger with altitude, smaller with zoom.
"""
alt = drone["alt"]
zoom = drone["zoom"]

base_half_w = 15.0
base_half_h = 12.0
scale_alt = alt / 30.0
scale_zoom = 1.0 - 0.5 * zoom # zoom=1 shrinks footprint

half_w = base_half_w * scale_alt * scale_zoom

half_h = base_half_h * scale_alt * scale_zoom

half_w = clamp(half_w, 8.0, 60.0)
half_h = clamp(half_h, 6.0, 50.0)

cx, cy = drone["x"], drone["y"]
return {
"x_min": clamp(cx - half_w, 0.0, SITE_WIDTH),
"x_max": clamp(cx + half_w, 0.0, SITE_WIDTH),
"y_min": clamp(cy - half_h, 0.0, SITE_HEIGHT),
"y_max": clamp(cy + half_h, 0.0, SITE_HEIGHT),
}

def compute_zone_violations(
workers: List[Dict[str, Any]],
machines: List[Dict[str, Any]],
zones: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
"""
Worker/machine in wrong zones → violations with severity.
"""
violations: List[Dict[str, Any]] = []

# Workers
for w in workers:

for z in zones:
if point_in_poly(w["x"], w["y"], z["poly"]):
if z["type"] in ("NO_GO", "EQUIP_ONLY", "SPOTTER_REQUIRED"):
violations.append(
{
"actor_id": w["id"],
"actor_type": "WORKER",
"x": w["x"],
"y": w["y"],
"zone": z["name"],
"zone_type": z["type"],
"severity": z["severity"],
}
)

# Machines
for m in machines:
for z in zones:
if point_in_poly(m["x"], m["y"], z["poly"]):
if z["type"] in ("NO_GO", "PEDESTRIAN_ONLY"):
violations.append(
{
"actor_id": m["id"],
"actor_type": "MACHINE",
"x": m["x"],
"y": m["y"],

"zone": z["name"],
"zone_type": z["type"],
"severity": z["severity"],
}
)

return violations

def compute_clarity(
drone: Dict[str, Any],
workers: List[Dict[str, Any]],
machines: List[Dict[str, Any]],
zones: List[Dict[str, Any]],
) -> Dict[str, Any]:
"""
Clarity (0–100) and risk (0–100) based on:
- FOV coverage
- Zone violations
- Worker-machine proximity
- Altitude / zoom sanity
"""
fov = compute_fov_bounds(drone)
site_area = SITE_WIDTH * SITE_HEIGHT
fov_area = (fov["x_max"] - fov["x_min"]) * (fov["y_max"] - fov["y_min"])
coverage_frac = clamp(fov_area / site_area, 0.05, 1.0)

violations = compute_zone_violations(workers, machines, zones)
total_violation_severity = sum(v["severity"] for v in violations)
violation_factor = clamp(total_violation_severity / 5.0, 0.0, 1.5)

# min worker/machine separation
min_dist = 999.0
for w in workers:
for m in machines:
dx = w["x"] - m["x"]
dy = w["y"] - m["y"]
d = math.hypot(dx, dy)
if d < min_dist:
min_dist = d
prox_factor = clamp((20.0 - min_dist) / 20.0, 0.0, 1.0)

alt = drone["alt"]
zoom = drone["zoom"]

alt_factor = 0.0
if alt < 18.0:
alt_factor = (18.0 - alt) / 18.0

zoom_factor = 0.0
if zoom > 0.7:
zoom_factor = (zoom - 0.7) / 0.3

# baseline clarity from coverage
base_clarity = 0.4 + 0.5 * coverage_frac # 0.4–0.9

penalty = (
0.4 * violation_factor
+ 0.3 * prox_factor
+ 0.2 * alt_factor
+ 0.1 * zoom_factor
)

clarity = clamp(base_clarity - penalty, 0.1, 0.98)
clarity_pct = round(clarity * 100, 1)

risk = clamp((1.0 - clarity) * 100 + 20 * violation_factor + 15 * prox_factor, 0, 100)
risk = round(risk, 1)

if clarity_pct >= 90 and risk < 25:
state = "STABLE"
elif clarity_pct >= 80:
state = "TENSE"
elif clarity_pct >= 65:
state = "HIGH_RISK"
else:
state = "CRITICAL"

predicted_risk = clamp(
risk + 0.5 * (violation_factor * 40 + prox_factor * 30 + zoom_factor * 20),
0,
100,
)
predicted_risk = round(predicted_risk, 1)

return {
"clarity": clarity_pct,
"risk": risk,
"state": state,
"predicted_risk": predicted_risk,
"coverage_frac": round(coverage_frac, 3),
"min_worker_machine_dist": round(min_dist, 2),
"violations": violations,
"fov": fov,
}

def compute_snapshot() -> Dict[str, Any]:
drone = st.session_state.drone
workers = st.session_state.workers
machines = st.session_state.machines
zones = st.session_state.zones

clarity_pack = compute_clarity(drone, workers, machines, zones)

snapshot = {
"tick": st.session_state.tick,
"timestamp": datetime.utcnow().isoformat() + "Z",
"drone": drone.copy(),
"clarity": clarity_pack["clarity"],
"risk": clarity_pack["risk"],
"state": clarity_pack["state"],
"predicted_risk": clarity_pack["predicted_risk"],
"coverage_frac": clarity_pack["coverage_frac"],
"min_worker_machine_dist": clarity_pack["min_worker_machine_dist"],
"violations": clarity_pack["violations"],
"fov": clarity_pack["fov"],
}

log_audit_event(
"SNAPSHOT",
{
"state": snapshot["state"],
"clarity": snapshot["clarity"],
"risk": snapshot["risk"],
"violations": len(snapshot["violations"]),
},
)
return snapshot

# ================================================================
# PROPOSALS (HUMAN-GATED)
# ================================================================
def maybe_generate_proposal(snapshot: Dict[str, Any]) -> None:
"""
Drone never auto-acts.
This only queues small, bounded proposals for human review.
"""
if len(st.session_state.proposals) > 6:
return

clarity = snapshot["clarity"]
risk = snapshot["risk"]
predicted_risk = snapshot["predicted_risk"]
violations = snapshot["violations"]
drone = snapshot["drone"]

reasons: List[str] = []
if violations:
reasons.append("zone_violations")
if clarity < 80:
reasons.append("low_clarity")
if predicted_risk > 70:
reasons.append("rising_risk")

if not reasons:
return

pid = len(st.session_state.proposals) + 1

if "zone_violations" in reasons:
title = "Recenter drone over conflict cluster"
action = "recenter_on_violations"
elif "low_clarity" in reasons and drone["alt"] < 40:
title = "Increase altitude for wider FOV"
action = "increase_altitude"
elif "low_clarity" in reasons and drone["zoom"] > 0.5:
title = "Reduce zoom to widen coverage"
action = "zoom_out"
else:
title = "Hold position and monitor"
action = "hold_position"

rationale_bits: List[str] = []
if "zone_violations" in reasons:
rationale_bits.append(f"{len(violations)} zone violations detected")
if "low_clarity" in reasons:
rationale_bits.append(f"clarity {clarity:.1f}% below 80% target")
if "rising_risk" in reasons:
rationale_bits.append(f"predicted risk {predicted_risk:.1f}% above safe band")

proposal = {
"id": pid,
"created": datetime.utcnow().strftime("%H:%M:%S"),
"title": title,
"rationale": "; ".join(rationale_bits),
"status": "PENDING",
"action": action,
"snapshot": {
"tick": snapshot["tick"],
"clarity": snapshot["clarity"],
"risk": snapshot["risk"],
"predicted_risk": snapshot["predicted_risk"],
},
"violations_cache": [
{"x": v["x"], "y": v["y"], "severity": v["severity"]}
for v in violations
],
}

st.session_state.proposals.append(proposal)
log_audit_event("PROPOSAL_CREATED", {"id": pid, "title": title})

def apply_proposal_action(proposal: Dict[str, Any]) -> None:
"""
Apply a small, bounded trim to drone state.

Only called AFTER approval and gate open.
"""
drone = st.session_state.drone

if proposal["action"] == "increase_altitude":
drone["alt"] = clamp(drone["alt"] + 5.0, 10.0, 60.0)
elif proposal["action"] == "zoom_out":
drone["zoom"] = clamp(drone["zoom"] - 0.15, 0.1, 1.0)
elif proposal["action"] == "recenter_on_violations":
viols = proposal.get("violations_cache") or []
if viols:
xs = [v["x"] for v in viols]
ys = [v["y"] for v in viols]
cx = sum(xs) / len(xs)
cy = sum(ys) / len(ys)
drone["x"] = clamp(0.7 * drone["x"] + 0.3 * cx, 0.0, SITE_WIDTH)
drone["y"] = clamp(0.7 * drone["y"] + 0.3 * cy, 0.0, SITE_HEIGHT)
elif proposal["action"] == "hold_position":
pass

log_audit_event(
"PROPOSAL_APPLIED", {"id": proposal["id"], "action": proposal["action"]}
)

# ================================================================

# SIMULATION STEP
# ================================================================
def simulate_step() -> None:
"""Move workers/machines slightly and recompute snapshot."""
st.session_state.tick += 1
rng = np.random.default_rng(int(time.time()) % 2**32)

# workers wander more
for w in st.session_state.workers:
w["x"] = clamp(w["x"] + float(rng.normal(0, 1.2)), 0.0, SITE_WIDTH)
w["y"] = clamp(w["y"] + float(rng.normal(0, 1.2)), 0.0, SITE_HEIGHT)

# machines move slower / smoother
for m in st.session_state.machines:
m["x"] = clamp(m["x"] + float(rng.normal(0, 0.6)), 0.0, SITE_WIDTH)
m["y"] = clamp(m["y"] + float(rng.normal(0, 0.6)), 0.0, SITE_HEIGHT)

# drone drifts slightly
st.session_state.drone["x"] = clamp(
st.session_state.drone["x"] + float(rng.normal(0, 0.3)), 0.0, SITE_WIDTH
)
st.session_state.drone["y"] = clamp(
st.session_state.drone["y"] + float(rng.normal(0, 0.3)), 0.0, SITE_HEIGHT
)

snapshot = compute_snapshot()

st.session_state.history.append(snapshot)
st.session_state.history = st.session_state.history[-MAX_HISTORY:]
maybe_generate_proposal(snapshot)

# ================================================================
# VISUALS
# ================================================================
def draw_site_map(snapshot: Dict[str, Any]) -> None:
"""Top-down map of zones, workers, machines, and drone FOV."""
zones = st.session_state.zones
workers = st.session_state.workers
machines = st.session_state.machines
drone = st.session_state.drone
fov = snapshot["fov"]

fig, ax = plt.subplots(figsize=(7, 5))

# Background: aerial photo if available, else procedural dirt texture.
try:
bg = Image.open(BACKGROUND_IMG_PATH).convert("RGB")
ax.imshow(bg, extent=[0, SITE_WIDTH, 0, SITE_HEIGHT])
except Exception:
tex = np.random.default_rng(123).normal(0.55, 0.09, (400, 600))
tex = np.clip(tex, 0.25, 0.85)
ax.imshow(tex, extent=[0, SITE_WIDTH, 0, SITE_HEIGHT], cmap="gray")

# Slight vignette
vx = np.linspace(0.4, 1.0, 200)
vy = np.linspace(0.4, 1.0, 200)
vignette = np.outer(vx, vy)
ax.imshow(
vignette,
extent=[0, SITE_WIDTH, 0, SITE_HEIGHT],
cmap="gray",
alpha=0.35,
origin="lower",
)

# Zone styles
zone_styles = {
"NO_GO": {
"fc": (1, 0, 0, 0.35),
"ec": "#ff5555",
"lw": 2.0,
},
"EQUIP_ONLY": {
"fc": (1, 0.9, 0, 0.25),
"ec": "#ffcc33",
"lw": 2.0,
},
"HIGH_TRAFFIC": {

"fc": (0, 1, 1, 0.25),
"ec": "#00d0ff",
"lw": 1.8,
},
"SPOTTER_REQUIRED": {
"fc": (1, 0.4, 0, 0.28),
"ec": "#ff8844",
"lw": 1.8,
},
"PEDESTRIAN_ONLY": {
"fc": (0, 1, 0, 0.25),
"ec": "#44ff44",
"lw": 1.8,
},
}

# Draw zones
for z in zones:
xs = [p[0] for p in z["poly"]] + [z["poly"][0][0]]
ys = [p[1] for p in z["poly"]] + [z["poly"][0][1]]

style = zone_styles.get(
z["type"], {"fc": (1, 1, 1, 0.1), "ec": "white", "lw": 1.0}
)

ax.fill(xs, ys, facecolor=style["fc"], edgecolor=style["ec"], linewidth=style["lw"])

ax.text(
sum(xs[:-1]) / (len(xs) - 1),
sum(ys[:-1]) / (len(ys) - 1),
z["name"],
color="white",
fontsize=7,
ha="center",
va="center",
bbox=dict(facecolor=(0, 0, 0, 0.45), edgecolor="none", pad=1.5),
)

# FOV rectangle
ax.add_patch(
plt.Rectangle(
(fov["x_min"], fov["y_min"]),
fov["x_max"] - fov["x_min"],
fov["y_max"] - fov["y_min"],
linewidth=1.8,
edgecolor="white",
facecolor=(1, 1, 1, 0.05),
linestyle="--",
)
)

# Drone icon

ax.scatter([drone["x"]], [drone["y"]], c="white", s=50, marker="^")
ax.text(
drone["x"],
drone["y"] + 2.0,
"SUPERVISOR UAV",
color="white",
fontsize=7,
ha="center",
bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=1),
)

# Workers
for w in workers:
ax.scatter([w["x"]], [w["y"]], c="#00ff99", s=22)
ax.text(
w["x"],
w["y"] + 1.4,
f"WORKER {w['id'][1:]}",
color="white",
fontsize=6.5,
ha="center",
bbox=dict(facecolor=(0, 0, 0, 0.7), edgecolor="none", pad=1),
)

# Machines
icon_map = {

"1": " ",
"2": " ",
"3": " ",
"4": " ",
}
for m in machines:
icon = icon_map.get(m["id"][1:], " ")
ax.text(m["x"], m["y"], icon, fontsize=16, ha="center", va="center")
ax.text(
m["x"],
m["y"] - 2.0,
f"{m['id']}",
color="white",
fontsize=7,
ha="center",
bbox=dict(facecolor=(0, 0, 0, 0.7), edgecolor="none", pad=1),
)

# Conflict callout (highest severity violation if any)
if snapshot["violations"]:
top_v = sorted(snapshot["violations"], key=lambda v: -v["severity"])[0]
msg = f"CONFLICT:\n{top_v['actor_type']} in {top_v['zone_type']} ZONE"
ax.text(
top_v["x"],
top_v["y"] + 4,

msg,
ha="center",
va="center",
fontsize=8,
color="white",
bbox=dict(facecolor=(0.7, 0, 0, 0.85), edgecolor="red", pad=4),
)

ax.set_xlim(0, SITE_WIDTH)
ax.set_ylim(0, SITE_HEIGHT)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
spine.set_color("#aaaaaa")
ax.set_title(
"SITE: D11 NORTHEAST CUT — 360° SUPERVISOR VIEW",
color="white",
fontsize=10,
pad=8,
)
fig.patch.set_facecolor("#101724")
ax.set_facecolor("none")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

def draw_360_camera(snapshot: Dict[str, Any]) -> None:
"""
Synthetic 360° camera band with worker/machine highlights.
"""
fov = snapshot["fov"]
drone = st.session_state.drone
workers = st.session_state.workers
machines = st.session_state.machines

width, height = 400, 80
img = np.zeros((height, width, 3), dtype=np.float32)

# base gradient
for y in range(height):
shade = 0.12 + 0.25 * (y / height)
img[y, :, :] = shade

# noise for "camera" feel
noise = np.random.normal(0, 0.06, img.shape)
img = np.clip(img + noise, 0, 1)

def add_box(x, y, color):
dx = x - drone["x"]
dy = y - drone["y"]

angle = math.atan2(dy, dx)
theta = (angle + math.pi) / (2 * math.pi) # 0..1
cx = int(theta * width)
cx = int(clamp(cx, 5, width - 5))
w_box = 12
h_box = 32
x0 = int(cx - w_box / 2)
x1 = int(cx + w_box / 2)
y0 = int(height / 2 - h_box / 2)
y1 = int(height / 2 + h_box / 2)
x0 = int(clamp(x0, 0, width - 1))
x1 = int(clamp(x1, 0, width - 1))
y0 = int(clamp(y0, 0, height - 1))
y1 = int(clamp(y1, 0, height - 1))
img[y0:y1, x0:x1, :] = img[y0:y1, x0:x1, :] * 0.2 + np.array(color) * 0.9

# highlight only actors inside FOV rect
for w in workers:
if (
fov["x_min"] <= w["x"] <= fov["x_max"]
and fov["y_min"] <= w["y"] <= fov["y_max"]
):
add_box(w["x"], w["y"], [0.1, 1.0, 0.4]) # green

for m in machines:
if (

fov["x_min"] <= m["x"] <= fov["x_max"]
and fov["y_min"] <= m["y"] <= fov["y_max"]
):
add_box(m["x"], m["y"], [1.0, 0.7, 0.1]) # orange

img = np.clip(img, 0.0, 1.0)

fig, ax = plt.subplots(figsize=(6, 2))
ax.imshow(img)
ax.set_axis_off()
ax.set_title(
"Synthetic 360° Camera Band — Workers (green) / Machines (orange)",
color="white",
fontsize=9,
)
fig.patch.set_facecolor("#101724")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ================================================================
# UI — HEADER
# ================================================================
header_left, header_right = st.columns([3, 1])

with header_left:
st.markdown(
"<h2 style='margin-bottom:0;'>Supervisor Drone HUD — Heavy Equipment
Safety</h2>",
unsafe_allow_html=True,
)
st.caption(
"Synthetic demo. Overhead supervisor UAV monitoring heavy equipment and workers "
"across a multi-zone site. All actuation is proposal-only; human gate decides."
)

with header_right:
st.write("")
st.write("")
st.session_state.human_gate_open = st.toggle(
"Human Gate OPEN",
value=st.session_state.human_gate_open,
help="Must be open before applying any proposal to drone state.",
)
st.metric("Tick", st.session_state.tick)

st.markdown("---")

# ================================================================
# CONTROLS
# ================================================================

ctrl1, ctrl2, ctrl3 = st.columns(3)
with ctrl1:
if st.button("▶ Step Simulation", use_container_width=True):
simulate_step()
with ctrl2:
if st.button("⟲ Reset", use_container_width=True):
init_state()
with ctrl3:
# Action mode buttons – they don't change logic yet, just log intent.
mode = st.radio("Supervisor Mode", ["OBSERVE", "ADVISE", "HOLD"], horizontal=True)
log_event("MODE", f"Mode set to {mode}")

# Ensure at least one snapshot
if not st.session_state.history:
st.session_state.history.append(compute_snapshot())
latest = st.session_state.history[-1]

# ================================================================
# TOP METRICS
# ================================================================
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
st.metric("Clarity (%)", latest["clarity"])
st.metric("State", latest["state"])
with m2:
st.metric("Risk (%)", latest["risk"])

st.metric("Predicted Risk", latest["predicted_risk"])
with m3:
st.metric("Coverage (site %)", f"{latest['coverage_frac']*100:.1f}")
st.metric("Min Worker–Machine Dist (m)", latest["min_worker_machine_dist"])
with m4:
st.metric("Drone Altitude (m)", round(st.session_state.drone["alt"], 1))
st.metric("Drone Zoom (0–1)", round(st.session_state.drone["zoom"], 2))
with m5:
st.metric("Zone Violations", len(latest["violations"]))
st.metric(
"Open Proposals",
sum(1 for p in st.session_state.proposals if p["status"] == "PENDING"),
)

st.markdown("---")

# ================================================================
# MAIN VISUALS ROW
# ================================================================
top_left, top_mid, top_right = st.columns([1.6, 1.1, 0.9])

with top_left:
st.subheader("Site Overview")
draw_site_map(latest)

with top_mid:

st.subheader("360° Camera Band")
draw_360_camera(latest)

st.subheader("Drone Manual Trim")
alt_delta = st.slider("Adjust altitude (Δ m)", -10.0, 10.0, 0.0, 1.0)
zoom_delta = st.slider("Adjust zoom (Δ)", -0.3, 0.3, 0.0, 0.05)
if st.button("Apply Manual Trim"):
st.session_state.drone["alt"] = clamp(
st.session_state.drone["alt"] + alt_delta, 10.0, 60.0
)
st.session_state.drone["zoom"] = clamp(
st.session_state.drone["zoom"] + zoom_delta, 0.1, 1.0
)
log_audit_event(
"MANUAL_TRIM",
{"alt_delta": alt_delta, "zoom_delta": zoom_delta},
)
st.session_state.history.append(compute_snapshot())
st.session_state.history = st.session_state.history[-MAX_HISTORY:]

with top_right:
st.subheader("Risk / Clarity Timeline")
if len(st.session_state.history) > 1:
hist_df = pd.DataFrame(
{
"tick": [h["tick"] for h in st.session_state.history],

"clarity": [h["clarity"] for h in st.session_state.history],
"risk": [h["risk"] for h in st.session_state.history],
}
).set_index("tick")
st.line_chart(hist_df)
else:
st.info("Step the simulation to build history.")

st.subheader("Current Tick Violations")
if latest["violations"]:
vdf = pd.DataFrame(latest["violations"])
st.dataframe(vdf, use_container_width=True, height=180)
else:
st.success("No active zone violations detected.")

st.markdown("---")

# ================================================================
# BOTTOM ROW — PROPOSALS / ANALYTICS / AUDIT
# ================================================================
bottom_left, bottom_mid, bottom_right = st.columns([1.4, 1.1, 1.0])

# ----- Proposals -----
with bottom_left:
st.subheader("Human-Gated Proposals")
if not st.session_state.proposals:

st.info(
"No proposals yet. Step the simulation to let the system surface suggestions."
)
else:
labels = [
f"#{p['id']} [{p['status']}] {p['title']}"
for p in st.session_state.proposals
]
selected = st.selectbox("Select proposal", labels)
selected_id = int(selected.split(" ")[0].replace("#", ""))
proposal = next(p for p in st.session_state.proposals if p["id"] == selected_id)

st.markdown(f"**Title:** {proposal['title']}")
st.markdown(f"**Status:** `{proposal['status']}`")
st.markdown(f"**Created:** {proposal['created']}")
st.markdown("**Rationale:**")
st.write(proposal["rationale"])
st.markdown("**Snapshot at Creation:**")
st.json(proposal["snapshot"])

col_a, col_b, col_c = st.columns(3)
approve = col_a.button(
"Approve", disabled=proposal["status"] != "PENDING", key="approve_btn"
)
reject = col_b.button(
"Reject", disabled=proposal["status"] != "PENDING", key="reject_btn"

)
defer = col_c.button(
"Defer", disabled=proposal["status"] != "PENDING", key="defer_btn"
)

if approve and proposal["status"] == "PENDING":
if not st.session_state.human_gate_open:
st.warning("Human gate is CLOSED. Open the gate before approving.")
log_audit_event(
"APPROVE_BLOCKED_GATE_CLOSED",
{"id": proposal["id"]},
)
else:
proposal["status"] = "APPROVED"
apply_proposal_action(proposal)
log_audit_event(
"PROPOSAL_APPROVED",
{"id": proposal["id"], "title": proposal["title"]},
)

if reject and proposal["status"] == "PENDING":
proposal["status"] = "REJECTED"
log_audit_event(
"PROPOSAL_REJECTED",
{"id": proposal["id"], "title": proposal["title"]},
)

if defer and proposal["status"] == "PENDING":
proposal["status"] = "DEFERRED"
log_audit_event(
"PROPOSAL_DEFERRED",
{"id": proposal["id"], "title": proposal["title"]},
)

# ----- Proposal Analytics + Telemetry -----
with bottom_mid:
st.subheader("Proposal Analytics")
ps = st.session_state.proposals
if ps:
total = len(ps)
approved = sum(1 for p in ps if p["status"] == "APPROVED")
rejected = sum(1 for p in ps if p["status"] == "REJECTED")
deferred = sum(1 for p in ps if p["status"] == "DEFERRED")
pending = sum(1 for p in ps if p["status"] == "PENDING")

metric_left, metric_right = st.columns(2)
with metric_left:
st.metric("Total Proposals", total)
st.metric("Approved", approved)
with metric_right:
st.metric("Pending", pending)
st.metric("Rejected", rejected)

dfp = pd.DataFrame(
{
"Status": ["APPROVED", "REJECTED", "DEFERRED", "PENDING"],
"Count": [approved, rejected, deferred, pending],
}
).set_index("Status")
st.bar_chart(dfp)
else:
st.info("No proposals yet.")

st.subheader("Raw Telemetry (Last 40 Ticks)")
dfh = (
pd.DataFrame(st.session_state.history)[
[
"tick",
"timestamp",
"clarity",
"risk",
"state",
"coverage_frac",
"min_worker_machine_dist",
]
]
.set_index("tick")
.tail(40)

)
st.dataframe(dfh, use_container_width=True, height=260)

# ----- Audit -----
with bottom_right:
st.subheader("Audit Chain (Last 25 Events)")
if st.session_state.audit_log:
recent = st.session_state.audit_log[-25:]
rows = []
for ev in recent:
rows.append(
{
"timestamp": ev["timestamp"],
"tick": ev["tick"],
"kind": ev["kind"],
"hash": ev["hash"][:10] + "...",
"prev": ev["prev_hash"][:10] + "...",
}
)
dfa = pd.DataFrame(rows)
st.dataframe(dfa, use_container_width=True, height=240)
with st.expander("Raw Audit JSON"):
st.code(json.dumps(recent, indent=2), language="json")
else:
st.info("No audit entries yet.")
