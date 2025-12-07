________________________________________
Supervisor Drone HUD v2 — Heavy Equipment Site Oversight
A cinematic, site-scale synthetic safety HUD built with Streamlit.
It simulates an overhead supervisor UAV monitoring workers, machines, risk, and zone compliance across a multi-zone heavy-equipment site.
All positions, movements, and signals are synthetic.
No real-world control, actuation, or autonomy is performed.
The system limits itself to human-gated proposals for small drone-state trims.
________________________________________
Features
🔭 Synthetic Overhead UAV View
•	Real-time top-down map of workers, machines, zones, and drone FOV
•	Adjustable altitude + zoom shaping the clarity/risk model
•	Conflict callouts for highest-severity zone violations
📡 360° Camera Band
•	Synthetic panoramic strip
•	Highlights workers (green) and machines (orange) visible inside FOV
🧠 Clarity + Risk Engine
Computed each tick using:
•	FOV coverage fraction
•	Zone violations + severity
•	Worker–machine proximity
•	Altitude/zoom sanity
•	Predictive risk model
Outputs: clarity, risk, predicted risk, site state (STABLE → CRITICAL).
📝 Human-Gated Proposal System
The drone never auto-acts.
It only suggests small trims, such as:
•	Recenter drone over conflict cluster
•	Increase altitude
•	Reduce zoom
•	Hold and monitor
Humans approve/reject/defer via UI.
All actions require the Human Gate toggle to be open.
📈 Telemetry, History & Audit Chain
•	Rolling 300-tick history with clarity/risk chart
•	Proposal analytics dashboard
•	Tamper-evident SHA-256 audit chain for all important events
•	Downloadable JSON audit segments
________________________________________
Running the App
1. Install dependencies
pip install -r requirements.txt
2. Launch the HUD
streamlit run app.py
3. Optional
Place an aerial background image at:
/mnt/data/A0A0FBF0-9206-41B1-B530-C500C5C891BB.png
If missing, the app falls back to a procedural texture.
________________________________________
Project Structure
app.py                 # Main Streamlit application
requirements.txt       # Dependencies
README.md              # Documentation
________________________________________
Intended Use
This system is for research, simulation, UI prototyping, and human-in-the-loop autonomy concepts only.
It must not be integrated with real equipment, real UAVs, or real jobsite telemetry.
________________________________________
License
MIT 
________________________________________
