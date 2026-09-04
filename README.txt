IDShield — Final Local Release Candidate

Frontend
- frontend/src/App.jsx
- frontend/src/App.css

Backend
- backend/main.py

Run backend (inside backend/.venv):
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001

Run frontend:
npm run dev

The frontend uses /api and expects the Vite proxy to forward /api to http://127.0.0.1:8001.

Current behavior:
- Supported identity documents proceed to screening.
- Unknown/unrecognized uploads stop before tampering/face/risk stages.
- Case history is retained locally in browser storage.
- Evidence and Audit Trail pages show current screening information.
- Passport orientation is auto-corrected only when the rotated version improves document classification.
