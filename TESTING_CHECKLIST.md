# RewardHackWatch v1.2 — Testing Checklist

## Automated Test Results (run by CI)

| Test | Status | Details |
|------|--------|---------|
| pytest tests/ -v --tb=short | PASS | 388 passed, 96 skipped, 0 failures |
| pip install -e . | PASS | Installed rewardhackwatch-1.2.0 |
| `from rewardhackwatch import RewardHackDetector` | PASS | Import succeeds |
| Quick Start code (README) | PASS | Returns risk_level, ml_score, detections |
| `rewardhackwatch version` | PASS | Shows v1.2.0 |
| `rewardhackwatch analyze trajectory.json` | PASS | Full detection output |
| `rewardhackwatch calibrate --help` | PASS | Shows usage |
| `rewardhackwatch serve --help` | PASS | Shows usage |
| POST /analyze | PASS | Returns 200 with risk_level |
| POST /analyze/batch | PASS | Returns 200 with results array |
| Frontend `npm run build` | PASS | Builds 4 chunks, 0 errors |
| Frontend `tsc --noEmit` | PASS | 0 type errors |

---

## Manual Testing Checklist

### Frontend Pages

- [ ] **Dashboard**: all 5 stat cards show data (Total Analysed, Flagged, Critical, Avg ML Score, High Risk)
- [ ] **Dashboard**: risk distribution donut chart renders with colored segments
- [ ] **Dashboard**: category bar chart renders horizontally with labels
- [ ] **Dashboard**: RMGI timeline area chart shows hack/RMGI scores
- [ ] **Dashboard**: recent alerts list shows colored risk badges

- [ ] **Quick Analysis**: paste code, click Analyze, see risk result
- [ ] **Quick Analysis**: "Load Hack Example" button populates textarea
- [ ] **Quick Analysis**: "Load Clean Example" button populates textarea
- [ ] **Quick Analysis**: radar chart renders for detection profile
- [ ] **Quick Analysis**: detection list shows pattern matches with evidence
- [ ] **Quick Analysis**: Export JSON button downloads file

- [ ] **Timeline**: charts render with demo data (hack, misalignment, RMGI lines)
- [ ] **Timeline**: toggle buttons show/hide individual score lines
- [ ] **Timeline**: RMGI transition marker renders at step 29
- [ ] **Timeline**: step detail table scrolls with color-coded scores
- [ ] **Timeline**: summary stats show steps, avg hack, peak RMGI

- [ ] **Alerts**: feed shows colored entries with risk badges
- [ ] **Alerts**: severity filter buttons work (critical/high/medium/low)
- [ ] **Alerts**: search field filters alerts by text
- [ ] **Alerts**: "Hide acknowledged" toggle works
- [ ] **Alerts**: acknowledge button marks alert as read

- [ ] **Cross-Model**: comparison table renders with F1/Precision/Recall
- [ ] **Cross-Model**: bar chart shows per-model performance
- [ ] **Cross-Model**: transfer matrix shows colored F1 cells
- [ ] **Cross-Model**: insights section displays transfer analysis

- [ ] **CoT Viewer**: highlighted patterns visible (red borders for suspicious)
- [ ] **CoT Viewer**: click to expand step shows full content
- [ ] **CoT Viewer**: keyword highlighting (sys.exit, mock.patch, etc.)
- [ ] **CoT Viewer**: hack score progression area chart renders
- [ ] **CoT Viewer**: suspicion flags show reason text

- [ ] **JSONL Analyzer**: drag-drop file loads entries
- [ ] **JSONL Analyzer**: "Analyze All" button processes entries
- [ ] **JSONL Analyzer**: risk distribution pie chart renders
- [ ] **JSONL Analyzer**: per-entry score bar chart renders
- [ ] **JSONL Analyzer**: entry list is clickable, shows detail view
- [ ] **JSONL Analyzer**: Export Results downloads JSON

- [ ] **Session Logs**: entries load with sortable columns
- [ ] **Session Logs**: risk level filter dropdown works
- [ ] **Session Logs**: source filter dropdown works
- [ ] **Session Logs**: search field filters by session ID
- [ ] **Session Logs**: Export button downloads JSON

- [ ] **Settings**: threshold slider moves and shows value
- [ ] **Settings**: auto-refresh toggle shows interval slider when on
- [ ] **Settings**: notifications toggle works
- [ ] **Settings**: Save button shows "Saved!" confirmation
- [ ] **Settings**: Reset button restores defaults
- [ ] **Settings**: Export Settings downloads JSON file
- [ ] **Settings**: version shows 1.2.0

### Layout & Responsive

- [ ] **Sidebar**: collapses/expands on click (chevron button at bottom)
- [ ] **Sidebar**: active page highlighted in blue
- [ ] **Sidebar**: logo and navigation icons visible when collapsed
- [ ] **Responsive**: looks good at 1280px width
- [ ] **Responsive**: looks good at 1920px width
- [ ] **Responsive**: stat cards stack on narrow screens

### Integration

- [ ] `rewardhackwatch serve` starts FastAPI on port 8000
- [ ] Frontend dev server (`npm run dev`) proxies API calls to FastAPI
- [ ] Streamlit fallback still works: `streamlit run rewardhackwatch/dashboard/app.py`

### Export

- [ ] Quick Analysis JSON export downloads a file
- [ ] Session Logs JSON export downloads a file
- [ ] Settings JSON export downloads a file
- [ ] JSONL Analyzer batch results export downloads a file
