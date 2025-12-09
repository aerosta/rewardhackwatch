# RewardHackWatch Assets

This directory contains visual assets for the project.

## Required Screenshots

To complete the release, take the following screenshots:

### 1. Dashboard Hero Image (Required for README)
- **File:** `dashboard_hero.png`
- **Source:** Timeline tab at http://localhost:8501
- **Dimensions:** ~1200x600px recommended
- **Content:** Show the risk timeline with hack scores, generalization risk, and deception score

### How to Capture

1. Start the dashboard:
   ```bash
   streamlit run rewardhackwatch/dashboard/app.py --server.port 8501
   ```

2. Open http://localhost:8501 in your browser

3. Take a screenshot of the Timeline tab showing:
   - The risk timeline graph
   - Top metrics (ML F1 Score, ML Accuracy, etc.)
   - Alert thresholds visible

4. Save as `assets/dashboard_hero.png`

### Optional Screenshots

- `dashboard_alerts.png` - Alerts tab
- `dashboard_cot_viewer.png` - CoT Viewer tab with highlighted suspicious text
- `dashboard_judge_comparison.png` - Judge Comparison tab
- `dashboard_quick_analysis.png` - Quick Analysis tab with sample output

## Adding to README

After capturing the hero image, add this to README.md after the badges:

```markdown
<p align="center">
  <img src="assets/dashboard_hero.png" alt="RewardHackWatch Dashboard" width="800">
</p>
```
