# RewardHackWatch v1.3.0 Promotion Checklist

## Pre-Push

- [ ] Verify .gitignore includes all secrets and private files (.env, API keys, credentials, local configs)
- [ ] Verify no API keys, tokens, or passwords in any committed file
- [ ] Verify no private paths or usernames in committed files
- [ ] `npm run build` passes with 0 errors
- [ ] `pytest tests/ -q` passes
- [ ] Version is 1.3.0 in pyproject.toml, __init__.py, frontend/package.json, CHANGELOG.md
- [ ] README has no em dashes, no "Aerosta Research", no AI attribution
- [ ] CHANGELOG [1.3.0] section is complete
- [ ] LICENSE is Apache 2.0
- [ ] NOTICE says "Aerosta"

## Push and Release

- [ ] Push to GitHub
- [ ] Create v1.3.0 release with tag `v1.3.0`
- [ ] Write release notes from CHANGELOG [1.3.0]

## Screenshots

- [ ] Take screenshots of all 9 pages:
  - [ ] assets/screenshots/dashboard.png
  - [ ] assets/screenshots/quick-analysis.png
  - [ ] assets/screenshots/timeline.png
  - [ ] assets/screenshots/alerts.png
  - [ ] assets/screenshots/cross-model.png
  - [ ] assets/screenshots/cot-viewer.png
  - [ ] assets/screenshots/eval-workbench.png
  - [ ] assets/screenshots/session-logs.png
  - [ ] assets/screenshots/settings.png
- [ ] Drop screenshots into assets/screenshots/
- [ ] Update README if screenshot paths change
- [ ] Push screenshot commit

## Package Distribution

- [ ] Update HuggingFace model card with v1.3.0 notes
- [ ] Update PyPI package

## Community Posts

- [ ] Write HackerNews Show HN post
- [ ] Write Reddit r/MachineLearning [P] post
- [ ] Write Reddit r/LocalLLaMA post
- [ ] Write LessWrong post
- [ ] Write Twitter/X thread
