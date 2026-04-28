## Summary

Describe what changed and why. Include the user-facing or operator-facing impact.

## Type of Change

- [ ] Bug fix
- [ ] Feature
- [ ] Admin or operations tooling
- [ ] Data pipeline or ingestion
- [ ] Mobile app
- [ ] CI, tests, or developer workflow
- [ ] Documentation

## Validation

- [ ] Unit tests: `PYTHONPATH=. python -m pytest tests/unit/ -v`
- [ ] Backend tests: `PYTHONPATH=. python -m pytest ../tests/backend -v`
- [ ] Integration tests: `PYTHONPATH=. python -m pytest tests/integration/ -v --timeout=120`
- [ ] Smoke/contract tests: `PYTHONPATH=. python -m pytest ../tests/backend -v -m "smoke or contract"`
- [ ] Docker build: `docker build -t dharmagpt-ci .`
- [ ] Manual admin/API check

## Configuration and Data

- [ ] No new environment variables
- [ ] New or changed environment variables are documented in `dharmagpt/.env.example`
- [ ] No secrets, generated data, local databases, or private uploads are included
- [ ] Database/schema migration or backfill steps are documented below

## Deployment Notes

List any required deployment, migration, indexing, or service restart steps.

## Screenshots or Logs

Add screenshots for UI changes, or paste the key command output for backend/CI changes.

## Follow-Ups

List known gaps, risks, or follow-up work that should not block this PR.
