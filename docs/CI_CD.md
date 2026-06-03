# CI/CD

This project uses lightweight CI/CD suitable for a local-first geospatial portfolio project.

## Continuous Integration

The CI workflow runs on pushes and pull requests to `main` or `master`.

Workflow:

```text
.github/workflows/ci.yml
```

It checks:

- Conda environment creation from `environment.yml`.
- Python backend and pipeline tests with `pytest -q`.
- Frontend dependency install with `npm ci`.
- WebGIS production build with `npm run build`.

## Continuous Delivery

The release workflow runs when a version tag is pushed, for example:

```powershell
git tag -a v0.1.1 -m "Next portfolio milestone"
git push origin v0.1.1
```

Workflow:

```text
.github/workflows/release.yml
```

It packages:

- documentation from `docs/`
- frontend production build from `web/dist`
- `README.md`
- `CHANGELOG.md`
- `environment.yml`
- `pyproject.toml`

The workflow uploads these as a GitHub Actions artifact named:

```text
geoai-change-monitor-release
```

## Why This Is Appropriate Here

This project is intentionally local-first. It does not deploy to cloud infrastructure yet because the current scope is a reproducible portfolio system, not a hosted production service.

The CD layer therefore focuses on release packaging:

- verify the code
- build the WebGIS
- package docs and build outputs
- attach a downloadable artifact to tagged runs

Future production CD could add container image publishing, hosted documentation, or cloud deployment once the project has a deployment target.

## Hosted Documentation

GitHub Pages publishes the WebGIS demo and static documentation from the `docs/` folder on `main`.

See [Hosted Documentation](HOSTED_DOCS.md) for setup instructions.

The project intentionally does not use a custom Pages deployment workflow. This avoids Pages permission/environment failures and keeps CI/CD focused on tests, builds, and tagged release artifacts.
