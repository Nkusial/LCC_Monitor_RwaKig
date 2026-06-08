# CI/CD

This project uses CI/CD in a local-first, portfolio-appropriate form. It does
not claim full cloud production deployment. Instead, it separates three concerns:

- **Continuous integration:** automated tests, geospatial validation checks, and WebGIS build checks.
- **Continuous delivery:** publication-ready static WebGIS and documentation artifacts delivered through GitHub Pages and release packaging.
- **Future deployment:** hosted FastAPI/PostGIS deployment when a dynamic backend is required.

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

Continuous delivery is present for the parts of the project that are designed to
be public and static at this stage:

- the hosted MapLibre WebGIS demo in `docs/app/`
- public documentation in `docs/`
- release-ready build artifacts from tagged runs

GitHub Pages publishes the WebGIS demo and static documentation from the `docs/`
folder on `main`. This means a clean update to the public release branch can be
served as a hosted portfolio artifact without deploying the local FastAPI/PostGIS
backend.

The release workflow also runs when a version tag is pushed, for example:

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

## What CD Means Here

For this local-first prototype, CD means **continuous delivery of static and
release-ready artifacts**, not continuous cloud deployment of every service.

Included now:

- validated commits on `main`
- hosted static WebGIS and documentation through GitHub Pages
- tagged release artifacts for reproducible portfolio milestones

Not included yet:

- hosted FastAPI runtime
- hosted PostGIS database
- online API authentication, monitoring, backups, or scaling

This is intentional. The current scope is a reproducible local-first GeoAI
system with a public WebGIS demo. Full cloud deployment should be added only when
a hosted dynamic backend is needed.

## Hosted Documentation

GitHub Pages publishes the WebGIS demo and static documentation from the `docs/`
folder on `main`.

See [Hosted Documentation](HOSTED_DOCS.md) for setup instructions.

The project intentionally does not use a custom Pages deployment workflow. This
avoids Pages permission/environment failures and keeps CI/CD focused on tests,
builds, static delivery, and tagged release artifacts.
