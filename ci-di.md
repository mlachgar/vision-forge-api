CI/CD requirements

Build and release goals
- Add a complete CI/CD setup for GitHub Actions
- The pipeline must validate, test, build, and publish Docker images for vision-forge-api
- The project must use one Dockerfile with build args for all image variants
- The CI/CD workflow must support these public Docker image variants:
  - cpu-lite   => DEVICE=cpu, BUNDLE_MODEL=false
  - cpu-full   => DEVICE=cpu, BUNDLE_MODEL=true
  - gpu-lite   => DEVICE=gpu, BUNDLE_MODEL=false
  - gpu-full   => DEVICE=gpu, BUNDLE_MODEL=true
- latest must point to cpu-full
- The CI/CD design must be production-oriented, explicit, and easy to maintain

Repository and workflow files
Generate GitHub Actions workflows under:
- .github/workflows/ci.yml
- .github/workflows/docker-publish.yml

Also generate any helper files/scripts needed, such as:
- .github/dependabot.yml
- scripts/ci/check_versions.py if useful
- scripts/ci/compute_docker_tags.py if useful

CI workflow requirements
Create a CI workflow that runs on:
- pull_request
- push to main
- optionally workflow_dispatch

The CI workflow must:
1. set up the pinned Python version
2. install dependencies
3. run lint
4. run format check
5. run type checking
6. run tests
7. optionally validate configuration files
8. optionally do a lightweight Docker build smoke test for at least cpu-lite

The CI workflow should:
- use caching where it improves performance
- keep jobs readable
- fail fast on real errors
- upload useful artifacts if appropriate
- avoid unnecessary complexity

Docker publish workflow requirements
Create a separate workflow for Docker publishing that runs on:
- push of version tags like v1.0.0
- optionally workflow_dispatch for manual releases
- optionally push to main for edge/dev tags, if that is implemented cleanly

Publishing targets
- Publish to Docker Hub
- Use repository/image name vision-forge-api
- Make the Docker Hub namespace configurable via repository secrets or variables
- Do not hardcode credentials
- Use GitHub Secrets for Docker Hub authentication

Required Docker tags
For a release version v1.0.0, publish at least:
- vision-forge-api:1.0.0-cpu-lite
- vision-forge-api:1.0.0-cpu-full
- vision-forge-api:1.0.0-gpu-lite
- vision-forge-api:1.0.0-gpu-full

Also publish convenience tags:
- vision-forge-api:cpu-lite
- vision-forge-api:cpu-full
- vision-forge-api:gpu-lite
- vision-forge-api:gpu-full

Also publish:
- vision-forge-api:latest -> must point to cpu-full only

Optional but recommended:
- on push to main, publish edge tags such as:
  - vision-forge-api:edge-cpu-lite
  - vision-forge-api:edge-cpu-full
  - vision-forge-api:edge-gpu-lite
  - vision-forge-api:edge-gpu-full

Build strategy requirements
- Use a matrix build for the 4 variants
- Use the single docker/Dockerfile with build args
- Pass:
  - DEVICE=cpu or gpu
  - BUNDLE_MODEL=true or false
- Ensure the tag mapping is explicit and easy to understand
- Avoid duplicating large workflow blocks
- Use docker/setup-buildx-action
- Use docker/login-action
- Use docker/build-push-action
- Prefer modern supported GitHub Actions
- Use OCI labels where practical

Metadata and labels
Add image metadata labels where appropriate, such as:
- org.opencontainers.image.title
- org.opencontainers.image.description
- org.opencontainers.image.source
- org.opencontainers.image.version
- org.opencontainers.image.revision
- org.opencontainers.image.licenses

Version handling
- Derive release version from Git tag names like v1.0.0
- Strip the leading v when generating Docker tags
- Ensure latest is only published on a proper release workflow and only for cpu-full
- Do not assign latest to lite or gpu variants
- If implementing edge tags from main, keep them clearly separate from stable tags

Security and reliability requirements
- Do not expose secrets in logs
- Use least privilege permissions in workflows
- Pin major versions of GitHub Actions
- Keep workflow steps explicit
- Avoid unsafe shell practices
- If possible, include a Trivy or equivalent container vulnerability scan as a non-blocking or configurable step
- If practical, include dependency vulnerability scanning for Python packages
- Ensure CI/CD remains readable and maintainable

Caching requirements
- Use pip cache or equivalent
- Use Docker build cache with GitHub Actions cache backend if practical
- Keep cache usage correct and not overly complex

README and documentation requirements
Update README to include:
- CI badges if appropriate
- how CI works
- how Docker publishing works
- required GitHub secrets and variables
- release process
- exact example docker pull commands for:
  - latest
  - cpu-lite
  - cpu-full
  - gpu-lite
  - gpu-full
- explanation that latest maps to cpu-full
- explanation of edge tags if implemented

Release process requirements
Document a clean release process:
1. merge to main
2. create and push a git tag like v1.0.0
3. GitHub Actions builds and publishes all four images
4. latest is updated to cpu-full
5. versioned tags remain immutable by convention

Suggested secrets/variables
Use clear names such as:
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN
- DOCKERHUB_NAMESPACE

Generated output expectations for CI/CD
Generate:
1. .github/workflows/ci.yml
2. .github/workflows/docker-publish.yml
3. any helper scripts needed for tag computation or validation
4. README updates covering CI/CD and release workflow

Quality bar for CI/CD
- Workflows must be realistic and usable
- Avoid placeholder pseudo-YAML
- Keep them concise but complete
- Prefer maintainable matrix-based workflows over copy-paste duplication
- Ensure the Docker publish workflow clearly maps the 4 variants and latest behavior