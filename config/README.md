# Sample configuration

These YAML files show the minimum required structure that the server expects under `/config`. The Docker image copies them into place at build time.

## Auth tokens
- `admin` entry in `data/api_keys.json` is seeded with the plaintext token `vfk_admin_token_12345678abcdef12` (roles: `admin`, `predict`).
- `predictor` entry uses `vfk_predict_token_87654321fedcba` (roles: `predict`).

If you regenerate tokens, write the SHA256 hash (prefixed with `sha256:`) into `/data/api_keys.json` and call `/admin/reload`.
