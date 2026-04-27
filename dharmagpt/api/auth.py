import hmac

from fastapi import Header, HTTPException

from core.config import get_settings


def require_staging_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = (get_settings().staging_api_key or "").strip()
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def require_admin_api_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = get_settings()
    expected_keys = [
        key.strip()
        for key in (
            settings.admin_api_key,
            settings.admin_operator_api_key,
            settings.staging_api_key,
        )
        if key and key.strip()
    ]
    provided = (x_admin_key or x_api_key or "").strip()
    if not expected_keys:
        return
    if not provided or not any(hmac.compare_digest(provided, key) for key in expected_keys):
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
