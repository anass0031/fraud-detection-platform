from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: Literal["analyst", "admin"]
    is_active: bool
    created_at: dt.datetime
    last_login_at: dt.datetime | None


class TokenResponse(BaseModel):
    """Access token in the body; the refresh token is set as an httpOnly cookie."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # seconds
    user: UserOut
