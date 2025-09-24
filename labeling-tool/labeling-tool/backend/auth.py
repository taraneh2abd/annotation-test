# auth.py
import time
from fastapi import HTTPException, Header, Depends
import jwt
from jwt import PyJWTError
from pydantic import BaseModel
import os

SECRET_KEY = os.getenv("SECRET_KEY", "supersecret123")
ALGORITHM = "HS256"

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

def create_token(username: str, expires_in: int = 3600):
    payload = {"sub": username, "exp": time.time() + expires_in}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(authorization: str = Header(...)):
    """Dependency for verifying Bearer token"""
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except (ValueError, PyJWTError):
        raise HTTPException(status_code=401, detail="Invalid token")

def login_user(req: LoginRequest):
    # Replace this with hashed password check in real app
    if req.username == "user" and req.password == "supersecret123":
        token = create_token(req.username)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")
