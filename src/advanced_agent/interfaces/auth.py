"""
認証・認可システム

API キー認証、JWT トークン、レート制限機能
"""

import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

import jwt
from fastapi import HTTPException, status
from pydantic import BaseModel


class APIKey(BaseModel):
    """API キー情報"""
    key_id: str
    key_hash: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    permissions: List[str] = field(default_factory=list)
    rate_limit: int = 1000  # requests per hour
    metadata: Dict[str, Any] = field(default_factory=dict)


class JWTToken(BaseModel):
    """JWT トークン情報"""
    token: str
    user_id: str
    expires_at: datetime
    permissions: List[str]


@dataclass
class RateLimitInfo:
    """レート制限情報"""
    requests: deque = field(default_factory=deque)
    limit: int = 1000
    window_seconds: int = 3600  # 1 hour


class AuthenticationManager:
    """認証管理"""
    
    def __init__(self,
                 jwt_secret: str = None,
                 jwt_algorithm: str = "HS256",
                 jwt_expiry_hours: int = 24):
        
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiry_hours = jwt_expiry_hours
        
        # API キー管理
        self.api_keys: Dict[str, APIKey] = {}
        
        # レート制限管理
        self.rate_limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        
        # デフォルト API キー作成（デモ用）
        self._create_default_api_key()
    
    def _create_default_api_key(self):
        """デフォルト API キー作成"""
        default_key = "sk-advanced-agent-demo-key-12345"
        key_hash = self._hash_api_key(default_key)
        
        api_key = APIKey(
            key_id="default",
            key_hash=key_hash,
            name="Default Demo Key",
            created_at=datetime.now(),
            permissions=["*"],  # 全権限
            rate_limit=10000,  # 高いレート制限
            metadata={"type": "demo"}
        )
        
        self.api_keys[key_hash] = api_key
    
    def _hash_api_key(self, api_key: str) -> str:
        """API キーハッシュ化"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """API キー検証"""
        if not api_key:
            return None
        
        key_hash = self._hash_api_key(api_key)
        api_key_info = self.api_keys.get(key_hash)
        
        if not api_key_info:
            return None
        
        # アクティブ状態チェック
        if not api_key_info.is_active:
            return None
        
        # 有効期限チェック
        if api_key_info.expires_at and datetime.now() > api_key_info.expires_at:
            return None
        
        return api_key_info
    
    def create_api_key(self,
                      name: str,
                      permissions: List[str] = None,
                      rate_limit: int = 1000,
                      expires_in_days: Optional[int] = None) -> tuple[str, APIKey]:
        """新しい API キー作成"""
        
        # ランダムキー生成
        key_id = secrets.token_urlsafe(8)
        raw_key = f"sk-advanced-agent-{key_id}-{secrets.token_urlsafe(16)}"
        key_hash = self._hash_api_key(raw_key)
        
        # 有効期限設定
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.now(),
            expires_at=expires_at,
            permissions=permissions or [],
            rate_limit=rate_limit
        )
        
        self.api_keys[key_hash] = api_key
        
        return raw_key, api_key
    
    def revoke_api_key(self, key_id: str) -> bool:
        """API キー無効化"""
        for api_key in self.api_keys.values():
            if api_key.key_id == key_id:
                api_key.is_active = False
                return True
        return False
    
    def create_jwt_token(self,
                        user_id: str,
                        permissions: List[str] = None) -> JWTToken:
        """JWT トークン作成"""
        
        expires_at = datetime.now() + timedelta(hours=self.jwt_expiry_hours)
        
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "exp": expires_at.timestamp(),
            "iat": datetime.now().timestamp()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        return JWTToken(
            token=token,
            user_id=user_id,
            expires_at=expires_at,
            permissions=permissions or []
        )
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """JWT トークン検証"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # 有効期限チェック
            if datetime.now().timestamp() > payload.get("exp", 0):
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def check_rate_limit(self, identifier: str, limit: Optional[int] = None) -> bool:
        """レート制限チェック"""
        current_time = time.time()
        rate_info = self.rate_limits[identifier]
        
        # 制限値設定
        if limit:
            rate_info.limit = limit
        
        # 古いリクエストを削除
        window_start = current_time - rate_info.window_seconds
        while rate_info.requests and rate_info.requests[0] < window_start:
            rate_info.requests.popleft()
        
        # 制限チェック
        if len(rate_info.requests) >= rate_info.limit:
            return False
        
        # 新しいリクエストを記録
        rate_info.requests.append(current_time)
        return True
    
    def get_rate_limit_info(self, identifier: str) -> Dict[str, Any]:
        """レート制限情報取得"""
        current_time = time.time()
        rate_info = self.rate_limits[identifier]
        
        # 古いリクエストを削除
        window_start = current_time - rate_info.window_seconds
        while rate_info.requests and rate_info.requests[0] < window_start:
            rate_info.requests.popleft()
        
        remaining = max(0, rate_info.limit - len(rate_info.requests))
        reset_time = int(current_time + rate_info.window_seconds)
        
        return {
            "limit": rate_info.limit,
            "remaining": remaining,
            "reset": reset_time,
            "window_seconds": rate_info.window_seconds
        }
    
    def check_permission(self, api_key_info: APIKey, required_permission: str) -> bool:
        """権限チェック"""
        if "*" in api_key_info.permissions:
            return True
        
        return required_permission in api_key_info.permissions
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """API キー一覧取得"""
        keys = []
        for api_key in self.api_keys.values():
            keys.append({
                "key_id": api_key.key_id,
                "name": api_key.name,
                "created_at": api_key.created_at.isoformat(),
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "is_active": api_key.is_active,
                "permissions": api_key.permissions,
                "rate_limit": api_key.rate_limit
            })
        return keys


class AuthenticationError(HTTPException):
    """認証エラー"""
    
    def __init__(self, detail: str = "認証が必要です"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class PermissionError(HTTPException):
    """権限エラー"""
    
    def __init__(self, detail: str = "権限が不足しています"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


class RateLimitError(HTTPException):
    """レート制限エラー"""
    
    def __init__(self, detail: str = "レート制限に達しました", headers: Dict[str, str] = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers or {}
        )