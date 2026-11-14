"""Explicit, focus-free reconnect requests to the project-local game bridge."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from .game_platform import require_isolated_mac_game


REQUEST_SCHEMA = "celeste-assist-reconnect-v1"
STATUS_SCHEMA = "celeste-assist-reconnect-status-v1"
REQUEST_FILE = "AssistReconnect.request.json"
STATUS_FILE = "AssistReconnect.status.json"
TIMEOUT_SECONDS = 30


class ReconnectRequest:
    """An acknowledgement is not a connection; wait for a live handshake."""

    def __init__(self, game_dir, *, clock=time.monotonic, utcnow=None):
        self.game_dir = Path(game_dir)
        self.clock = clock
        self.utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self.request_id = None
        self.pending = False
        self.started = 0.
        self.message = None

    def request(self):
        if self.pending:
            return False
        directory = require_isolated_mac_game(self.game_dir)
        if not directory.is_dir():
            raise FileNotFoundError("找不到辅助版游戏目录")
        identifier = uuid4().hex
        payload = dict(schema=REQUEST_SCHEMA, request_id=identifier,
                       requested_at_utc=self.utcnow().astimezone(timezone.utc).isoformat())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                    prefix=".assist-reconnect-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream)
            os.replace(temporary, directory / REQUEST_FILE)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.game_dir = directory
        self.request_id = identifier
        self.started = self.clock()
        self.pending = True
        self.message = "正在重新连接游戏…"
        return True

    def _status(self):
        try:
            path = self.game_dir / STATUS_FILE
            if path.stat().st_size > 4096:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(value, dict) and value.get("schema") == STATUS_SCHEMA
                    and value.get("request_id") == self.request_id):
                return value.get("status")
        except (OSError, ValueError):
            pass
        return None

    def poll(self, connected):
        if not self.pending:
            return None
        if connected:
            self.close()
            return "已重新连接当前游戏，手动控制"
        status = self._status()
        if isinstance(status, str) and status.startswith("rejected_"):
            self.close()
            return ("游戏仍在执行断开操作；松开 F12 后再点重新连接"
                    if status == "rejected_user_detached" else "重连请求已失效，请再次点击重新连接")
        if self.clock() - self.started >= TIMEOUT_SECONDS:
            self.close()
            return "游戏未回应重连；请确认辅助版游戏已进入关卡，并已重新启动以加载新版桥接"
        if status in {"accepted_pending", "ignored_duplicate", "ignored_connected"}:
            self.message = "游戏已收到重连请求，等待连接…"
        return self.message

    def close(self):
        """Remove only our unconsumed request; never remove another caller's."""
        self.pending = False
        if self.request_id is None:
            return
        path = self.game_dir / REQUEST_FILE
        try:
            if path.stat().st_size <= 4096:
                value = json.loads(path.read_text(encoding="utf-8"))
                if (isinstance(value, dict) and value.get("schema") == REQUEST_SCHEMA
                        and value.get("request_id") == self.request_id):
                    path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
