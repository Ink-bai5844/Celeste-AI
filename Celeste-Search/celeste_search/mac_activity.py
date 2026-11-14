"""Scoped macOS activity assertion for the live input connection.

Apple documents user-initiated activities as preventing App Nap. The extra
latency-critical flag requests precise timer/I/O scheduling for the live bridge;
it is not a real-time guarantee. Idle system and display sleep remain allowed.
No process preference or system-wide configuration is changed.

https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheAppLevel.html
https://developer.apple.com/documentation/foundation/processinfo/activityoptions/latencycritical
"""
from __future__ import annotations

import ctypes
import sys
from threading import Lock


# NSProcessInfo.h: NS_OPTIONS(uint64_t, NSActivityOptions), available since 10.9.
USER_INITIATED_ALLOWING_IDLE_SYSTEM_SLEEP = 0x00FFFFFF & ~(1 << 20)
LATENCY_CRITICAL = 0xFF00000000
LIVE_ACTIVITY_OPTIONS = USER_INITIATED_ALLOWING_IDLE_SYSTEM_SLEEP | LATENCY_CRITICAL


class _ObjCActivityBackend:
    """Typed Objective-C messaging; each signature has its own callable."""

    def __init__(self):
        self.foundation = ctypes.CDLL(
            "/System/Library/Frameworks/Foundation.framework/Foundation",
            mode=ctypes.RTLD_GLOBAL,
        )
        self.objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        pointer = ctypes.c_void_p
        self.message = ctypes.CFUNCTYPE(pointer, pointer, pointer)(("objc_msgSend", self.objc))
        self.message_string = ctypes.CFUNCTYPE(pointer, pointer, pointer, ctypes.c_char_p)(
            ("objc_msgSend", self.objc))
        self.message_begin = ctypes.CFUNCTYPE(pointer, pointer, pointer, ctypes.c_uint64, pointer)(
            ("objc_msgSend", self.objc))
        self.message_void = ctypes.CFUNCTYPE(None, pointer, pointer)(("objc_msgSend", self.objc))
        self.message_end = ctypes.CFUNCTYPE(None, pointer, pointer, pointer)(("objc_msgSend", self.objc))
        self.process_info = None

    def selector(self, name):
        return self.objc.sel_registerName(name.encode("ascii"))

    def class_named(self, name):
        value = self.objc.objc_getClass(name.encode("ascii"))
        if not value:
            raise RuntimeError(f"Objective-C class unavailable: {name}")
        return value

    def new_pool(self):
        pool = self.message(self.class_named("NSAutoreleasePool"), self.selector("new"))
        if not pool:
            raise RuntimeError("Could not create an Objective-C autorelease pool")
        return pool

    def begin(self, options, reason):
        pool = self.new_pool()
        native_reason = None
        try:
            self.process_info = self.message(self.class_named("NSProcessInfo"), self.selector("processInfo"))
            if not self.process_info:
                raise RuntimeError("NSProcessInfo is unavailable")
            allocated = self.message(self.class_named("NSString"), self.selector("alloc"))
            native_reason = self.message_string(allocated, self.selector("initWithUTF8String:"), reason.encode("utf-8"))
            if not native_reason:
                raise RuntimeError("Could not create the activity reason")
            token = self.message_begin(self.process_info, self.selector("beginActivityWithOptions:reason:"),
                                       options, native_reason)
            if not token:
                raise RuntimeError("NSProcessInfo did not return an activity token")
            # beginActivity returns an autoreleased token. Keep explicit ownership
            # across callbacks, then release it only after endActivity.
            return self.message(token, self.selector("retain"))
        finally:
            if native_reason:
                self.message_void(native_reason, self.selector("release"))
            self.message_void(pool, self.selector("drain"))

    def end(self, token):
        pool = self.new_pool()
        try:
            try:
                self.message_end(self.process_info, self.selector("endActivity:"), token)
            finally:
                self.message_void(token, self.selector("release"))
        finally:
            self.message_void(pool, self.selector("drain"))


class MacActivity:
    """Hold one process activity until close, reporting failures without raising.

    Use ``with MacActivity(reason):`` or pair ``start()`` with ``close()`` in a
    connection's finally block. On other platforms this is a transparent no-op.
    ``status()`` is suitable for diagnostics; active means a token was acquired,
    not proof of a particular scheduler priority or future deadline compliance.
    """

    def __init__(self, reason="Celeste AI live input connection"):
        self.reason = reason
        self.supported = sys.platform == "darwin"
        self.warning = None
        self._backend = None
        self._token = None
        self._lock = Lock()

    def start(self):
        with self._lock:
            if not self.supported or self._token is not None:
                return self
            self.warning = None
            try:
                if not isinstance(self.reason, str) or not self.reason.strip() or "\0" in self.reason:
                    raise ValueError("Activity reason must be a nonempty string without NUL")
                self._backend = _ObjCActivityBackend()
                self._token = self._backend.begin(LIVE_ACTIVITY_OPTIONS, self.reason)
                if not self._token:
                    raise RuntimeError("No activity token was acquired")
            except Exception as error:
                self._token = None
                self._backend = None
                self.warning = f"macOS live activity unavailable: {type(error).__name__}: {error}"
        return self

    def close(self):
        with self._lock:
            token, backend = self._token, self._backend
            self._token = self._backend = None
            if token is not None:
                try:
                    backend.end(token)
                except Exception as error:
                    self.warning = f"macOS live activity release failed: {type(error).__name__}: {error}"
        return self

    def status(self):
        with self._lock:
            return {"supported": self.supported, "active": self._token is not None,
                    "options": LIVE_ACTIVITY_OPTIONS, "reason": self.reason, "warning": self.warning}

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
