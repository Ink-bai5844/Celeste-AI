"""Activity ownership and failure behavior, without altering machine settings."""
import unittest
from unittest.mock import Mock, patch

from celeste_search import mac_activity


class MacActivityTests(unittest.TestCase):
    def create(self, backend=None):
        backend = backend or Mock()
        backend.begin.return_value = 123
        with patch.object(mac_activity.sys, "platform", "darwin"):
            activity = mac_activity.MacActivity("live bridge test")
        return activity, backend

    def test_options_allow_idle_sleep_and_request_latency(self):
        options = mac_activity.LIVE_ACTIVITY_OPTIONS
        self.assertEqual(options, 0xFF00EFFFFF)
        self.assertFalse(options & (1 << 20))
        self.assertFalse(options & (1 << 40))
        self.assertEqual(options & mac_activity.LATENCY_CRITICAL, mac_activity.LATENCY_CRITICAL)

    def test_other_platform_never_loads_native_libraries(self):
        with patch.object(mac_activity.sys, "platform", "linux"), \
                patch.object(mac_activity, "_ObjCActivityBackend") as factory:
            with mac_activity.MacActivity() as activity:
                self.assertFalse(activity.status()["supported"])
                self.assertFalse(activity.status()["active"])
            factory.assert_not_called()

    def test_single_token_released_once_when_context_raises(self):
        activity, backend = self.create()
        with patch.object(mac_activity, "_ObjCActivityBackend", return_value=backend):
            with self.assertRaisesRegex(ValueError, "body failed"):
                with activity:
                    activity.start()
                    self.assertTrue(activity.status()["active"])
                    raise ValueError("body failed")
            activity.close()
        backend.begin.assert_called_once_with(mac_activity.LIVE_ACTIVITY_OPTIONS, "live bridge test")
        backend.end.assert_called_once_with(123)
        self.assertFalse(activity.status()["active"])

    def test_loading_failure_is_diagnostic_and_does_not_raise(self):
        activity, _ = self.create()
        with patch.object(mac_activity, "_ObjCActivityBackend", side_effect=OSError("not available")):
            activity.start()
        self.assertFalse(activity.status()["active"])
        self.assertIn("OSError: not available", activity.status()["warning"])
        activity.close()

    def test_nil_token_is_not_reported_as_active(self):
        activity, backend = self.create()
        backend.begin.return_value = None
        with patch.object(mac_activity, "_ObjCActivityBackend", return_value=backend):
            activity.start()
        self.assertFalse(activity.status()["active"])
        self.assertIn("No activity token", activity.status()["warning"])
        activity.close()
        backend.end.assert_not_called()

    def test_release_failure_is_diagnostic_and_not_retried(self):
        activity, backend = self.create()
        backend.end.side_effect = OSError("release failed")
        with patch.object(mac_activity, "_ObjCActivityBackend", return_value=backend):
            activity.start().close().close()
        backend.end.assert_called_once_with(123)
        self.assertFalse(activity.status()["active"])
        self.assertIn("release failed", activity.status()["warning"])

    def test_invalid_reason_does_not_call_native_backend(self):
        for reason in ("", "  ", "bad\0reason", None):
            with self.subTest(reason=reason), patch.object(mac_activity.sys, "platform", "darwin"), \
                    patch.object(mac_activity, "_ObjCActivityBackend") as factory:
                activity = mac_activity.MacActivity(reason).start()
                self.assertFalse(activity.status()["active"])
                self.assertIn("ValueError", activity.status()["warning"])
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
