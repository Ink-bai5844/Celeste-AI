"""Reconnect is an owned file request; only a real connection confirms success."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from celeste_search.assist_reconnect import ReconnectRequest


class ReconnectRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.game_dir=Path(self.temporary.name)
        guard=patch("celeste_search.assist_reconnect.require_isolated_mac_game",
                    return_value=self.game_dir)
        self.guard=guard.start()
        self.addCleanup(guard.stop)
        self.now=[100.]
        self.utc=datetime(2026,11,13,12,30,0,tzinfo=timezone.utc)
        self.request=ReconnectRequest(self.game_dir,clock=lambda:self.now[0],
                                      utcnow=lambda:self.utc)
        self.request_file=self.game_dir/"AssistReconnect.request.json"
        self.status_file=self.game_dir/"AssistReconnect.status.json"

    def status(self,status,request_id=None):
        self.status_file.write_text(json.dumps(dict(
            schema="celeste-assist-reconnect-status-v1",
            request_id=request_id or self.request.request_id,status=status,
            processed_at_utc=self.utc.isoformat(),bridge_process_id=1234)),encoding="utf-8")

    def test_atomic_request_has_complete_schema_and_no_system_or_network_input(self):
        replace=os.replace
        writes=[]
        def inspect_atomic(source,destination,*args,**kwargs):
            source,destination=Path(source),Path(destination)
            self.assertNotEqual(source,destination)
            self.assertEqual(source.parent,destination.parent)
            self.assertFalse(destination.exists())
            writes.append(json.loads(source.read_text(encoding="utf-8")))
            return replace(source,destination,*args,**kwargs)
        with patch("os.replace",side_effect=inspect_atomic), \
                patch("subprocess.Popen",side_effect=AssertionError("must not launch anything")), \
                patch("subprocess.run",side_effect=AssertionError("must not send system input")), \
                patch("os.system",side_effect=AssertionError("must not run a shell")), \
                patch("socket.socket",side_effect=AssertionError("file IPC only")):
            self.assertTrue(self.request.request())
        self.assertEqual(len(writes),1)
        record=json.loads(self.request_file.read_text(encoding="utf-8"))
        self.assertEqual(record,writes[0])
        self.assertEqual(record["schema"],"celeste-assist-reconnect-v1")
        self.assertEqual(record["request_id"],self.request.request_id)
        self.assertTrue(record["request_id"])
        self.assertEqual(datetime.fromisoformat(record["requested_at_utc"].replace("Z","+00:00")),
                         self.utc)
        self.assertTrue(self.request.pending)
        self.guard.assert_called_once_with(self.game_dir)

    def test_project_game_guard_rejects_outside_directory_before_any_write(self):
        self.guard.side_effect=ValueError("project-local GAME copy required")
        with self.assertRaises(ValueError):
            self.request.request()
        self.assertFalse(self.request.pending)
        self.assertEqual(list(self.game_dir.iterdir()),[])

    def test_pending_duplicate_click_does_not_replace_the_request(self):
        self.assertTrue(self.request.request())
        contents=self.request_file.read_bytes()
        identity=self.request.request_id
        self.assertFalse(self.request.request())
        self.assertEqual(self.request_file.read_bytes(),contents)
        self.assertEqual(self.request.request_id,identity)

    def test_bridge_acknowledgements_are_not_connection_success(self):
        self.request.request()
        for acknowledgement in ("accepted_pending","ignored_connected","ignored_duplicate"):
            with self.subTest(acknowledgement=acknowledgement):
                self.status(acknowledgement)
                self.request.poll(False)
                self.assertTrue(self.request.pending)
        self.request.poll(True)
        self.assertFalse(self.request.pending)

    def test_stale_request_id_cannot_reject_a_new_request(self):
        self.request.request()
        identity=self.request.request_id
        self.status("rejected_stale","old-request")
        self.request.poll(False)
        self.assertTrue(self.request.pending)
        self.assertEqual(self.request.request_id,identity)

    def test_matching_rejection_releases_pending_and_allows_retry(self):
        for rejected in ("rejected_stale","rejected_future","rejected_invalid","rejected_user_detached"):
            with self.subTest(status=rejected):
                self.request.request()
                identity=self.request.request_id
                self.status(rejected)
                self.assertIsNotNone(self.request.poll(False))
                self.assertFalse(self.request.pending)
                self.assertTrue(self.request.request())
                self.assertNotEqual(self.request.request_id,identity)
                self.request.poll(True)

    def test_acknowledgement_does_not_reset_timeout_and_retry_uses_new_identity(self):
        self.request.request()
        identity=self.request.request_id
        self.now[0]+=29.9
        self.status("accepted_pending")
        self.request.poll(False)
        self.assertTrue(self.request.pending)
        self.now[0]+=.2
        self.assertIsNotNone(self.request.poll(False))
        self.assertFalse(self.request.pending)
        self.assertTrue(self.request.request())
        self.assertNotEqual(self.request.request_id,identity)
        self.request.poll(False)  # Old acknowledgement is still on disk.
        self.assertTrue(self.request.pending)

    def test_failed_atomic_write_never_claims_pending_and_can_retry(self):
        with patch("os.replace",side_effect=PermissionError("read-only destination")):
            with self.assertRaises(OSError):
                self.request.request()
        self.assertFalse(self.request.pending)
        self.assertFalse(self.request_file.exists())
        self.assertTrue(self.request.request())

    def test_close_removes_only_its_own_unconsumed_request_and_preserves_status(self):
        self.request.request()
        self.status("accepted_pending")
        self.request.close()
        self.assertFalse(self.request.pending)
        self.assertFalse(self.request_file.exists())
        self.assertTrue(self.status_file.exists())
        self.request.close()

    def test_close_preserves_a_later_request_owned_by_another_instance(self):
        self.request.request()
        replacement=dict(schema="celeste-assist-reconnect-v1",request_id="other-instance",
                         requested_at_utc=self.utc.isoformat())
        self.request_file.write_text(json.dumps(replacement),encoding="utf-8")
        self.request.close()
        self.assertEqual(json.loads(self.request_file.read_text(encoding="utf-8")),replacement)

    def test_close_tolerates_read_only_or_unreadable_files_without_removing_others(self):
        self.request.request()
        before=self.request_file.read_bytes()
        with patch.object(Path,"read_text",side_effect=PermissionError("unreadable")):
            self.request.close()
        self.assertEqual(self.request_file.read_bytes(),before)
        with patch.object(Path,"unlink",side_effect=PermissionError("read-only")):
            self.request.close()
        self.assertEqual(self.request_file.read_bytes(),before)


class ReconnectButtonTests(unittest.TestCase):
    def setUp(self):
        from celeste_search.assist_gui import AssistWindow
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.game_dir=Path(self.temporary.name)
        guard=patch("celeste_search.assist_reconnect.require_isolated_mac_game",
                    return_value=self.game_dir)
        guard.start()
        self.addCleanup(guard.stop)
        self.window=window=object.__new__(AssistWindow)
        window.shared=SimpleNamespace(snapshot=Mock(return_value={"connected":False}),
                                      update=Mock(),commands=Mock())
        window.reconnect_request=ReconnectRequest(self.game_dir)
        window.reconnect_button=Mock()
        window.status=Mock()
        window.launch=Mock(side_effect=AssertionError("reconnect must not launch game"))

    def test_disconnected_button_only_writes_ipc_and_never_submits_route_or_cancel(self):
        with patch("celeste_search.assist_gui.launch_game_process",
                   side_effect=AssertionError("must not launch game")):
            self.window.reconnect()
        self.assertTrue((self.game_dir/"AssistReconnect.request.json").is_file())
        self.assertTrue(self.window.reconnect_request.pending)
        self.window.shared.commands.put.assert_not_called()
        self.window.launch.assert_not_called()
        self.assertEqual(self.window.reconnect_button.configure.call_args.kwargs["state"],"disabled")

    def test_connected_button_does_not_write_request_or_change_control(self):
        self.window.shared.snapshot.return_value={"connected":True}
        self.window.reconnect()
        self.assertFalse((self.game_dir/"AssistReconnect.request.json").exists())
        self.assertFalse(self.window.reconnect_request.pending)
        self.window.shared.commands.put.assert_not_called()
        self.window.shared.update.assert_not_called()
        self.window.launch.assert_not_called()

    def test_io_failure_is_displayed_and_does_not_disable_retry_or_send_commands(self):
        with patch.object(self.window.reconnect_request,"request",side_effect=OSError("fixture read-only")):
            self.window.reconnect()
        self.assertIn("fixture read-only",self.window.status.set.call_args.args[0])
        self.assertIn("fixture read-only",self.window.shared.update.call_args.kwargs["message"])
        self.window.reconnect_button.configure.assert_not_called()
        self.window.shared.commands.put.assert_not_called()
        self.window.launch.assert_not_called()

    def test_pending_repeated_button_does_not_replace_request_or_repeat_submission(self):
        self.window.reconnect()
        identity=self.window.reconnect_request.request_id
        self.window.reconnect()
        self.assertEqual(self.window.reconnect_request.request_id,identity)
        self.window.shared.update.assert_called_once()
        self.window.reconnect_button.configure.assert_called_once()
        self.window.shared.commands.put.assert_not_called()


if __name__=="__main__":
    unittest.main()
