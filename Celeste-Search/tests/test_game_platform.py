from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from celeste_search import game_platform
from celeste_search.playback import configure_game, install_bridge, restore_configuration


class GamePlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.app = self.root / "GAME/Celeste-Mac/Celeste.app"
        self.resources = self.app / "Contents/Resources"
        self.resources.mkdir(parents=True)
        self.launcher = self.app / "Contents/MacOS/Celeste"
        self.launcher.parent.mkdir()
        self.launcher.write_bytes(b"native-launcher")

    def test_bundle_and_resources_resolve_to_same_data(self):
        self.assertEqual(game_platform.normalize_game_dir(self.app), self.resources)
        self.assertEqual(game_platform.normalize_game_dir(self.app / "Contents"), self.resources)
        self.assertEqual(game_platform.normalize_game_dir(self.resources), self.resources)
        with patch.object(game_platform.sys, "platform", "darwin"):
            self.assertEqual(game_platform.game_launch_command(self.app), [str(self.launcher)])

    def test_mac_original_is_rejected_before_any_install(self):
        original = self.root / "Steam/steamapps/common/Celeste"
        original.mkdir(parents=True)
        with patch.object(game_platform.sys, "platform", "darwin"), \
             patch.object(game_platform, "WORKSPACE", self.root):
            with self.assertRaisesRegex(ValueError, "project-local"):
                install_bridge(original, self.root / "nonexistent-build")
        self.assertEqual(list(original.iterdir()), [])

    def test_resources_symlink_cannot_bypass_original_protection(self):
        original = self.root / "Steam/Resources"
        original.mkdir(parents=True)
        self.resources.rmdir()
        self.resources.symlink_to(original, target_is_directory=True)
        with patch.object(game_platform.sys, "platform", "darwin"), \
             patch.object(game_platform, "WORKSPACE", self.root):
            with self.assertRaisesRegex(ValueError, "project-local"):
                install_bridge(self.app, self.root / "build")
        self.assertEqual(list(original.iterdir()), [])

    def test_running_copy_blocks_install_and_launch_before_any_mutation(self):
        with patch.object(game_platform.sys, "platform", "darwin"), \
             patch.object(game_platform, "WORKSPACE", self.root), \
             patch.object(game_platform, "game_is_running", return_value=True), \
             patch.object(game_platform, "backup_mac_saves") as backup, \
             patch.object(game_platform.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "already running"):
                install_bridge(self.app, self.root / "build")
            with self.assertRaisesRegex(ValueError, "already running"):
                game_platform.launch_game_process(self.app)
        backup.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(list(self.resources.iterdir()), [])

    def test_process_lookup_matches_executable_without_matching_arguments(self):
        with patch.object(game_platform.sys, "platform", "darwin"), \
             patch.object(game_platform.subprocess, "run") as processes:
            processes.return_value.stdout = "/bin/zsh\n  " + str(self.launcher) + "\n"
            self.assertTrue(game_platform.game_is_running(self.app))
            processes.return_value.stdout = "/bin/zsh -c " + str(self.launcher) + "\n"
            self.assertFalse(game_platform.game_is_running(self.app))

    def test_bridge_checks_all_dependencies_before_copying(self):
        build = self.root / "build"
        build.mkdir()
        (build / "CelesteInstrumentation.dll").write_bytes(b"bridge")
        with patch.object(game_platform.sys, "platform", "darwin"), \
             patch.object(game_platform, "WORKSPACE", self.root):
            with self.assertRaises(FileNotFoundError):
                install_bridge(self.resources, build)
            self.assertFalse((self.resources / "CelesteInstrumentation.dll").exists())
            for name in game_platform.MAC_BRIDGE_FILES:
                (build / name).write_bytes(name.encode())
            install_bridge(self.resources, build)
        self.assertTrue(all((self.resources / name).is_file() for name in game_platform.MAC_BRIDGE_FILES))

    def test_fresh_configuration_restores_absence_and_existing_content(self):
        config = configure_game(self.app, 25102)
        tree = ET.fromstring(config[0].read_bytes())
        self.assertEqual(tree.findtext("ConnectionPort"), "25102")
        self.assertEqual(tree.findtext("GamespeedMultiplier"), "1")
        self.assertIsNone(tree.find("AreaKey"))
        restore_configuration(config)
        self.assertFalse(config[0].exists())
        original = b"<GlobalParameters><Enabled>false</Enabled><Custom>retained</Custom></GlobalParameters>"
        config[0].write_bytes(original)
        restore_configuration(configure_game(self.resources, 25103))
        self.assertEqual(config[0].read_bytes(), original)

    def test_save_snapshots_are_verified_deduplicated_and_keep_original(self):
        source = self.root / "user/Celeste"
        saves = source / "Saves"
        saves.mkdir(parents=True)
        save = saves / "0.celeste"
        save.write_bytes(b"existing-user-save")
        destination = self.root / "snapshots"
        first = game_platform.backup_mac_saves(source, destination)
        self.assertEqual(game_platform.backup_mac_saves(source, destination), first)
        self.assertEqual((first / "Saves/0.celeste").read_bytes(), save.read_bytes())
        save.write_bytes(b"new-progress")
        second = game_platform.backup_mac_saves(source, destination)
        self.assertNotEqual(first, second)
        self.assertEqual(save.read_bytes(), b"new-progress")
        (second / "Saves/0.celeste").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "verification"):
            game_platform.backup_mac_saves(source, destination)

    def test_launch_uses_native_binary_and_explicit_environment(self):
        with patch.object(game_platform.sys, "platform", "darwin"), \
             patch.object(game_platform, "WORKSPACE", self.root), \
             patch.object(game_platform, "game_is_running", return_value=False), \
             patch.object(game_platform, "backup_mac_saves") as backup, \
             patch.object(game_platform.subprocess, "Popen") as popen:
            game_platform.launch_game_process(self.app, fna3d_driver="OpenGL", sdl_videodriver="cocoa")
        backup.assert_called_once()
        self.assertEqual(popen.call_args.args[0], [str(self.launcher)])
        self.assertEqual(popen.call_args.kwargs["cwd"], self.resources)
        self.assertEqual(popen.call_args.kwargs["env"]["FNA3D_FORCE_DRIVER"], "OpenGL")
        self.assertEqual(popen.call_args.kwargs["env"]["SDL_VIDEODRIVER"], "cocoa")
        self.assertEqual(popen.call_args.kwargs["env"]["SteamAppId"], "504230")
        self.assertEqual((self.resources / "steam_appid.txt").read_text(), "504230\n")

    def test_gui_launch_failure_restores_configuration_without_queuing_route(self):
        from celeste_search.assist import SharedAssistState
        from celeste_search.assist_gui import AssistWindow
        window = object.__new__(AssistWindow)
        window.args = SimpleNamespace(game_dir=self.resources, port=25102)
        window.shared = SharedAssistState()
        window.status = SimpleNamespace(set=Mock())
        window.game_process = None
        window.configuration = None
        with patch("celeste_search.assist_gui.install_bridge"), \
             patch("celeste_search.assist_gui.launch_game_process", side_effect=OSError("native launcher failed")):
            window.launch_game()
        self.assertFalse((self.resources / "InstrumentationParameters.xml").exists())
        self.assertIsNone(window.configuration)
        self.assertIn("native launcher failed", window.shared.snapshot()["message"])
        self.assertTrue(window.shared.commands.empty())

    def test_connected_gui_does_not_reinstall_or_launch_another_game(self):
        from celeste_search.assist import SharedAssistState
        from celeste_search.assist_gui import AssistWindow
        window = object.__new__(AssistWindow)
        window.shared = SharedAssistState()
        window.shared.update(connected=True)
        window.status = SimpleNamespace(set=Mock())
        with patch("celeste_search.assist_gui.install_bridge") as install, \
             patch("celeste_search.assist_gui.launch_game_process") as launch:
            window.launch_game()
        install.assert_not_called()
        launch.assert_not_called()
        self.assertTrue(window.shared.commands.empty())


if __name__ == "__main__":
    unittest.main()
