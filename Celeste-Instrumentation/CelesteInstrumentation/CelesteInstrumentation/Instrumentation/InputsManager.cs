using System.Collections.Generic;
using Microsoft.Xna.Framework.Input;
using Celeste;

namespace Instrumentation
{
	public static class InputsManager
	{
		public static void ScheduleKey(Keys key)
		{
			InputsManager.NextKeysDown.Add((int)key);
		}

		public static bool Pressed(Keys key)
		{
			return InputsManager.KeysDown.Contains((int)key) && !InputsManager.PreviousKeysDown.Contains((int)key);
		}

		public static void Update()
		{
			InputsManager.PreviousKeysDown = new List<int>(InputsManager.KeysDown);
			InputsManager.KeysDown = new List<int>(InputsManager.NextKeysDown);
			InputsManager.NextKeysDown = new List<int>();
		}

		public static bool Check(Keys key)
		{
			return InputsManager.KeysDown.Contains((int)key);
		}

		public static bool Released(Keys key)
		{
			return !InputsManager.KeysDown.Contains((int)key) && InputsManager.PreviousKeysDown.Contains((int)key);
		}

		public static void ScheduleKey(int key)
		{
			InputsManager.NextKeysDown.Add(key);
		}

		public static void ScheduleKeys(int[] keys)
		{
			for (int i = 0; i < keys.Length; i++)
			{
				InputsManager.ScheduleKey(keys[i]);
			}
		}

		public static void Initialize(bool passThrough = true)
		{
			PassThrough = passThrough;
			PlaybackInput = 0;
			InputsManager.KeysDown = new List<int>();
			InputsManager.PreviousKeysDown = new List<int>();
			InputsManager.NextKeysDown = new List<int>();
		}

		public static void ApplyPlaybackInput(byte input)
		{
			if (input > 127) throw new System.ArgumentOutOfRangeException("input");
			PassThrough = false;
			PlaybackInput = input;
			InputsManager.PreviousKeysDown = new List<int>(InputsManager.KeysDown);
			InputsManager.KeysDown = new List<int>(InputsManager.ByteToKeys(input));
			InputsManager.NextKeysDown.Clear();
		}

		public static void PassPhysicalInput()
		{
			PassThrough = true;
			KeysDown.Clear();
			PreviousKeysDown.Clear();
			NextKeysDown.Clear();
		}

		// Called after the original virtual-input update. Keep the preceding
		// frame's effective human input separate from protocol command 128.
		public static void ObserveEffectiveInput(byte input)
		{
			PlaybackInput = input;
		}

		public static void HoldPlaybackInput()
		{
			// Frozen/transition ticks keep held keys, but are not fresh press edges.
			InputsManager.PreviousKeysDown = new List<int>(InputsManager.KeysDown);
			InputsManager.NextKeysDown.Clear();
		}

		public static void ApplyDialogueConfirm(bool confirm)
		{
			// A menu confirmation is not a gameplay action and cannot retain a
			// movement/dash button from the input that triggered the cutscene.
			ApplyPlaybackInput(0);
			if (!confirm || Settings.Instance.Confirm.Keyboard.Count == 0) return;
			Keys key = Settings.Instance.Confirm.Keyboard[0];
			if (Settings.Instance.Pause.Keyboard.Contains(key)) return;
			KeysDown.Add((int)key);
		}

		public static int[] ByteToKeys(byte inputByte)
		{
			List<int> positions = new List<int>();
			Keys[] bindings = keyMap;
			for (int i = 0; i < 7; i++)
			{
				if (((int)inputByte & 1 << i) != 0)
				{
					positions.Add((int)bindings[i]);
				}
			}
			return positions.ToArray();
		}

		private static Keys FirstKey(List<Keys> keys)
		{
			foreach (Keys key in keys) if (key != Keys.None) return key;
			return Keys.None;
		}

		// Settings are not loaded at instrumentation startup. Resolve bindings
		// lazily, and honor remapping performed in the normal settings screen.
		public static Keys[] keyMap
		{
			get
			{
				Settings settings = Settings.Instance;
				if (settings == null) return new Keys[7];
				return new [] { FirstKey(settings.Right.Keyboard), FirstKey(settings.Left.Keyboard),
					FirstKey(settings.Up.Keyboard), FirstKey(settings.Down.Keyboard), FirstKey(settings.Jump.Keyboard),
					FirstKey(settings.Dash.Keyboard), FirstKey(settings.Grab.Keyboard) };
			}
		}

		public static bool GameplayBindingsAvailable
		{
			get { foreach (Keys key in keyMap) if (key == Keys.None) return false; return true; }
		}

		public static List<int> KeysDown = new List<int>();

		public static List<int> PreviousKeysDown = new List<int>();

		public static List<int> NextKeysDown = new List<int>();

		public static byte PlaybackInput { get; private set; }
		public static bool PassThrough { get; private set; } = true;
		public static bool ReadingPhysicalInput;
	}
}
