using Celeste;
using Monocle;
using Microsoft.Xna.Framework.Input;

namespace Instrumentation
{
	// These reads use the hardware snapshot that MInput sampled immediately
	// before UpdateVirtualInputs. They do not update virtual inputs or buffers.
	public static class AssistInputState
	{
		public static byte HumanInputMask { get; private set; }
		public static bool HumanMenuInput { get; private set; }
		public static bool Available { get; private set; }
		public static long HumanInputEpoch { get; private set; }

		private static bool Held(Binding binding, float threshold = 0.2f)
		{
			return binding != null && binding.Check(Input.Gamepad, threshold);
		}

		public static void CaptureCurrent()
		{
			Available = Settings.Instance != null && Input.MoveX != null && Input.MoveY != null && Input.Jump != null &&
				Input.Dash != null && Input.Grab != null && MInput.Keyboard != null && MInput.GamePads != null;
			HumanInputMask = 0;
			HumanMenuInput = false;
			if (!Available) return;
			InputsManager.ReadingPhysicalInput = true;
			try
			{
				Settings settings = Settings.Instance;
				// Include move-only and dash-only bindings: even inputs which the
				// seven-bit simulator cannot represent must block a stale takeover.
				if (Held(settings.Right) || Held(settings.RightMoveOnly) || Held(settings.RightDashOnly)) HumanInputMask |= 1;
				if (Held(settings.Left) || Held(settings.LeftMoveOnly) || Held(settings.LeftDashOnly)) HumanInputMask |= 2;
				if (Held(settings.Up) || Held(settings.UpMoveOnly) || Held(settings.UpDashOnly)) HumanInputMask |= 4;
				if (Held(settings.Down) || Held(settings.DownMoveOnly) || Held(settings.DownDashOnly)) HumanInputMask |= 8;
				if (Held(settings.Jump)) HumanInputMask |= 16;
				if (Held(settings.Dash) || Held(settings.DemoDash)) HumanInputMask |= 32;
				if (Held(settings.Grab)) HumanInputMask |= 64;
				HumanMenuInput = MInput.Keyboard.Check(Keys.Escape) || Held(settings.Pause) ||
					Held(settings.QuickRestart) || Held(settings.Confirm) || Held(settings.Cancel) || Held(settings.Talk) ||
					Held(settings.MenuLeft) || Held(settings.MenuRight) || Held(settings.MenuUp) || Held(settings.MenuDown) || Held(settings.Journal);
				if (HumanInputMask != 0 || HumanMenuInput) HumanInputEpoch++;
			}
			finally { InputsManager.ReadingPhysicalInput = false; }
		}

		public static void ObserveEffectiveInput()
		{
			if (Input.MoveX == null || Input.MoveY == null || Input.Jump == null || Input.Dash == null || Input.Grab == null) return;
			byte mask = 0;
			if (Input.MoveX.Value > 0) mask |= 1;
			if (Input.MoveX.Value < 0) mask |= 2;
			if (Input.MoveY.Value < 0) mask |= 4;
			if (Input.MoveY.Value > 0) mask |= 8;
			if (Input.Jump.Check) mask |= 16;
			if (Input.Dash.Check) mask |= 32;
			if (Input.Grab.Check) mask |= 64;
			InputsManager.ObserveEffectiveInput(mask);
		}
	}
}
