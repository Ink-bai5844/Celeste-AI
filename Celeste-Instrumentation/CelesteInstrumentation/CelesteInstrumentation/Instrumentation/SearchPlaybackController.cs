using System;
using System.Reflection;
using Celeste;
using Monocle;

namespace Instrumentation
{
	public static class SearchPlaybackController
	{
		private static readonly FieldInfo UnpauseTimer = typeof(Level).GetField("unpauseTimer", BindingFlags.Instance | BindingFlags.NonPublic);
		private static ulong lastEngineFrame = ulong.MaxValue;
		private static string room;
		private static long roomFrames;
		private static bool f11Held;
		private static readonly DialogueAdvancePulse DialoguePulse = new DialogueAdvancePulse();
		public static Level StoppedLevel;
		public static string LastControlEvent;
		public static double LastRoundTripMilliseconds;
		public static double MaxRoundTripMilliseconds;

		public static bool Enabled
		{
			get { return Instrumentation.SessionParameters != null && Instrumentation.SessionParameters.SearchMode; }
		}

		public static bool LiveMode
		{
			get { return Enabled && !Instrumentation.SessionParameters.EndOnDeath; }
		}

		public static void Reset()
		{
			lastEngineFrame = ulong.MaxValue;
			room = null;
			roomFrames = 0;
			StoppedLevel = null;
			LastControlEvent = "session_started";
			LastRoundTripMilliseconds = MaxRoundTripMilliseconds = 0;
			DialoguePulse.Next(false);
		}

		public static void OnSpawn()
		{
			roomFrames = 0;
			LastControlEvent = "spawned";
		}

		public static void BeforeVirtualInputs()
		{
			if (!Enabled || lastEngineFrame == Engine.FrameCounter)
			{
				return;
			}
			lastEngineFrame = Engine.FrameCounter;
			AssistInputState.CaptureCurrent();
			Level level = Engine.Scene as Level;
			Player player = level == null ? null : level.Tracker.GetEntity<Player>();
			Microsoft.Xna.Framework.Input.KeyboardState physicalKeys = Microsoft.Xna.Framework.Input.Keyboard.GetState();
			bool f11 = physicalKeys.IsKeyDown(Microsoft.Xna.Framework.Input.Keys.F11);
			bool requestReattach = f11 && !f11Held;
			f11Held = f11;
			if (LiveMode && physicalKeys.IsKeyDown(Microsoft.Xna.Framework.Input.Keys.F12))
			{
				SearchReconnect.PollLocalRequest(true);
				// Read the physical device directly, bypassing instrumented virtual
				// keys. Also works during the native death/respawn interval.
				Instrumentation.DetachByUser(player);
				return;
			}
			SearchReconnect.PollLocalRequest(false);
			if (!Instrumentation.inSession)
			{
				if (LiveMode && requestReattach) SearchReconnect.Request();
				if (LiveMode) SearchReconnect.Poll(player);
				return;
			}
			if (LiveMode && !Instrumentation.AssistMode && !InputsManager.PassThrough && level != null && (level.InCutscene || level.Overlay != null))
			{
				Textbox textbox = level.Tracker.GetEntity<Textbox>();
				object waiting;
				bool canContinue = !level.Paused && !level.SkippingCutscene && textbox != null && textbox.Opened &&
					SearchModelState.Read(textbox, "waitingForInput", out waiting) && waiting is bool && (bool)waiting;
				bool pulse = DialoguePulse.Next(canContinue);
				InputsManager.ApplyDialogueConfirm(pulse);
				if (pulse) LastControlEvent = "dialogue_confirm";
				return;
			}
			DialoguePulse.Next(false);
			if (player == null || !player.Active || player.Dead || Engine.FreezeTimer > 0f || Engine.DashAssistFreeze ||
				level.Transitioning || level.FrozenOrPaused || level.SkippingCutscene || level.InCutscene ||
				level.Overlay != null || level.RetryPlayerCorpse != null)
			{
				// Human menus, death animations and cutscenes stay native. A room
				// transition alone retains an already admitted airborne route.
				if (Instrumentation.AssistMode && (player == null || player.Dead || level.InCutscene || level.Overlay != null || level.FrozenOrPaused))
					InputsManager.PassPhysicalInput();
				InputsManager.HoldPlaybackInput();
				return;
			}
			if (UnpauseTimer == null)
			{
				Instrumentation.EndSession(player, false, "unsupported_runtime");
				return;
			}
			if ((float)UnpauseTimer.GetValue(level) > 0f)
			{
				InputsManager.HoldPlaybackInput();
				return;
			}
			if (room != level.Session.Level)
			{
				room = level.Session.Level;
				roomFrames = 0;
			}
			if (Instrumentation.SessionParameters.TimeoutSeconds > 0f && roomFrames >= Instrumentation.SessionParameters.TimeoutSeconds * 60.0)
			{
				Instrumentation.EndSession(player, false, "timeout");
				return;
			}
			byte input;
			try
			{
				// Engine.Update has set dt, but no virtual input or scene physics has
				// advanced yet. The original virtual-input update runs exactly once.
				string snapshot = SearchPlaybackState.Build(player);
				long started = System.Diagnostics.Stopwatch.GetTimestamp();
				input = Socket.SendSearchStateAndReceiveInput(snapshot, Instrumentation.AssistMode ? 50 : (LiveMode ? 1000 : 0));
				LastRoundTripMilliseconds = (System.Diagnostics.Stopwatch.GetTimestamp() - started) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
				MaxRoundTripMilliseconds = Math.Max(MaxRoundTripMilliseconds, LastRoundTripMilliseconds);
			}
			catch (Exception exception) when (Instrumentation.IsPlaybackConnectionError(exception))
			{
				Instrumentation.CancelDisconnectedPlayback();
				return;
			}
			if (input == 253 || (input == 255 && LiveMode))
			{
				Instrumentation.DetachPlayback(player);
				return;
			}
			if (input == 128)
			{
				InputsManager.PassPhysicalInput();
				Instrumentation.PlaybackFrames++;
				return;
			}
			if (Instrumentation.AssistMode && (!AssistInputState.Available || AssistInputState.HumanInputMask != 0 || AssistInputState.HumanMenuInput))
			{
				InputsManager.PassPhysicalInput();
				Instrumentation.PlaybackFrames++;
				LastControlEvent = "human_input";
				return;
			}
			if (Instrumentation.AssistMode && input <= 127 && !InputsManager.GameplayBindingsAvailable)
			{
				InputsManager.PassPhysicalInput();
				Instrumentation.PlaybackFrames++;
				LastControlEvent = "unsupported_input_bindings";
				return;
			}
			if (input == 254)
			{
				InputsManager.ApplyPlaybackInput(0);
				Instrumentation.PlaybackFrames++;
				LastControlEvent = "room_respawn_requested";
				// Golden berries replace the native death action with a chapter
				// restart. Never request that path as a current-room rollback.
				foreach (var follower in player.Leader.Followers)
				{
					Strawberry berry = follower.Entity as Strawberry;
					if (berry != null && berry.Golden && !berry.Winged)
					{
						LastControlEvent = "room_respawn_rejected_golden";
						return;
					}
				}
				if (!LiveMode)
				{
					LastControlEvent = "room_respawn_requires_live_mode";
					return;
				}
				if (player.Die(Microsoft.Xna.Framework.Vector2.Zero, true, true) == null)
				{
					LastControlEvent = "room_respawn_rejected_by_game";
				}
				return;
			}
			if (input == 255)
			{
				Instrumentation.EndSession(player, false, "stopped");
				return;
			}
			InputsManager.ApplyPlaybackInput(input);
			if (Instrumentation.AssistMode) LastControlEvent = "assist_input";
			Instrumentation.PlaybackFrames++;
			roomFrames++;
		}
	}
}
