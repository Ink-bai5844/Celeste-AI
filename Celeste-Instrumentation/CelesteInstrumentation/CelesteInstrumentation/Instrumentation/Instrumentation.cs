using System;
using System.IO;
using System.Text;
using Celeste;
using Monocle;
using System.Xml.Serialization;

namespace Instrumentation
{
	public static class Instrumentation
	{
		public static void Initialize()
        {
			XmlSerializer serializer = new XmlSerializer(typeof(GlobalParameters));
			using (FileStream fileStream = new FileStream(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "InstrumentationParameters.xml"), FileMode.Open))
			{
				GlobalParameters = (GlobalParameters)serializer.Deserialize(fileStream);
			}
			if (AssistMode)
			{
				// Enable observations, but leave the original menus, save selection,
				// chapter loader and player input entirely under native control.
				SessionParameters = new SessionParameters { SearchMode = true, EndOnDeath = false, EndOnLevelExit = false };
				InputsManager.Initialize(true);
				SearchPlaybackController.Reset();
				SearchReconnect.Request();
			}
		}

		public static bool AssistMode { get { return GlobalParameters != null && GlobalParameters.AssistMode; } }

		public static void BeginSession(SessionParameters sessionParameters)
		{
			if (AssistMode) throw new InvalidOperationException("Assist mode attaches to the native scene and cannot start an instrumentation session.");
			SearchReconnect.Disable();
			inSession = true;
			SessionParameters = sessionParameters;
			PlaybackFrames = 0;
			PlaybackSpawns = 0;
			PlaybackDeaths = 0;
			lastSpawnPlayer = null;
			InputsManager.Initialize(false);
			SearchPlaybackController.Reset();
			SearchEnvironmentState.Reset();
			SaveData.Start(new SaveData
			{
				Name = "run",
				AssistMode = false,
				VariantMode = false,
			}, 666);
			Engine.Scene = new LevelLoader(new Celeste.Session(new AreaKey(int.Parse(SessionParameters.AreaKey), (AreaMode)SessionParameters.AreaMode), null, null)
			{
				Level = SessionParameters.Level,
				InArea = true,
				JustStarted = false
			}, null);
		}

		public static void EndSession(Player player, bool died = false, string reason = null)
		{
			if (inSession && SearchPlaybackController.LiveMode)
			{
				DetachPlayback(player, reason ?? (died ? "death" : "ended"));
				return;
			}
			if (inSession)
            {
				if (SessionParameters.SearchMode)
				{
					SearchPlaybackController.StoppedLevel = player.SceneAs<Level>();
				}
				if (SessionParameters.SearchMode)
				{
					try
					{
						Socket.SendSearchDoneAndReceiveAck(SearchPlaybackState.Build(player, died, reason ?? (died ? "death" : "ended")));
					}
					catch (Exception exception) when (IsPlaybackConnectionError(exception))
					{
						CancelDisconnectedPlayback();
						return;
					}
				}
				else
				{
					player.Components.Get<PlayerStateTracker>().PlayerState.Died = died;
					Socket.SendAndReceiveData(Encoding.ASCII.GetBytes("END"));
					Socket.SendAndReceiveData(player.Components.Get<PlayerStateTracker>().PlayerState.Serialize());
				}
				InputsManager.Initialize();
				Engine.Scene = new WaitingScene();
				GC.Collect();
				inSession = false;
			}
			
		}

		public static bool IsPlaybackConnectionError(Exception exception)
		{
			return exception is IOException || exception is System.Net.Sockets.SocketException || exception is ObjectDisposedException;
		}

		public static void CancelDisconnectedPlayback()
		{
			// A lost controller must release the existing game, never reload it.
			ReleasePlaybackControl();
			SearchPlaybackController.LastControlEvent = "controller_disconnected";
			if (SearchPlaybackController.LiveMode) SearchReconnect.Request();
		}

		public static void ResumePlayback(Player player)
		{
			// Reattach to the current session: no SaveData.Start, LevelLoader,
			// respawn, pause, or reset of accumulated run counters.
			InputsManager.Initialize();
			SearchPlaybackController.StoppedLevel = null;
			inSession = true;
			RegisterPlaybackSpawn(player);
			SearchPlaybackController.LastControlEvent = "controller_reattached";
		}

		public static void DetachPlayback(Player player, string reason = "detached")
		{
			if (!inSession) return;
			InputsManager.PassPhysicalInput();
			try
			{
				Socket.SendSearchDoneAndReceiveAck(SearchPlaybackState.Build(player, false, reason), AssistMode ? 50 : 1000);
			}
			catch (Exception exception) when (IsPlaybackConnectionError(exception))
			{
				// The current level and native respawn are independent of the socket.
			}
			finally
			{
				ReleasePlaybackControl();
			}
		}

		public static void DetachByUser(Player player)
		{
			SearchReconnect.Disable();
			Player observedPlayer = player ?? lastSpawnPlayer;
			if (observedPlayer != null)
			{
				DetachPlayback(observedPlayer, "user_detached");
			}
			else
			{
				// Before the first spawn there is no valid player observation yet.
				ReleasePlaybackControl();
			}
		}

		private static void ReleasePlaybackControl()
		{
			SearchReconnect.Disable();
			inSession = false;
			SearchPlaybackController.StoppedLevel = null;
			SearchPlaybackController.LastControlEvent = "detached";
			InputsManager.Initialize();
			Socket.Reset();
		}

		public static void RegisterPlaybackSpawn(Player player)
		{
			if (!inSession || !SearchPlaybackController.Enabled || player == null || player == lastSpawnPlayer) return;
			lastSpawnPlayer = player;
			PlaybackSpawns++;
			SearchPlaybackController.OnSpawn();
		}

		public static void OnTransition(Player player)
        {
			if (SessionParameters == null || (SearchPlaybackController.Enabled && !inSession)) return;
			PlayerStateTracker tracker = player.Components.Get<PlayerStateTracker>();
			if (tracker == null) return;
			tracker.PlayerState.FinishedLevelsNumber += 1;
			if (Instrumentation.SessionParameters.EndOnLevelExit)
			{
				Instrumentation.EndSession(player, false, "transition");
			}
			else
			{
				tracker.ResetRoomTimer();
			}
		}

        public static GlobalParameters GlobalParameters;
		public static SessionParameters SessionParameters;
		public static bool inSession;
		// Number of accepted playback input commands; does not reset at room transitions.
		public static long PlaybackFrames;
		public static long PlaybackSpawns;
		public static long PlaybackDeaths;
		private static Player lastSpawnPlayer;
	}
}
