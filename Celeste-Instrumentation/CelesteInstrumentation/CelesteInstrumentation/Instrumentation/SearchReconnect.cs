using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using Celeste;

namespace Instrumentation
{
	// Polling this coordinator never waits on a socket. Explicit detaches cancel
	// retries; only lost connections automatically resume the unchanged scene.
	public static class SearchReconnect
	{
		private static bool enabled;
		private static DateTime nextAttempt;
		private static Task<TcpClient> pending;
		private static readonly SearchReconnectSignal LocalSignal = new SearchReconnectSignal(AppDomain.CurrentDomain.BaseDirectory, DateTime.UtcNow);
		public static long Attempts { get; private set; }
		public static long Successes { get; private set; }

		public static void PollLocalRequest(bool userDetachHeld)
		{
			if (!Instrumentation.AssistMode || !SearchPlaybackController.LiveMode) return;
			DateTime now = DateTime.UtcNow;
			if (userDetachHeld) LocalSignal.UserDetached(now);
			if (LocalSignal.Poll(Instrumentation.inSession, userDetachHeld, now)) Request();
		}

		public static void Request()
		{
			enabled = true;
			nextAttempt = DateTime.UtcNow;
		}
		public static void Disable()
		{
			enabled = false;
			if (pending != null)
			{
				pending.ContinueWith(task => { if (task.Status == TaskStatus.RanToCompletion && task.Result != null) task.Result.Close(); });
				pending = null;
			}
		}
		public static void Poll(Player player)
		{
			if (!enabled) return;
			if (pending != null)
			{
				if (!pending.IsCompleted) return;
				TcpClient client = pending.Result;
				pending = null;
				if (client != null)
				{
					Socket.AdoptLiveConnection(client);
					enabled = false;
					Successes++;
					Instrumentation.ResumePlayback(player);
					return;
				}
				nextAttempt = DateTime.UtcNow.AddSeconds(1);
			}
			if (player == null || player.Dead || player.SceneAs<Level>() == null || DateTime.UtcNow < nextAttempt) return;
			// The only game-state access is here, on the game thread.
			string snapshot = SearchPlaybackState.Build(player);
			Attempts++;
			pending = Task.Run(() => {
				try { return Socket.OpenLiveConnection(snapshot); }
				catch (Exception exception) when (Instrumentation.IsPlaybackConnectionError(exception)) { return null; }
			});
		}
	}
}
