using System.Collections.Generic;
using System.Web.Script.Serialization;
using Celeste;

namespace Instrumentation
{
	// Observations only: this class does not move the player or award collectibles.
	public static class SearchPlaybackState
	{
		public static string Build(Player player, bool dead = false, string reason = null)
		{
			Level level = player.SceneAs<Level>();
			List<string> berries = new List<string>();
			List<string> following = new List<string>();
			if (level != null)
			{
				foreach (EntityID id in level.Session.Strawberries)
				{
					berries.Add(id.Level + ":" + id.ID);
				}
			}
			if (player.Leader != null)
			{
				foreach (var follower in player.Leader.Followers)
				{
					Strawberry berry = follower.Entity as Strawberry;
					if (berry != null)
					{
						following.Add(berry.ID.Level + ":" + berry.ID.ID);
					}
				}
			}
			berries.Sort(System.StringComparer.Ordinal);
			following.Sort(System.StringComparer.Ordinal);
			SearchModelState model = SearchModelState.Capture(player, level, berries, following, dead);
			Dictionary<string, object> state = new Dictionary<string, object>
			{
				{ "schema_version", "celeste-playback-state-v1" },
				{ "input_phase", reason == null ? "before_virtual_inputs" : "terminal" },
				{ "room", level == null ? "" : level.Session.Level },
				{ "area_id", level == null ? -1 : level.Session.Area.ID },
				{ "area_mode", level == null ? -1 : (int)level.Session.Area.Mode },
				{ "map_name", MapName(level) },
				{ "assist_protocol_version", 1 },
				{ "capabilities", new [] { "physical_input_passthrough", "explicit_takeover", "native_room_respawn", "current_scene_attach", "gui_reconnect" } },
				{ "assist_mode", Instrumentation.AssistMode },
				{ "input_bindings_supported", InputsManager.GameplayBindingsAvailable },
				{ "control_mode", InputsManager.PassThrough || !Instrumentation.inSession ? "manual" : "assist" },
				{ "human_input_mask", AssistInputState.Available ? (object)AssistInputState.HumanInputMask : null },
				{ "human_menu_input", AssistInputState.Available ? (object)AssistInputState.HumanMenuInput : null },
				{ "human_input_epoch", AssistInputState.HumanInputEpoch },
				{ "camera_x", level == null ? (object)null : level.Camera.Position.X },
				{ "camera_y", level == null ? (object)null : level.Camera.Position.Y },
				{ "level_bounds", level == null ? null : new Dictionary<string, int> { { "x", level.Bounds.X }, { "y", level.Bounds.Y }, { "width", level.Bounds.Width }, { "height", level.Bounds.Height } } },
				{ "x", player.Position.X },
				{ "y", player.Position.Y },
				{ "vx", player.Speed.X },
				{ "vy", player.Speed.Y },
				{ "dashes", player.Dashes },
				{ "stamina", player.Stamina },
				{ "on_ground", level != null && player.OnGround(1) },
				{ "on_safe_ground", player.OnSafeGround },
				{ "state", player.StateMachine.State },
				{ "dead", dead || player.Dead },
				{ "berries", berries },
				{ "following", following },
				{ "frame", Instrumentation.PlaybackFrames },
				{ "engine_frame", Monocle.Engine.FrameCounter },
				{ "delta_time", Monocle.Engine.DeltaTime },
				{ "raw_delta_time", Monocle.Engine.RawDeltaTime },
				{ "previous_round_trip_ms", SearchPlaybackController.LastRoundTripMilliseconds },
				{ "max_round_trip_ms", SearchPlaybackController.MaxRoundTripMilliseconds },
				{ "spawn_id", Instrumentation.PlaybackSpawns },
				{ "deaths", Instrumentation.PlaybackDeaths },
				{ "reconnect_attempts", SearchReconnect.Attempts },
				{ "reconnects", SearchReconnect.Successes },
				{ "respawn_x", level != null && level.Session.RespawnPoint.HasValue ? (object)level.Session.RespawnPoint.Value.X : null },
				{ "respawn_y", level != null && level.Session.RespawnPoint.HasValue ? (object)level.Session.RespawnPoint.Value.Y : null },
				{ "control_event", SearchPlaybackController.LastControlEvent },
				{ "model_state", model.Data },
				{ "room_entities", SearchEnvironmentState.Capture(level) },
				{ "model_state_unavailable", model.Unavailable },
				{ "model_state_approximate", model.Approximate },
				{ "runtime_timers_seconds", model.TimersSeconds },
				{ "state_fingerprint", model.Fingerprint() },
				{ "fingerprint_scope", "model-anchor-v1; excludes frame, entity coroutines and RNG" }
			};
			if (reason != null)
			{
				state.Add("reason", reason);
			}
			return Serializer.Serialize(state);
		}

		private static readonly JavaScriptSerializer Serializer = new JavaScriptSerializer();

		private static string MapName(Level level)
		{
			if (level == null) return null;
			AreaData area = AreaData.Get(level.Session.Area);
			int mode = (int)level.Session.Area.Mode;
			return area != null && area.Mode != null && mode >= 0 && mode < area.Mode.Length && area.Mode[mode] != null ? area.Mode[mode].Path : null;
		}
	}
}
