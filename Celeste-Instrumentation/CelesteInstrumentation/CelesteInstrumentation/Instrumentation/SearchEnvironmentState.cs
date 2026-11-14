using System;
using System.Collections.Generic;
using System.Collections;
using System.Reflection;
using Celeste;
using Microsoft.Xna.Framework;
using Monocle;

namespace Instrumentation
{
	// EntityData constructors provide identity before entities move. Never match by
	// nearest position: several maps deliberately overlap entities and nodes.
	public static class SearchEnvironmentState
	{
		private sealed class Record
		{
			public string ID, Room, Kind;
			public int EntityID;
			public float MapX, MapY;
			public WeakReference Entity;
			public Dictionary<string, object> Last;
		}
		private static readonly Dictionary<string, Record> Records = new Dictionary<string, Record>(StringComparer.Ordinal);
		private static readonly object Gate = new object();

		public static void Reset()
		{
			lock (Gate) Records.Clear();
		}

		public static void Register(Entity entity, EntityData data, Vector2 offset)
		{
			if (data == null || data.Level == null) return;
			string id = data.Level.Name + ":" + data.ID;
			lock (Gate)
			{
				Records[id] = new Record { ID = id, Room = data.Level.Name, Kind = data.Name,
					EntityID = data.ID, MapX = data.Position.X + offset.X, MapY = data.Position.Y + offset.Y,
					Entity = new WeakReference(entity) };
			}
		}

		public static Dictionary<string, object> Capture(Level level)
		{
			List<Dictionary<string, object>> entities = new List<Dictionary<string, object>>();
			List<string> doNotLoad = new List<string>();
			Dictionary<Entity, int> updateOrder = new Dictionary<Entity, int>();
			Dictionary<Entity, int> cassetteTrackerOrder = new Dictionary<Entity, int>();
			object playerUpdateOrder = null;
			object playerDepth = null, playerActualDepth = null;
			Dictionary<string, double> depthCounters = null;
			List<string> unavailable = new List<string> {
				"interpreted mechanism coroutine phase (raw iterator state is exported)",
				"dynamically created entities without EntityData except CassetteBlockManager"
			};
			if (level != null)
			{
				int index = 0;
				foreach (Entity entity in level.Entities)
				{
					updateOrder[entity] = index;
					if (entity.GetType().FullName == "Celeste.Player")
					{
						playerUpdateOrder = index;
						playerDepth = entity.Depth;
						object actual;
						if (SearchModelState.Read(entity, "actualDepth", out actual) && actual is double) playerActualDepth = actual;
					}
					index++;
				}
				index = 0;
				foreach (Entity entity in level.Tracker.GetEntities<CassetteBlock>()) cassetteTrackerOrder[entity] = index++;
				object rawDepthCounters;
				if (SearchModelState.Read(level, "actualDepthLookup", out rawDepthCounters) && rawDepthCounters is Dictionary<int, double>)
				{
					depthCounters = new Dictionary<string, double>(StringComparer.Ordinal);
					foreach (var pair in (Dictionary<int, double>)rawDepthCounters)
						depthCounters[pair.Key.ToString(System.Globalization.CultureInfo.InvariantCulture)] = pair.Value;
				}
				foreach (EntityID id in level.Session.DoNotLoad) doNotLoad.Add(id.Level + ":" + id.ID);
				lock (Gate)
				{
					foreach (Record record in Records.Values)
					{
						Entity entity = record.Entity.Target as Entity;
						bool present = entity != null && entity.Scene == level;
						if (record.Room != level.Session.Level && !present) continue;
						Dictionary<string, object> state = entity == null ? record.Last : Observe(record, entity, updateOrder, cassetteTrackerOrder);
						if (state == null) continue;
						state = new Dictionary<string, object>(state);
						state["present"] = present;
						state["retained_observation"] = entity == null;
						if (!present) state["update_order"] = null;
						record.Last = state;
						entities.Add(state);
					}
				}
			}
			if (playerUpdateOrder == null) unavailable.Add("player_update_order");
			if (playerDepth == null) unavailable.Add("player_depth");
			if (playerActualDepth == null) unavailable.Add("player_actual_depth");
			if (depthCounters == null) unavailable.Add("depth_counters");
			entities.Sort((a, b) => StringComparer.Ordinal.Compare((string)a["id"], (string)b["id"]));
			doNotLoad.Sort(StringComparer.Ordinal);
			return new Dictionary<string, object> {
				{ "schema_version", "celeste-room-entities-v1" },
				{ "entities", entities }, { "do_not_load", doNotLoad },
				{ "player_update_order", playerUpdateOrder },
				{ "player_depth", playerDepth }, { "player_actual_depth", playerActualDepth },
				{ "depth_counters", depthCounters },
				{ "cassette_manager", CaptureCassetteManager(level, updateOrder) },
				{ "coverage", "EntityData constructors: Strawberry, Refill, DashBlock, ZipMover, FallingBlock, CrumblePlatform, Spikes, Spring, Jumpthru, CassetteBlock; scene tracker: CassetteBlockManager" },
				{ "unavailable", unavailable }
			};
		}

		private static Dictionary<string, object> Observe(Record record, Entity entity, Dictionary<Entity, int> updateOrder, Dictionary<Entity, int> cassetteTrackerOrder)
		{
			var data = new Dictionary<string, object> {
				{ "id", record.ID }, { "room", record.Room }, { "entity_id", record.EntityID }, { "kind", record.Kind },
				{ "runtime_type", entity.GetType().FullName }, { "map_x", record.MapX }, { "map_y", record.MapY },
				{ "x", entity.X }, { "y", entity.Y }, { "collidable", entity.Collidable }, { "active", entity.Active },
				{ "left", entity.Collider == null ? entity.X : entity.Left }, { "top", entity.Collider == null ? entity.Y : entity.Top },
				{ "width", entity.Collider == null ? 0f : entity.Width }, { "height", entity.Collider == null ? 0f : entity.Height }
			};
			List<string> unavailable = new List<string>();
			CaptureDepth(entity, data, unavailable);
			int order;
			data["update_order"] = updateOrder.TryGetValue(entity, out order) ? (object)order : null;
			if (data["update_order"] == null) unavailable.Add("update_order");
			CassetteBlock cassette = entity as CassetteBlock;
			if (cassette != null)
			{
				int trackerOrder;
				data["tracker_order"] = cassetteTrackerOrder.TryGetValue(entity, out trackerOrder) ? (object)trackerOrder : null;
				if (data["tracker_order"] == null) unavailable.Add("tracker_order");
				Field(cassette, data, unavailable, "Index", "index");
				Field(cassette, data, unavailable, "Tempo", "tempo");
				Field(cassette, data, unavailable, "Activated", "activated");
				Field(cassette, data, unavailable, "Mode", "mode");
				Field(cassette, data, unavailable, "blockHeight", "block_height");
				Field(cassette, data, unavailable, "groupLeader", "group_leader");
				object group;
				if (SearchModelState.Read(cassette, "group", out group) && group is IEnumerable)
				{
					List<string> ids = new List<string>();
					foreach (object member in (IEnumerable)group)
					{
						CassetteBlock block = member as CassetteBlock;
						if (block == null) { unavailable.Add("group_ids"); break; }
						ids.Add(block.ID.Level + ":" + block.ID.ID);
					}
					// Preserve native group order: a preceding block's wiggle can
					// move the player even when a later member remains obstructed.
					if (!unavailable.Contains("group_ids")) data["group_ids"] = ids;
				}
				else unavailable.Add("group_ids");
			}
			if (entity is Refill)
			{
				Field(entity, data, unavailable, "respawnTimer", "respawn_timer");
				Field(entity, data, unavailable, "oneUse", "one_use");
				Field(entity, data, unavailable, "twoDashes", "two_dashes");
			}
			Strawberry berry = entity as Strawberry;
			if (berry != null)
			{
				data["winged"] = berry.Winged;
				data["golden"] = berry.Golden;
				data["has_leader"] = berry.Follower != null && berry.Follower.Leader != null;
				data["follow_delay_timer"] = berry.Follower == null ? 0f : berry.Follower.DelayTimer;
				Field(entity, data, unavailable, "collected", "collected");
				Field(entity, data, unavailable, "collectTimer", "collect_timer");
				Field(entity, data, unavailable, "flyingAway", "flying_away");
				Field(entity, data, unavailable, "flapSpeed", "flap_speed");
				data["waiting_on_seeds"] = berry.WaitingOnSeeds;
			}
			if (entity is DashBlock)
			{
				Field(entity, data, unavailable, "permanent", "permanent");
				Field(entity, data, unavailable, "canDash", "can_dash");
			}
			if (entity is ZipMover)
			{
				Field(entity, data, unavailable, "percent", "percent");
				Vector(entity, data, unavailable, "target", "target_x", "target_y");
			}
			FallingBlock falling = entity as FallingBlock;
			if (falling != null)
			{
				data["triggered"] = falling.Triggered;
				data["fall_delay"] = falling.FallDelay;
				data["has_started_falling"] = falling.HasStartedFalling;
			}
			Platform platform = entity as Platform;
			if (platform != null)
			{
				Vector(entity, data, unavailable, "movementCounter", "rem_x", "rem_y");
				data["lift_vx"] = platform.LiftSpeed.X;
				data["lift_vy"] = platform.LiftSpeed.Y;
			}
			if (entity is ZipMover || entity is FallingBlock || entity is CrumblePlatform || entity is Refill || (berry != null && berry.Winged))
			{
				data["coroutines"] = Coroutines(entity, unavailable);
				unavailable.Add("coroutine_phase");
			}
			data["unavailable"] = unavailable;
			return data;
		}

		private static Dictionary<string, object> CaptureCassetteManager(Level level, Dictionary<Entity, int> updateOrder)
		{
			CassetteBlockManager manager = level == null ? null : level.Tracker.GetEntity<CassetteBlockManager>();
			List<string> unavailable = new List<string>();
			var data = new Dictionary<string, object> { { "present", manager != null }, { "unavailable", unavailable } };
			if (level == null) unavailable.Add("current_level");
			if (manager == null) return data;
			CaptureDepth(manager, data, unavailable);
			data["active"] = manager.Active;
			int order;
			data["update_order"] = updateOrder.TryGetValue(manager, out order) ? (object)order : null;
			if (data["update_order"] == null) unavailable.Add("update_order");
			Field(manager, data, unavailable, "currentIndex", "current_index");
			Field(manager, data, unavailable, "beatTimer", "beat_timer");
			Field(manager, data, unavailable, "beatIndex", "beat_index");
			Field(manager, data, unavailable, "tempoMult", "tempo_mult");
			Field(manager, data, unavailable, "leadBeats", "lead_beats");
			Field(manager, data, unavailable, "maxBeat", "max_beat");
			Field(manager, data, unavailable, "isLevelMusic", "is_level_music");
			Field(manager, data, unavailable, "beatIndexOffset", "beat_index_offset");
			object isLevelMusic, sfx;
			if (SearchModelState.Read(manager, "isLevelMusic", out isLevelMusic) && isLevelMusic is bool
				&& SearchModelState.Read(manager, "sfx", out sfx))
				data["music_ready"] = (bool)isLevelMusic || sfx != null;
			else unavailable.Add("music_ready");
			return data;
		}

		private static void CaptureDepth(Entity entity, Dictionary<string, object> data, List<string> unavailable)
		{
			data["depth"] = entity.Depth;
			object actual;
			if (SearchModelState.Read(entity, "actualDepth", out actual) && actual is double) data["actual_depth"] = actual;
			else unavailable.Add("actual_depth");
		}

		private static List<Dictionary<string, object>> Coroutines(Entity entity, List<string> unavailable)
		{
			var result = new List<Dictionary<string, object>>();
			foreach (Component component in entity.Components)
			{
				Coroutine coroutine = component as Coroutine;
				if (coroutine == null) continue;
				object raw;
				if (!SearchModelState.Read(coroutine, "enumerators", out raw) || !(raw is IEnumerable))
				{
					unavailable.Add("coroutine_stack");
					continue;
				}
				var stack = new List<Dictionary<string, object>>();
				bool mainSequence = false;
				foreach (object iterator in (IEnumerable)raw)
				{
					string name = iterator.GetType().FullName;
					mainSequence |= name.Contains("<Sequence>") || name.Contains("<FlyAwayRoutine>") || name.Contains("<RefillRoutine>");
					var fields = new SortedDictionary<string, object>(StringComparer.Ordinal);
					foreach (FieldInfo field in iterator.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
					{
						Type type = field.FieldType;
						object value = field.GetValue(iterator);
						if (type.IsPrimitive || type.IsEnum) fields[field.Name] = type.IsEnum ? Convert.ToInt32(value) : value;
						else if (value is Vector2) fields[field.Name] = new[] { ((Vector2)value).X, ((Vector2)value).Y };
					}
					stack.Add(new Dictionary<string, object> { { "iterator_type", name }, { "fields", fields } });
				}
				if (!mainSequence) continue; // Omit decorative tile/outline animation coroutines.
				var observation = new Dictionary<string, object> { { "active", coroutine.Active }, { "finished", coroutine.Finished }, { "stack", stack } };
				Field(coroutine, observation, unavailable, "waitTimer", "wait_timer");
				result.Add(observation);
			}
			return result;
		}

		private static void Field(object entity, Dictionary<string, object> data, List<string> unavailable, string field, string key)
		{
			object value;
			if (SearchModelState.Read(entity, field, out value))
				data[key] = value != null && value.GetType().IsEnum ? Convert.ToInt32(value) : value;
			else unavailable.Add(key);
		}
		private static void Vector(object entity, Dictionary<string, object> data, List<string> unavailable, string field, string x, string y)
		{
			object value;
			if (SearchModelState.Read(entity, field, out value) && value is Vector2)
			{
				data[x] = ((Vector2)value).X;
				data[y] = ((Vector2)value).Y;
			}
			else { unavailable.Add(x); unavailable.Add(y); }
		}
	}
}
