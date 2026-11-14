using System;
using System.Collections.Generic;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Web.Script.Serialization;
using Celeste;
using Microsoft.Xna.Framework;
using Monocle;

namespace Instrumentation
{
	// A read-only anchor for the independent model, never an engine savestate.
	public sealed class SearchModelState
	{
		public readonly SortedDictionary<string, object> Data = new SortedDictionary<string, object>(StringComparer.Ordinal);
		public readonly SortedDictionary<string, object> TimersSeconds = new SortedDictionary<string, object>(StringComparer.Ordinal);
		public readonly List<string> Unavailable = new List<string>();
		public readonly List<string> Approximate = new List<string>();

		public static SearchModelState Capture(Player player, Level level, List<string> berries, List<string> following, bool dead)
		{
			SearchModelState result = new SearchModelState();
			var data = result.Data;
			data["room"] = level == null ? "" : level.Session.Level;
			data["x"] = player.X;
			data["y"] = player.Y;
			data["vx"] = player.Speed.X;
			data["vy"] = player.Speed.Y;
			data["dashes"] = player.Dashes;
			data["stamina"] = player.Stamina;
			data["on_ground"] = level != null && player.OnGround(1);
			data["dead"] = dead || player.Dead;
			data["berries"] = berries;
			data["following"] = following;
			data["frame"] = Instrumentation.PlaybackFrames;
			data["facing"] = (int)player.Facing;
			data["state"] = player.StateMachine.State;
			data["prev_input"] = (int)InputsManager.PlaybackInput;
			data["ducking"] = player.Ducking;
			data["dash_dx"] = player.DashDir.X;
			data["dash_dy"] = player.DashDir.Y;
			data["lift_vx"] = player.LiftSpeed.X;
			data["lift_vy"] = player.LiftSpeed.Y;
			result.Vector(player, "movementCounter", "rem_x", "rem_y");
			result.Vector(player, "beforeDashSpeed", "dash_before_x", null);
			result.Scalar(player, "varJumpSpeed", "var_speed");
			result.Scalar(player, "forceMoveX", "force_x");
			result.Scalar(player, "dashStartedOnGround", "dash_started_ground");
			result.Scalar(player, "maxFall", "max_fall");
			result.Scalar(player, "wallSpeedRetained", "wall_retained");
			result.Scalar(player, "hopWaitX", "hop_wait_x");
			result.Scalar(player, "hopWaitXSpeed", "hop_wait_x_speed");
			result.Timer(player, "dashAttackTimer", "dash_attack");
			result.Timer(player, "wallSpeedRetentionTimer", "wall_retention");
			result.Timer(player, "jumpGraceTimer", "grace");
			result.Timer(player, "varJumpTimer", "var_jump");
			result.Timer(player, "dashCooldownTimer", "dash_cooldown");
			result.Timer(player, "dashRefillCooldownTimer", "refill_cooldown");
			result.Timer(player, "forceMoveXTimer", "force_frames");
			result.Timer(player, "wallSlideTimer", "wall_slide");
			result.Timer(player, "climbNoMoveTimer", "climb_wait");
			result.Timer(player, "liftSpeedTimer", "lift_timer");
			result.Button(Input.Jump, "jump_buffer");
			result.Button(Input.Dash, "dash_buffer");
			result.TimersSeconds["AutoJumpTimer"] = player.AutoJumpTimer;
			// AutoJump and its timer are independent engine fields. A true flag
			// with a zero timer persists until an actual gameplay reset clears it.
			data["auto_jump_active"] = player.AutoJump;
			data["auto_jump"] = TimerFrames(player.AutoJumpTimer);
			if (player.StateMachine.State == 2)
			{
				// Coroutine execution position is not recoverable from dashAttackTimer.
				result.Unavailable.Add("dash_frames");
			}
			else
			{
				data["dash_frames"] = 0;
			}

			List<string> order = new List<string>();
			Strawberry first = null;
			foreach (var follower in player.Leader.Followers)
			{
				Strawberry berry = follower.Entity as Strawberry;
				if (berry != null)
				{
					order.Add(berry.ID.Level + ":" + berry.ID.ID);
					if (first == null) first = berry;
				}
			}
			data["follow_order"] = order;
			if (first == null)
			{
				data["collect_delay"] = 0;
				data["collect_ground"] = 0;
			}
			else
			{
				result.TimersSeconds["firstFollower.DelayTimer"] = first.Follower.DelayTimer;
				data["collect_delay"] = TimerFrames(first.Follower.DelayTimer);
				result.Timer(first, "collectTimer", "collect_ground", true);
				result.Approximate.Add("collection: first-follower timers mapped to aggregate model counters");
			}
			// Do not fabricate fresh entity states during a mid-room re-anchor.
			result.Unavailable.AddRange(new[] { "mechanisms", "refill_timers", "fled", "broken" });
			result.Unavailable.Sort(StringComparer.Ordinal);
			result.Approximate.Sort(StringComparer.Ordinal);
			return result;
		}

		public string Fingerprint()
		{
			var stable = new SortedDictionary<string, object>(Data, StringComparer.Ordinal);
			stable.Remove("frame");
			stable["timers_seconds"] = TimersSeconds;
			stable["unavailable"] = Unavailable;
			byte[] bytes = Encoding.UTF8.GetBytes(new JavaScriptSerializer().Serialize(stable));
			using (SHA256 algorithm = SHA256.Create())
			{
				return BitConverter.ToString(algorithm.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
			}
		}

		public static int TimerFrames(float seconds, bool signed = false)
		{
			int frames = (int)Math.Ceiling(seconds * 60.0 - 0.0001);
			return signed ? frames : Math.Max(0, frames);
		}

		private void Scalar(object target, string field, string key)
		{
			object value;
			if (Read(target, field, out value)) Data[key] = value;
			else Unavailable.Add(key);
		}

		private void Vector(object target, string field, string x, string y)
		{
			object value;
			if (Read(target, field, out value) && value is Vector2)
			{
				Vector2 vector = (Vector2)value;
				Data[x] = vector.X;
				if (y != null) Data[y] = vector.Y;
			}
			else
			{
				Unavailable.Add(x);
				if (y != null) Unavailable.Add(y);
			}
		}

		private void Timer(object target, string field, string key, bool signed = false)
		{
			object value;
			if (Read(target, field, out value) && value is float)
			{
				float seconds = (float)value;
				TimersSeconds[field] = seconds;
				Data[key] = TimerFrames(seconds, signed);
			}
			else Unavailable.Add(key);
		}

		private void Button(VirtualButton button, string key)
		{
			object counter, consumed;
			if (Read(button, "bufferCounter", out counter) && Read(button, "consumed", out consumed))
			{
				TimersSeconds[key] = counter;
				Data[key] = (bool)consumed ? 0 : TimerFrames((float)counter);
			}
			else Unavailable.Add(key);
		}

		internal static bool Read(object target, string name, out object value)
		{
			value = null;
			if (target == null) return false;
			string key = target.GetType().FullName + ":" + name;
			FieldInfo field;
			if (!Fields.TryGetValue(key, out field))
			{
				for (Type type = target.GetType(); type != null && field == null; type = type.BaseType)
				{
					field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly);
				}
				Fields[key] = field;
			}
			if (field == null) return false;
			value = field.GetValue(target);
			return true;
		}

		private static readonly Dictionary<string, FieldInfo> Fields = new Dictionary<string, FieldInfo>();
	}
}
