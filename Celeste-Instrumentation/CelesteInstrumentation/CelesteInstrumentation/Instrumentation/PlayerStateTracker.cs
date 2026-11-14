using System;
using System.Collections.Generic;
using Celeste;
using Microsoft.Xna.Framework;
using Monocle;
using System.Web.Script.Serialization;

namespace Instrumentation
{
	public class PlayerStateTracker : Component
	{
		public override void Update()
		{
			if (SearchPlaybackController.Enabled)
			{
				// SearchMode observes before virtual input/scene update, not midway
				// through Player.Update after its StateMachine has changed velocity.
				return;
			}
			Level level = base.Entity.SceneAs<Level>();

			if (this.PlayerState.SecondsElapsed >= Instrumentation.SessionParameters.TimeoutSeconds)
			{
				Instrumentation.EndSession(this.Player, false, "timeout");
				return;
			}

			this.UpdatePlayerState(level);

			if (Instrumentation.inSession)
			{
				this.SendMapInfoIfNeeded(level);
				InputsManager.ScheduleKeys(InputsManager.ByteToKeys(Socket.SendAndReceiveData(this.PlayerState.Serialize())[0]));
			}

			base.Update();
			this.FramesElapsed += 1f;
			this.PlayerState.SecondsElapsed = this.FramesElapsed / 60f;
		}

		private void UpdatePlayerState(Level level)
		{
			this.PlayerState.CanDash = (this.Player.Dashes != 0);
			this.PlayerState.XVelocity = this.Player.Speed.X;
			this.PlayerState.YVelocity = this.Player.Speed.Y;
			this.PlayerState.OnGround = this.Player.OnGround(1);
			this.PlayerState.Stamina = this.Player.Stamina;
			this.PlayerState.DistanceToObjective = Vector2.Distance(this.Player.Position, this.objective);
			this.PlayerState.AngleToObjective = (this.objective - this.Player.Position).Angle() + (float)Math.PI;
			this.PlayerState.LevelDiagonalLength = (float)Math.Pow((double)(level.Bounds.Width * level.Bounds.Width + level.Bounds.Height * level.Bounds.Height), 0.5);
			this.PlayerState.SpeedMagnitude = this.Player.Speed.Length();
			this.PlayerState.ProgressToObjective = this.PlayerState.InitialDistanceToObjective - this.PlayerState.DistanceToObjective;
			this.PlayerState.HorizontalProgress = this.Player.Position.X - this.initialPosition.X;
			this.PlayerState.VerticalProgress = this.initialPosition.Y - this.Player.Position.Y;

			if (this.PlayerState.DistanceToObjective < this.PlayerState.BestDistanceToObjective - 2f)
			{
				this.PlayerState.BestDistanceToObjective = this.PlayerState.DistanceToObjective;
				this.FramesSinceProgress = 0;
			}
			else
			{
				this.FramesSinceProgress += 1;
			}
			this.PlayerState.IsStuck = this.FramesSinceProgress > 90 && this.PlayerState.SecondsElapsed > 1f;

			this.PlayerState.NormalizedX = SafeNormalize(this.Player.Position.X - level.Bounds.Left, level.Bounds.Width);
			this.PlayerState.NormalizedY = SafeNormalize(this.Player.Position.Y - level.Bounds.Top, level.Bounds.Height);
			this.PlayerState.NormalizedObjectiveX = SafeNormalize(this.objective.X - level.Bounds.Left, level.Bounds.Width);
			this.PlayerState.NormalizedObjectiveY = SafeNormalize(this.objective.Y - level.Bounds.Top, level.Bounds.Height);
			this.PlayerState.DeltaXToObjective = SafeNormalize(this.objective.X - this.Player.Position.X, level.Bounds.Width);
			this.PlayerState.DeltaYToObjective = SafeNormalize(this.objective.Y - this.Player.Position.Y, level.Bounds.Height);

			for (int i = 0; i < this.PlayerState.Distances.Length; i++)
			{
				ValueTuple<int, int> valueTuple = this.Raycast(this.RaycastDirections[i], level);
				this.PlayerState.Distances[i] = valueTuple.Item1;
				this.PlayerState.DetectedTypes[i] = valueTuple.Item2;
			}

			this.UpdateLocalGrid(level);
		}

		private void SendMapInfoIfNeeded(Level level)
		{
			string roomKey = level.Session.Level + ":" + level.Bounds.X + ":" + level.Bounds.Y + ":" + level.Bounds.Width + ":" + level.Bounds.Height;
			if (roomKey == this.LastRoomKey)
			{
				return;
			}
			this.LastRoomKey = roomKey;
			string mapSummary = MapSummaryBuilder.Build(level, base.Scene, this.Player, this.objective);
			this.LastRoutePlan = Socket.SendMapInfoAndReceivePlan(mapSummary);
		}

		private static float SafeNormalize(float value, float divisor)
		{
			if (Math.Abs(divisor) < 0.0001f)
			{
				return 0f;
			}
			return value / divisor;
		}

		private void UpdateLocalGrid(Level level)
		{
			int halfSize = PlayerState.LocalGridSize / 2;
			int index = 0;
			for (int y = -halfSize; y <= halfSize; y++)
			{
				for (int x = -halfSize; x <= halfSize; x++)
				{
					Vector2 sample = this.Player.Position + new Vector2(x * LocalGridCellPixels, y * LocalGridCellPixels);
					this.PlayerState.LocalGridTypes[index] = this.ClassifyPoint(sample, level);
					index++;
				}
			}
		}

		private int ClassifyPoint(Vector2 point, Level level)
		{
			if (!level.Bounds.Contains((int)point.X, (int)point.Y))
			{
				Vector2 direction = point - this.Player.Position;
				if (direction.LengthSquared() > 0.0001f && level.Session.MapData.CanTransitionTo(level, this.Player.Position + direction.SafeNormalize() * 8f))
				{
					return 1;
				}
				return 0;
			}
			if (base.Scene.CollideCheck<Solid>(point))
			{
				return 2;
			}
			if (base.Scene.CollideCheck<Spikes>(point))
			{
				return 3;
			}
			return 0;
		}

		public override void DebugRender(Camera camera)
		{
			if (Instrumentation.GlobalParameters.DebugRendering || Engine.Commands.Open)
			{
				for (int i = 0; i < this.PlayerState.Distances.Length; i++)
				{
					Draw.Line(base.Entity.Position, base.Entity.Position + this.RaycastDirections[i] * (float)this.PlayerState.Distances[i], PlayerStateTracker.RayColors[this.PlayerState.DetectedTypes[i]]);
				}
				string text = ((int)this.Player.Position.X).ToString() + ", " + ((int)this.Player.Position.Y).ToString();
				Draw.Text(Draw.DefaultFont, text, this.Player.Position, Color.White);
			}
			if (Instrumentation.GlobalParameters.RouteRendering || Engine.Commands.Open)
			{
				this.DrawRoutePlan();
			}
			base.DebugRender(camera);
		}

		private void DrawRoutePlan()
		{
			List<RouteWaypoint> waypoints = this.GetRouteWaypoints();
			if (waypoints.Count == 0)
			{
				return;
			}

			int nextIndex = this.GetNextRouteWaypointIndex(waypoints);
			Vector2 previous = this.Player.Position;
			for (int i = 0; i < waypoints.Count; i++)
			{
				Vector2 current = waypoints[i].Position;
				Color lineColor = i < nextIndex ? Color.DarkGray : Color.Cyan;
				Draw.Line(previous, current, lineColor);
				previous = current;
			}

			for (int i = 0; i < waypoints.Count; i++)
			{
				RouteWaypoint waypoint = waypoints[i];
				Color color = i == nextIndex ? Color.Yellow : Color.Cyan;
				this.DrawWaypointMarker(waypoint.Position, color);
				Draw.Text(
					Draw.DefaultFont,
					(i + 1).ToString() + " " + waypoint.Action,
					waypoint.Position + new Vector2(5f, -18f),
					color
				);
			}

			if (nextIndex >= 0 && nextIndex < waypoints.Count)
			{
				RouteWaypoint next = waypoints[nextIndex];
				Draw.Line(this.Player.Position, next.Position, Color.Yellow);
				this.DrawWaypointMarker(next.Position, Color.Yellow);
			}
		}

		private void DrawWaypointMarker(Vector2 position, Color color)
		{
			float size = 5f;
			Draw.Line(position + new Vector2(-size, -size), position + new Vector2(size, -size), color);
			Draw.Line(position + new Vector2(size, -size), position + new Vector2(size, size), color);
			Draw.Line(position + new Vector2(size, size), position + new Vector2(-size, size), color);
			Draw.Line(position + new Vector2(-size, size), position + new Vector2(-size, -size), color);
			Draw.Line(position + new Vector2(-size - 2f, 0f), position + new Vector2(size + 2f, 0f), color);
			Draw.Line(position + new Vector2(0f, -size - 2f), position + new Vector2(0f, size + 2f), color);
		}

		private int GetNextRouteWaypointIndex(List<RouteWaypoint> waypoints)
		{
			for (int i = 0; i < waypoints.Count; i++)
			{
				if (Vector2.Distance(this.Player.Position, waypoints[i].Position) > waypoints[i].Radius)
				{
					return i;
				}
			}
			return waypoints.Count - 1;
		}

		private List<RouteWaypoint> GetRouteWaypoints()
		{
			if (string.IsNullOrEmpty(this.LastRoutePlan))
			{
				return EmptyRouteWaypoints;
			}
			if (this.LastRoutePlan == this.LastParsedRoutePlan && this.LastParsedWaypoints != null)
			{
				return this.LastParsedWaypoints;
			}

			this.LastParsedRoutePlan = this.LastRoutePlan;
			this.LastParsedWaypoints = this.ParseRouteWaypoints(this.LastRoutePlan);
			return this.LastParsedWaypoints;
		}

		private List<RouteWaypoint> ParseRouteWaypoints(string routePlan)
		{
			List<RouteWaypoint> result = new List<RouteWaypoint>();
			try
			{
				Dictionary<string, object> root = RoutePlanJsonSerializer.DeserializeObject(routePlan) as Dictionary<string, object>;
				if (root == null || !root.ContainsKey("waypoints"))
				{
					return result;
				}
				object[] waypoints = root["waypoints"] as object[];
				if (waypoints == null)
				{
					return result;
				}

				foreach (object waypointObject in waypoints)
				{
					Dictionary<string, object> waypoint = waypointObject as Dictionary<string, object>;
					if (waypoint == null)
					{
						continue;
					}

					float x = GetFloat(waypoint, "x", this.Player.Position.X);
					float y = GetFloat(waypoint, "y", this.Player.Position.Y);
					float radius = GetFloat(waypoint, "radius", 24f);
					string action = waypoint.ContainsKey("action") && waypoint["action"] != null ? waypoint["action"].ToString() : "route";
					result.Add(new RouteWaypoint(new Vector2(x, y), radius, action));
				}
			}
			catch
			{
				result.Clear();
			}
			return result;
		}

		private static float GetFloat(Dictionary<string, object> values, string key, float fallback)
		{
			if (!values.ContainsKey(key) || values[key] == null)
			{
				return fallback;
			}
			try
			{
				return Convert.ToSingle(values[key]);
			}
			catch
			{
				return fallback;
			}
		}

		public PlayerStateTracker(Player player) : base(true, true)
		{
			this.PlayerState = new PlayerState();
			this.objective = new Vector2(Instrumentation.SessionParameters.ObjectiveXCoordinate, Instrumentation.SessionParameters.ObjectiveYCoordinate);
			this.PlayerState.InitialDistanceToObjective = Vector2.Distance(player.Position, this.objective);
			this.PlayerState.BestDistanceToObjective = this.PlayerState.InitialDistanceToObjective;
			this.Player = player;
			this.initialPosition = player.Position;
			this.FramesElapsed = 0f;
			this.FramesSinceProgress = 0;
		}

		public void ResetRoomTimer()
		{
			this.FramesElapsed = 0f;
			this.FramesSinceProgress = 0;
			this.PlayerState.SecondsElapsed = 0f;
			this.PlayerState.IsStuck = false;
		}

		public ValueTuple<int, int> Raycast(Vector2 directionUnitVector, Level level)
		{
			int num = 1;
			Vector2 value = base.Entity.Position - Vector2.UnitY;
			for (;;)
			{
				Vector2 vector = value + directionUnitVector * (float)num;
				if (!level.Bounds.Contains((int)vector.X, (int)vector.Y))
				{
					break;
				}
				if (base.Scene.CollideCheck<Solid>(vector))
				{
					goto IL_93;
				}
				if (base.Scene.CollideCheck<Spikes>(vector))
				{
					goto IL_97;
				}
				num++;
			}
			int item;
			if (!level.Session.MapData.CanTransitionTo(level, value + directionUnitVector * (float)(num + 5)))
			{
				item = 0;
				goto IL_99;
			}
			item = 1;
			goto IL_99;
			IL_93:
			item = 2;
			goto IL_99;
			IL_97:
			item = 3;
			IL_99:
			return new ValueTuple<int, int>(num, item);
		}

		private static Vector2[] BuildRaycastDirections()
		{
			Vector2[] directions = new Vector2[PlayerState.RaycastCount];
			for (int i = 0; i < directions.Length; i++)
			{
				float angle = ((float)i / directions.Length) * (float)Math.PI * 2f;
				directions[i] = new Vector2((float)Math.Cos(angle), (float)Math.Sin(angle)).SafeNormalize();
			}
			return directions;
		}

		public Vector2[] RaycastDirections = BuildRaycastDirections();

		public static Color[] RayColors = new Color[]
		{
			Color.Purple,
			Color.Green,
			Color.Orange,
			Color.Red
		};

		private class RouteWaypoint
		{
			public RouteWaypoint(Vector2 position, float radius, string action)
			{
				this.Position = position;
				this.Radius = radius;
				this.Action = action;
			}

			public Vector2 Position;

			public float Radius;

			public string Action;
		}

		private static readonly JavaScriptSerializer RoutePlanJsonSerializer = new JavaScriptSerializer();

		private static readonly List<RouteWaypoint> EmptyRouteWaypoints = new List<RouteWaypoint>();

		public PlayerState PlayerState;

		public Player Player;

		public float FramesElapsed;

		public int FramesSinceProgress;

		private const int LocalGridCellPixels = 8;

		private Vector2 objective;

		private Vector2 initialPosition;

		private string LastRoomKey;

		private string LastRoutePlan;

		private string LastParsedRoutePlan;

		private List<RouteWaypoint> LastParsedWaypoints;
	}
}
