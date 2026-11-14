using System;
using System.Collections.Generic;
using System.Text;
using Celeste;
using Microsoft.Xna.Framework;
using Monocle;

namespace Instrumentation
{
	public static class MapSummaryBuilder
	{
		private const int GridCellPixels = 16;
		private const int MaxGridWidth = 96;
		private const int MaxGridHeight = 54;
		private const int MaxEntities = 160;

		public static string Build(Level level, Scene scene, Player player, Vector2 objective)
		{
			StringBuilder builder = new StringBuilder();
			Rectangle bounds = level.Bounds;
			int gridWidth = Math.Min(MaxGridWidth, Math.Max(1, (int)Math.Ceiling(bounds.Width / (float)GridCellPixels)));
			int gridHeight = Math.Min(MaxGridHeight, Math.Max(1, (int)Math.Ceiling(bounds.Height / (float)GridCellPixels)));
			float stepX = bounds.Width / (float)gridWidth;
			float stepY = bounds.Height / (float)gridHeight;

			builder.Append("{");
			AppendJsonProperty(builder, "schema_version", "celeste-map-summary-v1");
			builder.Append(",");
			AppendJsonProperty(builder, "level", level.Session.Level ?? string.Empty);
			builder.Append(",");
			builder.Append("\"bounds\":{");
			AppendJsonProperty(builder, "x", bounds.X);
			builder.Append(",");
			AppendJsonProperty(builder, "y", bounds.Y);
			builder.Append(",");
			AppendJsonProperty(builder, "width", bounds.Width);
			builder.Append(",");
			AppendJsonProperty(builder, "height", bounds.Height);
			builder.Append("},");
			builder.Append("\"player_start\":{");
			AppendJsonProperty(builder, "x", player.Position.X);
			builder.Append(",");
			AppendJsonProperty(builder, "y", player.Position.Y);
			builder.Append("},");
			builder.Append("\"objective\":{");
			AppendJsonProperty(builder, "x", objective.X);
			builder.Append(",");
			AppendJsonProperty(builder, "y", objective.Y);
			builder.Append("},");
			builder.Append("\"grid\":{");
			AppendJsonProperty(builder, "cell_pixels", GridCellPixels);
			builder.Append(",");
			AppendJsonProperty(builder, "width", gridWidth);
			builder.Append(",");
			AppendJsonProperty(builder, "height", gridHeight);
			builder.Append(",");
			AppendJsonProperty(builder, "encoding", "0 air, 1 transition, 2 solid, 3 spikes");
			builder.Append(",\"rows\":[");
			for (int y = 0; y < gridHeight; y++)
			{
				if (y > 0)
				{
					builder.Append(",");
				}
				builder.Append("\"");
				for (int x = 0; x < gridWidth; x++)
				{
					Vector2 sample = new Vector2(bounds.Left + (x + 0.5f) * stepX, bounds.Top + (y + 0.5f) * stepY);
					builder.Append(ClassifyPoint(sample, level, scene));
				}
				builder.Append("\"");
			}
			builder.Append("]},");
			builder.Append("\"entities\":");
			AppendEntities(builder, level);
			builder.Append("}");
			return builder.ToString();
		}

		private static int ClassifyPoint(Vector2 point, Level level, Scene scene)
		{
			if (!level.Bounds.Contains((int)point.X, (int)point.Y))
			{
				return 1;
			}
			if (scene.CollideCheck<Solid>(point))
			{
				return 2;
			}
			if (scene.CollideCheck<Spikes>(point))
			{
				return 3;
			}
			return 0;
		}

		private static void AppendEntities(StringBuilder builder, Level level)
		{
			builder.Append("[");
			int count = 0;
			try
			{
				foreach (Entity entity in level.Entities)
				{
					if (entity == null)
					{
						continue;
					}
					if (count >= MaxEntities)
					{
						break;
					}
					if (count > 0)
					{
						builder.Append(",");
					}
					string typeName = entity.GetType().Name;
					builder.Append("{");
					AppendJsonProperty(builder, "type", typeName);
					builder.Append(",");
					AppendJsonProperty(builder, "semantic", SemanticType(typeName));
					builder.Append(",");
					AppendJsonProperty(builder, "x", entity.Position.X);
					builder.Append(",");
					AppendJsonProperty(builder, "y", entity.Position.Y);
					builder.Append("}");
					count++;
				}
			}
			catch
			{
			}
			builder.Append("]");
		}

		private static string SemanticType(string typeName)
		{
			string name = typeName.ToLowerInvariant();
			if (name.Contains("spike") || name.Contains("spinner") || name.Contains("blade") || name.Contains("hazard") || name.Contains("dust"))
			{
				return "hazard";
			}
			if (name.Contains("refill"))
			{
				return "refill";
			}
			if (name.Contains("spring"))
			{
				return "spring";
			}
			if (name.Contains("booster"))
			{
				return "booster";
			}
			if (name.Contains("dream"))
			{
				return "dream_block";
			}
			if (name.Contains("zip") || name.Contains("moveblock") || name.Contains("falling") || name.Contains("crush"))
			{
				return "moving_platform";
			}
			if (name.Contains("platform") || name.Contains("solid"))
			{
				return "platform";
			}
			if (name.Contains("strawberry") || name.Contains("cassette"))
			{
				return "collectible";
			}
			if (name.Contains("key") || name.Contains("lock"))
			{
				return "key_or_lock";
			}
			if (name.Contains("trigger"))
			{
				return "trigger";
			}
			if (name.Contains("switch"))
			{
				return "switch";
			}
			return "other";
		}

		private static void AppendJsonProperty(StringBuilder builder, string name, string value)
		{
			AppendJsonString(builder, name);
			builder.Append(":");
			AppendJsonString(builder, value);
		}

		private static void AppendJsonProperty(StringBuilder builder, string name, int value)
		{
			AppendJsonString(builder, name);
			builder.Append(":");
			builder.Append(value);
		}

		private static void AppendJsonProperty(StringBuilder builder, string name, float value)
		{
			AppendJsonString(builder, name);
			builder.Append(":");
			builder.Append(value.ToString(System.Globalization.CultureInfo.InvariantCulture));
		}

		private static void AppendJsonString(StringBuilder builder, string value)
		{
			builder.Append("\"");
			foreach (char c in value)
			{
				switch (c)
				{
					case '\\':
						builder.Append("\\\\");
						break;
					case '"':
						builder.Append("\\\"");
						break;
					case '\n':
						builder.Append("\\n");
						break;
					case '\r':
						builder.Append("\\r");
						break;
					case '\t':
						builder.Append("\\t");
						break;
					default:
						builder.Append(c);
						break;
				}
			}
			builder.Append("\"");
		}
	}
}
