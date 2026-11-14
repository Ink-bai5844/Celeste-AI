using System;
using System.Collections.Generic;
using System.Reflection;
using Celeste;
using HarmonyLib;
using Microsoft.Xna.Framework;
using Monocle;

namespace Instrumentation.Patches
{
	[HarmonyPatch]
	internal static class EntityDataConstructors
	{
		static IEnumerable<MethodBase> TargetMethods()
		{
			foreach (string name in new[] { "Strawberry", "Refill", "DashBlock", "ZipMover", "FallingBlock", "CrumblePlatform", "Spikes", "Spring", "Jumpthru", "CassetteBlock" })
			{
				Type type = typeof(Player).Assembly.GetType("Celeste." + name);
				if (type == null) continue;
				foreach (ConstructorInfo constructor in type.GetConstructors())
				{
					ParameterInfo[] parameters = constructor.GetParameters();
					if (parameters.Length >= 2 && parameters[0].ParameterType == typeof(EntityData) && parameters[1].ParameterType == typeof(Vector2)) yield return constructor;
				}
			}
		}
		static void Postfix(Entity __instance, EntityData __0, Vector2 __1)
		{
			SearchEnvironmentState.Register(__instance, __0, __1);
		}
	}
}
