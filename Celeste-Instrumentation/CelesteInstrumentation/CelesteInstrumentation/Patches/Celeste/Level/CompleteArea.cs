using Celeste;
using HarmonyLib;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(Level), "CompleteArea")]
	class Celeste_Level_CompleteArea_Patch
	{
		[HarmonyPostfix]
		static void Postfix(Level __instance)
		{
			if (!Instrumentation.inSession || Instrumentation.SessionParameters == null || !Instrumentation.SessionParameters.SearchMode)
			{
				return;
			}
			// CompleteArea calls RegisterAreaComplete first, including collection of
			// carried berries. Ordinary cinematic/room transitions do not call this hook.
			Player player = __instance.Tracker.GetEntity<Player>();
			if (__instance.Completed && player != null)
			{
				Instrumentation.EndSession(player, false, "chapter_complete");
			}
		}
	}
}
