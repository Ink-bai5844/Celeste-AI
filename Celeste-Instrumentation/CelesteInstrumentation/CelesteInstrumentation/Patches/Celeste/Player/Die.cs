using HarmonyLib;
using Celeste;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(Player), "Die")]
	class Celeste_Player_Die_Patch
	{
		[HarmonyPrefix]
		static bool Prefix(Player __instance)
		{
			if (Instrumentation.SessionParameters == null || (SearchPlaybackController.Enabled && !Instrumentation.inSession)) return true;
			if (Instrumentation.SessionParameters.EndOnDeath)
			{
				Instrumentation.EndSession(__instance, true);
				return false;
			}
			return true;
		}

		[HarmonyPostfix]
		static void Postfix(PlayerDeadBody __result)
		{
			if (__result != null && Instrumentation.inSession && SearchPlaybackController.Enabled)
			{
				if (Instrumentation.AssistMode) InputsManager.PassPhysicalInput();
				Instrumentation.PlaybackDeaths++;
				SearchPlaybackController.LastControlEvent = "died";
			}
		}
	}
}
