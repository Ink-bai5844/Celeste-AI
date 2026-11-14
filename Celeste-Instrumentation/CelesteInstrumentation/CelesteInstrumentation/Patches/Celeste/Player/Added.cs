using HarmonyLib;
using Celeste;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(Player), "Added")]
	class Celeste_Player_Added_Patch
	{
		[HarmonyPrefix]
		static void Prefix(Player __instance)
		{
			if (Instrumentation.AssistMode)
			{
				InputsManager.Initialize(true);
				Instrumentation.RegisterPlaybackSpawn(__instance);
				return;
			}
			if (SearchPlaybackController.Enabled && !Instrumentation.inSession) return;
			InputsManager.Initialize(false);
			if (!SearchPlaybackController.Enabled || __instance.Components.Get<PlayerStateTracker>() == null)
			{
				__instance.Components.Add(new PlayerStateTracker(__instance));
			}
			Instrumentation.RegisterPlaybackSpawn(__instance);
		}
	}
}
