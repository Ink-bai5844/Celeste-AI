using HarmonyLib;
using Celeste;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(Player), "Update")]
	class Celeste_Player_Update_Patch
	{
		[HarmonyPrefix]
		static void Prefix(Player __instance)
		{
			if (!SearchPlaybackController.Enabled)
			{
				InputsManager.Update();
			}
		}
	}
}
