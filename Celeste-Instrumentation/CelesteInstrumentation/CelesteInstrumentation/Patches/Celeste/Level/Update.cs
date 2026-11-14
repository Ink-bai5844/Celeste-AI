using Celeste;
using HarmonyLib;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(Level), "Update")]
	class Celeste_Level_Update_Patch
	{
		[HarmonyPrefix]
		static bool Prefix(Level __instance)
		{
			// Engine.Scene queues a transition until the end of the engine tick.
			// A pre-input stop must not advance the old level once more before then.
			return !SearchPlaybackController.Enabled || SearchPlaybackController.StoppedLevel != __instance;
		}
	}
}
