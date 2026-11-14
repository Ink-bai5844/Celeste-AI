using HarmonyLib;
using Monocle;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(MInput), "UpdateVirtualInputs")]
	class Monocle_MInput_UpdateVirtualInputs_Patch
	{
		[HarmonyPrefix]
		static void Prefix()
		{
			SearchPlaybackController.BeforeVirtualInputs();
		}

		[HarmonyPostfix]
		static void Postfix()
		{
			if (SearchPlaybackController.Enabled) AssistInputState.ObserveEffectiveInput();
		}
	}
}
