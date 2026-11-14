using System;
using Celeste;
using HarmonyLib;
using Microsoft.Xna.Framework;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(LevelLoader), MethodType.Constructor, new Type[] { typeof(Session), typeof(Vector2?) })]
	class Celeste_LevelLoader_Ctor_Patch
	{
		[HarmonyPrefix]
		static void Prefix()
		{
			if (!Instrumentation.AssistMode) return;
			// A human-selected native chapter/load creates a different level.
			// Clear observations before its loading thread constructs entities so
			// identical room:id values cannot retain an older chapter's removal.
			SearchEnvironmentState.Reset();
			InputsManager.PassPhysicalInput();
			SearchPlaybackController.LastControlEvent = "native_level_loading";
		}
	}
}
