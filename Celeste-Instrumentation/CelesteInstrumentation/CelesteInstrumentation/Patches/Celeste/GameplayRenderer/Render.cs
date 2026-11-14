using HarmonyLib;
using Monocle;
using Celeste;

namespace Instrumentation.Patches
{
	[HarmonyPatch(typeof(GameplayRenderer), "Render", new System.Type[] { typeof(Scene) })]
	class Celeste_GameplayRenderer_Render_Patch
	{
		[HarmonyPrefix]
		static bool Prefix(GameplayRenderer __instance, Scene scene)
		{
			if (Instrumentation.AssistMode) return true;
			GameplayRenderer.Begin();
			scene.Entities.RenderExcept(Tags.HUD);
			if (Instrumentation.GlobalParameters.DebugRendering || Instrumentation.GlobalParameters.RouteRendering || Engine.Commands.Open)
			{
				scene.Entities.DebugRender(__instance.Camera);
			}
			GameplayRenderer.End();
			return false;
		}
	}
}
