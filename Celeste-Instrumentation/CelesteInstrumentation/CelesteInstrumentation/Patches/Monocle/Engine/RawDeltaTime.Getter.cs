using HarmonyLib;
using Monocle;

namespace Instrumentation.Patches
{
    [HarmonyPatch(typeof(Engine), "RawDeltaTime", MethodType.Getter)]
    class Monocle_Engine_RawDeltaTime_Getter_Patch
    {
        [HarmonyPrefix]
        static bool Prefix()
        {
            return Instrumentation.AssistMode;
        }

        [HarmonyPostfix]
        static float Postfix(float __result)
        {
            return Instrumentation.AssistMode ? __result : 0.0166667f;
        }
    }
}
