using System;
using System.IO;

namespace Instrumentation
{
   public static class Entry
   {
        private static bool initialized;

        public static void Execute()
        {
            if (initialized) return;
            initialized = true;
            try
            {
                Log("Initializing bridge " + typeof(Entry).Assembly.Location + "; runtime=" + Environment.Version);
                Instrumentation.Initialize();
                if (Instrumentation.GlobalParameters.Enabled)
                {
                    Patcher.ApplyPatches();
                    Log("Bridge initialized; assist_mode=" + Instrumentation.AssistMode);
                }
                else Log("Bridge disabled by configuration.");
            }
            catch (Exception exception)
            {
                Log("Bridge initialization failed: " + exception);
                // A partial patch installation must not leave synthetic inputs
                // active when a different game build rejects a hook signature.
                try { new HarmonyLib.Harmony("CelesteInstrumentation").UnpatchAll("CelesteInstrumentation"); }
                catch (Exception rollback) { Log("Patch rollback failed: " + rollback); }
                InputsManager.Initialize(true);
                Instrumentation.inSession = false;
                SearchReconnect.Disable();
            }
        }

        private static void Log(string message)
        {
            string line = DateTime.UtcNow.ToString("o") + " " + message;
            try { Console.Error.WriteLine(line); } catch { }
            try { File.AppendAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bridge-startup.log"), line + Environment.NewLine); }
            catch { }
        }
    }
}
