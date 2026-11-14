namespace Instrumentation
{
	public class GlobalParameters
	{
		public bool Enabled;

		// Normal game startup with an optional, initially passive assistant.
		public bool AssistMode;

		public bool DebugRendering;

		public bool RouteRendering;

		public float GamespeedMultiplier;

		public bool Headless;

		public string ConnectionIP = "127.0.0.1";

		public int ConnectionPort = 25001;

	}
}
