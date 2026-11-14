namespace Instrumentation
{
	public class SessionParameters
	{
		// Opt-in playback telemetry. Legacy clients retain the 137-float protocol.
		public bool SearchMode;

		public bool EndOnDeath;

		public bool EndOnLevelExit;

		public float TimeoutSeconds;

		public float ObjectiveXCoordinate;

		public float ObjectiveYCoordinate;

		public string AreaKey;

		public string Level;

		public int AreaMode;
	}
}
