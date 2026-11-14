namespace Instrumentation
{
	public sealed class DialogueAdvancePulse
	{
		private int waitingFrames;
		public bool Next(bool waitingForInput)
		{
			if (!waitingForInput) { waitingFrames = 0; return false; }
			// Textbox sets waitingForInput before an initial 0.1-second yield.
			// Wait 15 native updates before confirming; repeat only if still waiting.
			return ++waitingFrames % 15 == 0;
		}
	}
}
