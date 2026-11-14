using System.IO;

namespace Instrumentation
{
	public class PlayerState
	{
		public const int RaycastCount = 32;
		public const int LocalGridSize = 7;
		public const int LocalGridCellCount = LocalGridSize * LocalGridSize;
		public const int InputValueCount = 133;
		public const int SerializedValueCount = 137;

		public PlayerState()
		{
			this.Distances = new int[RaycastCount];
			this.DetectedTypes = new int[RaycastCount];
			this.LocalGridTypes = new int[LocalGridCellCount];
		}

		public byte[] Serialize()
		{
			byte[] result;
			using (MemoryStream memoryStream = new MemoryStream())
			{
				using (BinaryWriter binaryWriter = new BinaryWriter(memoryStream))
				{
					foreach (float value in Distances)
					{
						binaryWriter.Write((float)value);
					}
					foreach (float value in DetectedTypes)
					{
						binaryWriter.Write((float)value);
					}
					foreach (float value in LocalGridTypes)
					{
						binaryWriter.Write((float)value);
					}
					binaryWriter.Write(this.XVelocity);
					binaryWriter.Write(this.YVelocity);
					binaryWriter.Write(this.CanDash ? 1f : 0f);
					binaryWriter.Write(this.OnGround ? 1f : 0f);
					binaryWriter.Write(this.Stamina);
					binaryWriter.Write(this.AngleToObjective);
					binaryWriter.Write(this.DistanceToObjective);
					binaryWriter.Write(this.InitialDistanceToObjective);
					binaryWriter.Write(this.BestDistanceToObjective);
					binaryWriter.Write(this.ProgressToObjective);
					binaryWriter.Write(this.NormalizedX);
					binaryWriter.Write(this.NormalizedY);
					binaryWriter.Write(this.NormalizedObjectiveX);
					binaryWriter.Write(this.NormalizedObjectiveY);
					binaryWriter.Write(this.DeltaXToObjective);
					binaryWriter.Write(this.DeltaYToObjective);
					binaryWriter.Write(this.HorizontalProgress);
					binaryWriter.Write(this.VerticalProgress);
					binaryWriter.Write(this.IsStuck ? 1f : 0f);
					binaryWriter.Write(this.SpeedMagnitude);
					binaryWriter.Write(this.SecondsElapsed);
					binaryWriter.Write(this.LevelDiagonalLength);
					binaryWriter.Write((float)this.FinishedLevelsNumber);
					binaryWriter.Write(this.Died ? 1f : 0f);
					result = memoryStream.ToArray();
				}
			}
			return result;
		}

		public int[] Distances;

		public int[] DetectedTypes;

		public int[] LocalGridTypes;

		public float XVelocity;

		public float YVelocity;

		public bool CanDash;

		public bool OnGround;

		public float Stamina;

		public float AngleToObjective;

		public float DistanceToObjective;

		public float InitialDistanceToObjective;

		public float BestDistanceToObjective;

		public float ProgressToObjective;

		public float NormalizedX;

		public float NormalizedY;

		public float NormalizedObjectiveX;

		public float NormalizedObjectiveY;

		public float DeltaXToObjective;

		public float DeltaYToObjective;

		public float HorizontalProgress;

		public float VerticalProgress;

		public bool IsStuck;

		public float SpeedMagnitude;

		public float SecondsElapsed;

		public float LevelDiagonalLength;

		public int FinishedLevelsNumber;

		public bool Died;
	}
}
