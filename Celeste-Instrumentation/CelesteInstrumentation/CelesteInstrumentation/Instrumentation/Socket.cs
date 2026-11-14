using System.Net.Sockets;
using System.Text;
using System;

namespace Instrumentation
{
	public static class Socket
	{
		public static void Initialize()
		{
			Socket.Reset();
			while (true)
			{
				try
				{
					Socket.Client = new TcpClient(Socket.GetConnectionIP(), Socket.GetConnectionPort());
					Socket.Client.NoDelay = true;
					return;
				}
				catch
				{
					System.Threading.Thread.Sleep(50);
				}
			}
		}

		public static void Reset()
		{
			if (Socket.Client == null)
			{
				return;
			}
			try
			{
				Socket.Client.Close();
			}
			catch
			{
			}
			Socket.Client = null;
		}

		// Runs only on a worker thread. It never changes the active connection or
		// touches the current scene; the game thread adopts a completed handshake.
		public static TcpClient OpenLiveConnection(string snapshot)
		{
			TcpClient client = new TcpClient();
			try
			{
				IAsyncResult connecting = client.BeginConnect(GetConnectionIP(), GetConnectionPort(), null, null);
				using (System.Threading.WaitHandle handle = connecting.AsyncWaitHandle)
				{
					if (!handle.WaitOne(500)) throw new System.IO.IOException("Live reconnect timed out.");
					client.EndConnect(connecting);
				}
				client.NoDelay = true;
				NetworkStream stream = client.GetStream();
				stream.ReadTimeout = 500;
				stream.WriteTimeout = 500;
				WriteJsonMessage(stream, "LIVEINFO", snapshot);
				if (ReadExact(stream, 1)[0] != 0) throw new System.IO.IOException("Live reconnect was declined.");
				return client;
			}
			catch { client.Close(); throw; }
		}

		public static void AdoptLiveConnection(TcpClient client)
		{
			Reset();
			Client = client;
		}

		public static byte[] SendAndReceiveData(byte[] data)
		{
			NetworkStream stream = Socket.Client.GetStream();
			stream.Write(data, 0, data.Length);
			// All binary exchanges expect exactly one action/acknowledgment byte.
			return ReadExact(stream, 1);
		}

		public static string SendAndReceiveMessage(string message)
		{
			byte[] bytes = Encoding.ASCII.GetBytes(message);
			NetworkStream stream = Socket.Client.GetStream();
			stream.Write(bytes, 0, bytes.Length);
			// LOADINFO's XML reply can span multiple TCP packets. There is no binary
			// length in the legacy handshake, so read its complete root element.
			using (System.IO.MemoryStream payload = new System.IO.MemoryStream())
			{
				byte[] buffer = new byte[4096];
				while (payload.Length < 1024 * 1024)
				{
					int count = stream.Read(buffer, 0, buffer.Length);
					if (count <= 0)
					{
						throw new System.IO.IOException("Socket closed while reading session XML");
					}
					payload.Write(buffer, 0, count);
					string xml = Encoding.UTF8.GetString(payload.ToArray());
					if (xml.IndexOf("</SessionParameters>", StringComparison.Ordinal) >= 0)
					{
						return xml;
					}
				}
				throw new System.IO.InvalidDataException("Session XML exceeds 1 MiB or has no closing root element.");
			}
		}

		public static string SendMapInfoAndReceivePlan(string mapInfo)
		{
			NetworkStream stream = Socket.Client.GetStream();
			byte[] commandBytes = Encoding.ASCII.GetBytes("MAPINFO");
			stream.Write(commandBytes, 0, commandBytes.Length);
			ReadExact(stream, 1);

			byte[] payload = Encoding.UTF8.GetBytes(mapInfo);
			byte[] payloadLength = BitConverter.GetBytes(payload.Length);
			stream.Write(payloadLength, 0, payloadLength.Length);
			stream.Write(payload, 0, payload.Length);

			byte[] responseLengthBytes = ReadExact(stream, 4);
			int responseLength = BitConverter.ToInt32(responseLengthBytes, 0);
			if (responseLength <= 0 || responseLength > 1024 * 1024)
			{
				return string.Empty;
			}
			byte[] responseBytes = ReadExact(stream, responseLength);
			return Encoding.UTF8.GetString(responseBytes);
		}

		public static byte SendSearchStateAndReceiveInput(string state, int timeoutMilliseconds = 0)
		{
			NetworkStream stream = Socket.Client.GetStream();
			stream.ReadTimeout = timeoutMilliseconds > 0 ? timeoutMilliseconds : System.Threading.Timeout.Infinite;
			stream.WriteTimeout = timeoutMilliseconds > 0 ? timeoutMilliseconds : System.Threading.Timeout.Infinite;
			WriteJsonMessage(stream, "STATE", state);
			byte input = ReadExact(stream, 1)[0];
			if (input > 128 && input < 253)
			{
				throw new System.IO.InvalidDataException("Input must be 0..127, 128 (physical input), 253 (detach), 254 (native room respawn), or 255 (stop).");
			}
			return input;
		}

		public static void SendSearchDoneAndReceiveAck(string state, int timeoutMilliseconds = 1000)
		{
			NetworkStream stream = Socket.Client.GetStream();
			stream.ReadTimeout = timeoutMilliseconds;
			stream.WriteTimeout = timeoutMilliseconds;
			WriteJsonMessage(stream, "DONE", state);
			ReadExact(stream, 1);
		}

		private static void WriteJsonMessage(NetworkStream stream, string kind, string json)
		{
			byte[] header = Encoding.ASCII.GetBytes(kind);
			byte[] payload = Encoding.UTF8.GetBytes(json);
			byte[] length = BitConverter.GetBytes(payload.Length);
			if (!BitConverter.IsLittleEndian)
			{
				Array.Reverse(length);
			}
			stream.Write(header, 0, header.Length);
			stream.Write(length, 0, length.Length);
			stream.Write(payload, 0, payload.Length);
		}

		private static byte[] ReadExact(NetworkStream stream, int byteCount)
		{
			byte[] buffer = new byte[byteCount];
			int offset = 0;
			while (offset < byteCount)
			{
				int bytesRead = stream.Read(buffer, offset, byteCount - offset);
				if (bytesRead <= 0)
				{
					throw new System.IO.IOException("Socket closed while reading data");
				}
				offset += bytesRead;
			}
			return buffer;
		}

		private static string GetConnectionIP()
		{
			if (Instrumentation.GlobalParameters == null || string.IsNullOrEmpty(Instrumentation.GlobalParameters.ConnectionIP))
			{
				return "127.0.0.1";
			}
			return Instrumentation.GlobalParameters.ConnectionIP;
		}

		private static int GetConnectionPort()
		{
			if (Instrumentation.GlobalParameters == null || Instrumentation.GlobalParameters.ConnectionPort <= 0)
			{
				return 25001;
			}
			return Instrumentation.GlobalParameters.ConnectionPort;
		}

		private static TcpClient Client;
	}
}
