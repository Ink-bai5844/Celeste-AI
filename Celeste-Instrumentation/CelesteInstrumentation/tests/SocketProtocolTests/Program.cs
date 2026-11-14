using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Monocle;
using Bridge = Instrumentation.Socket;

namespace Microsoft.Xna.Framework.Input
{
    public enum Keys { None, Right, Left, Up, Down, Jump, Dash, Grab, Confirm, Pause, Escape, Extra }
}

namespace Celeste
{
    public class Settings
    {
        public static Settings Instance = new Settings();
        public Binding Right = new Binding(Microsoft.Xna.Framework.Input.Keys.Right);
        public Binding Left = new Binding(Microsoft.Xna.Framework.Input.Keys.Left);
        public Binding Up = new Binding(Microsoft.Xna.Framework.Input.Keys.Up);
        public Binding Down = new Binding(Microsoft.Xna.Framework.Input.Keys.Down);
        public Binding Jump = new Binding(Microsoft.Xna.Framework.Input.Keys.Jump);
        public Binding Dash = new Binding(Microsoft.Xna.Framework.Input.Keys.Dash);
        public Binding Grab = new Binding(Microsoft.Xna.Framework.Input.Keys.Grab);
        public Binding Confirm = new Binding(Microsoft.Xna.Framework.Input.Keys.Confirm);
        public Binding Pause = new Binding(Microsoft.Xna.Framework.Input.Keys.Pause);
        public Binding RightMoveOnly = new Binding(), LeftMoveOnly = new Binding(), UpMoveOnly = new Binding(), DownMoveOnly = new Binding();
        public Binding RightDashOnly = new Binding(), LeftDashOnly = new Binding(), UpDashOnly = new Binding(), DownDashOnly = new Binding();
        public Binding DemoDash = new Binding(), QuickRestart = new Binding(), Cancel = new Binding(), Talk = new Binding();
        public Binding MenuLeft = new Binding(), MenuRight = new Binding(), MenuUp = new Binding(), MenuDown = new Binding(), Journal = new Binding();
    }
}

// Compiles the production Socket.cs without loading or launching Celeste.
namespace Instrumentation
{
    public class GlobalParameters
    {
        public string ConnectionIP;
        public int ConnectionPort;
    }
    public static class Instrumentation
    {
        public static GlobalParameters GlobalParameters;
        public static long PlaybackFrames;
    }
}

class Program
{
    static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    static byte[] Receive(NetworkStream stream, int count)
    {
        byte[] bytes = new byte[count];
        for (int offset = 0; offset < count;)
        {
            int received = stream.Read(bytes, offset, count - offset);
            if (received == 0) throw new EndOfStreamException();
            offset += received;
        }
        return bytes;
    }

    static void CheckMessage(NetworkStream stream, string token, string json)
    {
        Require(Encoding.ASCII.GetString(Receive(stream, token.Length)) == token, "Wrong token");
        byte[] size = Receive(stream, 4);
        uint count = (uint)(size[0] | size[1] << 8 | size[2] << 16 | size[3] << 24);
        Require(count == Encoding.UTF8.GetByteCount(json), "Length must count UTF-8 bytes");
        Require(Encoding.UTF8.GetString(Receive(stream, (int)count)) == json, "JSON payload changed");
    }

    sealed class Pair : IDisposable
    {
        public TcpClient BridgeClient;
        public TcpClient TestClient;
        public NetworkStream Stream { get { return TestClient.GetStream(); } }
        public Pair()
        {
            TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            BridgeClient = new TcpClient();
            Task connecting = BridgeClient.ConnectAsync(IPAddress.Loopback, ((IPEndPoint)listener.LocalEndpoint).Port);
            TestClient = listener.AcceptTcpClient();
            connecting.GetAwaiter().GetResult();
            listener.Stop();
            BridgeClient.ReceiveTimeout = 3000;
            TestClient.ReceiveTimeout = 3000;
            typeof(Bridge).GetField("Client", BindingFlags.NonPublic | BindingFlags.Static).SetValue(null, BridgeClient);
        }
        public void Dispose()
        {
            Bridge.Reset();
            TestClient.Dispose();
        }
    }

    static void Main()
    {
        FnaJsonCompatibilityTests.Run();
        ModelStateTests.Run();
        EnvironmentStateTests.Run();
        AssistInputTests.Run();
        Instrumentation.InputsManager.Initialize();
        Instrumentation.InputsManager.ApplyPlaybackInput(1 | 16);
        Require(Instrumentation.InputsManager.Check(Microsoft.Xna.Framework.Input.Keys.Right), "Direction must apply immediately");
        Require(Instrumentation.InputsManager.Pressed(Microsoft.Xna.Framework.Input.Keys.Jump), "New jump must produce an edge");
        Instrumentation.InputsManager.HoldPlaybackInput();
        Require(Instrumentation.InputsManager.Check(Microsoft.Xna.Framework.Input.Keys.Jump), "Freeze must retain a held button");
        Require(!Instrumentation.InputsManager.Pressed(Microsoft.Xna.Framework.Input.Keys.Jump), "Freeze must clear repeated pressed edge");
        Instrumentation.InputsManager.ApplyPlaybackInput(1);
        Require(Instrumentation.InputsManager.Released(Microsoft.Xna.Framework.Input.Keys.Jump), "Release must be visible immediately");
        Instrumentation.InputsManager.ApplyPlaybackInput(1 | 16);
        Require(Instrumentation.InputsManager.Pressed(Microsoft.Xna.Framework.Input.Keys.Jump), "Repress must produce a new edge");
        Require(!Instrumentation.InputsManager.Pressed(Microsoft.Xna.Framework.Input.Keys.Right), "Held direction must not repress");
        Console.WriteLine("PASS immediate input installation and freeze/release/repress edges");

        var dialogue = new Instrumentation.DialogueAdvancePulse();
        for (int tick = 1; tick < 15; tick++) Require(!dialogue.Next(true), "Dialogue must wait past the textbox's initial yield");
        Require(dialogue.Next(true), "A waiting textbox needs a normal confirmation pulse");
        Require(!dialogue.Next(true) && !dialogue.Next(false), "Confirmation must release and reset when the textbox resumes");
        Instrumentation.InputsManager.ApplyPlaybackInput(1 | 32);
        Instrumentation.InputsManager.ApplyDialogueConfirm(true);
        Require(Instrumentation.InputsManager.Pressed(Microsoft.Xna.Framework.Input.Keys.Confirm), "Dialogue confirmation edge missing");
        Require(!Instrumentation.InputsManager.Check(Microsoft.Xna.Framework.Input.Keys.Right) && !Instrumentation.InputsManager.Check(Microsoft.Xna.Framework.Input.Keys.Dash), "Dialogue must release gameplay movement");
        Require(!Instrumentation.InputsManager.Check(Microsoft.Xna.Framework.Input.Keys.Pause), "Dialogue must never pause");
        Instrumentation.InputsManager.ApplyDialogueConfirm(false);
        Require(Instrumentation.InputsManager.Released(Microsoft.Xna.Framework.Input.Keys.Confirm), "Dialogue confirmation must last one update");
        Celeste.Settings.Instance.Confirm.Keyboard[0] = Microsoft.Xna.Framework.Input.Keys.Pause;
        Instrumentation.InputsManager.ApplyDialogueConfirm(true);
        Require(!Instrumentation.InputsManager.Check(Microsoft.Xna.Framework.Input.Keys.Pause), "A conflicting user binding must never turn confirmation into pause");
        Celeste.Settings.Instance.Confirm.Keyboard[0] = Microsoft.Xna.Framework.Input.Keys.Confirm;
        Console.WriteLine("PASS textbox-only confirmation cadence, movement release, and pause-binding exclusion");

        string json = "{\"room\":\"测试\",\"frame\":12}";
        foreach (byte ack in new byte[] { 0, 1 })
        {
            using (Pair original = new Pair())
            {
                TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
                listener.Start();
                Instrumentation.Instrumentation.GlobalParameters = new Instrumentation.GlobalParameters { ConnectionIP = "127.0.0.1", ConnectionPort = ((IPEndPoint)listener.LocalEndpoint).Port };
                Task<TcpClient> connecting = Task.Run(() => Bridge.OpenLiveConnection(json));
                using (TcpClient server = listener.AcceptTcpClient())
                {
                    CheckMessage(server.GetStream(), "LIVEINFO", json);
                    Require(typeof(Bridge).GetField("Client", BindingFlags.NonPublic | BindingFlags.Static).GetValue(null) == original.BridgeClient, "Background handshake must not change active connection");
                    server.GetStream().WriteByte(ack);
                    if (ack == 0)
                    {
                        TcpClient client = connecting.GetAwaiter().GetResult();
                        Require(client.NoDelay, "Reconnect must disable Nagle batching");
                        Bridge.AdoptLiveConnection(client);
                        Task<byte> state = Task.Run(() => Bridge.SendSearchStateAndReceiveInput(json));
                        CheckMessage(server.GetStream(), "STATE", json);
                        server.GetStream().WriteByte(16);
                        Require(state.GetAwaiter().GetResult() == 16, "Adopted connection must accept fresh state commands");
                    }
                    else
                    {
                        bool rejected = false;
                        try { connecting.GetAwaiter().GetResult(); } catch (IOException) { rejected = true; }
                        Require(rejected, "A refused live attach must not acquire control");
                    }
                }
                listener.Stop();
            }
        }
        Instrumentation.Instrumentation.GlobalParameters = null;
        Console.WriteLine("PASS LIVEINFO framed reconnect, explicit acceptance, isolated background handshake and state exchange after adoption");
        using (Pair pair = new Pair())
        {
            Task<string> load = Task.Run(() => Bridge.SendAndReceiveMessage("LOADINFO"));
            Require(Encoding.ASCII.GetString(Receive(pair.Stream, 8)) == "LOADINFO", "Wrong load token");
            string xml = "<SessionParameters><Level>测试</Level><SearchMode>true</SearchMode></SessionParameters>";
            byte[] payload = Encoding.UTF8.GetBytes(xml);
            pair.Stream.Write(payload, 0, 19);
            pair.Stream.Flush();
            Require(!load.Wait(40), "XML must not return before the root element closes");
            pair.Stream.Write(payload, 19, payload.Length - 19);
            Require(load.GetAwaiter().GetResult() == xml, "Fragmented UTF-8 XML changed");
        }
        Console.WriteLine("PASS fragmented UTF-8 LOADINFO XML handshake");

        foreach (byte action in new byte[] { 0, 1, 64, 127, 128, 253, 254, 255 })
        {
            using (Pair pair = new Pair())
            {
                Task<byte> response = Task.Run(() => Bridge.SendSearchStateAndReceiveInput(json));
                CheckMessage(pair.Stream, "STATE", json);
                pair.Stream.WriteByte(action);
                Require(response.GetAwaiter().GetResult() == action, "Action was not preserved");
            }
        }
        Console.WriteLine("PASS STATE framing, UTF-8 byte length, normal/detach/native-respawn/stop commands");

        using (Pair pair = new Pair())
        {
            Task done = Task.Run(() => Bridge.SendSearchDoneAndReceiveAck(json));
            CheckMessage(pair.Stream, "DONE", json);
            Require(!done.IsCompleted, "DONE must wait for its acknowledgment");
            pair.Stream.WriteByte(0);
            done.GetAwaiter().GetResult();
        }
        Console.WriteLine("PASS DONE framing and acknowledgment");

        using (Pair pair = new Pair())
        {
            Task<byte> response = Task.Run(() => Bridge.SendSearchStateAndReceiveInput(json));
            CheckMessage(pair.Stream, "STATE", json);
            pair.Stream.WriteByte(129);
            bool rejected = false;
            try { response.GetAwaiter().GetResult(); }
            catch (InvalidDataException) { rejected = true; }
            Require(rejected, "Reserved action must be rejected");
        }
        Console.WriteLine("PASS invalid action rejection");

        using (Pair pair = new Pair())
        {
            MethodInfo method = typeof(Bridge).GetMethod("ReadExact", BindingFlags.NonPublic | BindingFlags.Static);
            Task<byte[]> reading = Task.Run(() => (byte[])method.Invoke(null, new object[] { pair.BridgeClient.GetStream(), 8 }));
            pair.Stream.Write(new byte[] { 1, 2, 3 }, 0, 3);
            pair.Stream.Flush();
            Require(!reading.Wait(40), "ReadExact returned a partial message");
            pair.Stream.Write(new byte[] { 4, 5, 6, 7, 8 }, 0, 5);
            byte[] bytes = reading.GetAwaiter().GetResult();
            Require(BitConverter.ToString(bytes) == "01-02-03-04-05-06-07-08", "Fragmented bytes changed");
        }
        Console.WriteLine("PASS fragmented exact-read behavior");

        using (Pair pair = new Pair())
        {
            Task<byte> response = Task.Run(() => Bridge.SendSearchStateAndReceiveInput(json));
            CheckMessage(pair.Stream, "STATE", json);
            pair.TestClient.Close();
            bool closed = false;
            try { response.GetAwaiter().GetResult(); }
            catch (IOException) { closed = true; }
            Require(closed, "EOF must fail instead of fabricating an input");
        }
        Console.WriteLine("PASS closed-client detection");

        using (Pair pair = new Pair())
        {
            Task<byte> response = Task.Run(() => Bridge.SendSearchStateAndReceiveInput(json, 80));
            CheckMessage(pair.Stream, "STATE", json);
            bool timedOut = false;
            try { response.GetAwaiter().GetResult(); }
            catch (IOException) { timedOut = true; }
            Require(timedOut, "Live control must fail back to detach instead of waiting forever for an idle controller");
        }
        Console.WriteLine("PASS bounded live-response watchdog");
        Console.WriteLine("All protocol checks passed; game was not launched.");
    }
}
