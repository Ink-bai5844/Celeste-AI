using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Instrumentation;

static class Program
{
    static int assertions;
    static void Require(bool value, string message)
    {
        assertions++;
        if (!value) throw new Exception(message);
    }

    sealed class Fixture : IDisposable
    {
        public readonly string Directory = Path.Combine(Path.GetTempPath(), "celeste-reconnect-test-" + Guid.NewGuid().ToString("N"));
        public readonly SearchReconnectSignal Signal;
        public int Requests;
        public Fixture()
        {
            System.IO.Directory.CreateDirectory(Directory);
            Signal = new SearchReconnectSignal(Directory, DateTime.UtcNow.AddSeconds(-1));
        }
        public string RequestPath { get { return Path.Combine(Directory, SearchReconnectSignal.RequestName); } }
        public string StatusPath { get { return Path.Combine(Directory, SearchReconnectSignal.StatusName); } }
        public void Write(string id, DateTime timestamp, string schema = SearchReconnectSignal.RequestSchema)
        {
            File.WriteAllText(RequestPath, JsonSerializer.Serialize(new { schema, request_id=id, requested_at_utc=timestamp.ToString("o") }));
        }
        public JsonElement Wait(string id, bool connected = false, bool held = false)
        {
            var watch = Stopwatch.StartNew();
            while (watch.Elapsed.TotalSeconds < 3)
            {
                if (Signal.Poll(connected, held, DateTime.UtcNow)) Requests++;
                if (File.Exists(StatusPath))
                {
                    using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(StatusPath)))
                        if (doc.RootElement.GetProperty("request_id").GetString() == id)
                            return doc.RootElement.Clone();
                }
                Thread.Sleep(1);
            }
            throw new Exception("Timed out waiting for signal acknowledgement " + id);
        }
        public string Status(string id, bool connected = false, bool held = false)
        { return Wait(id, connected, held).GetProperty("status").GetString(); }
        public void Dispose()
        {
            // A final no-file read may still be running; fixture directories are
            // private and can safely disappear while that read handles absence.
            System.IO.Directory.Delete(Directory, true);
        }
    }

    static void Main()
    {
        using (var f = new Fixture())
        {
            f.Write("fresh", DateTime.UtcNow);
            var ack = f.Wait("fresh");
            Require(ack.GetProperty("status").GetString() == "accepted_pending" && f.Requests == 1, "Fresh request must request connection exactly once");
            Require(ack.GetProperty("schema").GetString() == "celeste-assist-reconnect-status-v1" && ack.GetProperty("bridge_process_id").GetInt32() == Environment.ProcessId, "Status must identify protocol and owning process");
            Require(!File.Exists(f.RequestPath), "Accepted file must be consumed");
            File.Delete(f.StatusPath);
            f.Write("fresh", DateTime.UtcNow);
            Require(f.Status("fresh") == "ignored_duplicate" && f.Requests == 1, "Repeated identity must not reconnect twice");
            f.Write("connected", DateTime.UtcNow);
            Require(f.Status("connected", connected:true) == "ignored_connected" && f.Requests == 1, "Already connected must not interrupt input/takeover");
        }
        using (var f = new Fixture())
        {
            f.Write("stale", DateTime.UtcNow.AddSeconds(-31));
            Require(f.Status("stale") == "rejected_stale", "Old timestamp must be rejected");
            f.Write("future", DateTime.UtcNow.AddSeconds(10));
            Require(f.Status("future") == "rejected_future", "Future timestamp must be rejected");
            f.Write("schema", DateTime.UtcNow, "unknown");
            Require(f.Status("schema") == "rejected_invalid", "Unknown version must be rejected");
            File.Delete(f.StatusPath);
            File.WriteAllText(f.RequestPath, new string('x', 4097));
            Require(f.Status(null) == "rejected_invalid", "Oversized input must be rejected before parse");
            File.Delete(f.StatusPath);
            File.WriteAllText(f.RequestPath, "{invalid");
            Require(f.Status(null) == "rejected_invalid", "Malformed JSON must not throw on the game thread");
            Require(f.Requests == 0, "Invalid requests must have no control effect");
        }
        using (var f = new Fixture())
        {
            f.Write("before-start", DateTime.UtcNow);
            File.SetLastWriteTimeUtc(f.RequestPath, DateTime.UtcNow.AddMinutes(-1));
            Require(f.Status("before-start") == "rejected_stale", "A leftover file must not reconnect after bridge startup");
        }
        using (var f = new Fixture())
        {
            f.Write("inflight", DateTime.UtcNow.AddSeconds(2));
            f.Signal.Poll(false, false, DateTime.UtcNow); // Schedule IO, do not wait.
            f.Signal.UserDetached(DateTime.UtcNow);
            Require(f.Status("inflight") == "rejected_user_detached" && f.Requests == 0, "F12 must defeat an in-flight request even with sender clock skew");
            Thread.Sleep(3);
            f.Write("while-held", DateTime.UtcNow);
            Require(f.Status("while-held", held:true) == "rejected_user_detached", "Held F12 must win over a newer request");
            Thread.Sleep(3);
            f.Write("after-release", DateTime.UtcNow);
            Require(f.Status("after-release") == "accepted_pending" && f.Requests == 1, "A new click after releasing F12 must be allowed");
        }
        using (var f = new Fixture())
        {
            // Install an intentionally incomplete filesystem task; Poll must
            // not block, join, or use Result before IsCompleted.
            Type requestType = typeof(SearchReconnectSignal).GetNestedType("Request", BindingFlags.NonPublic);
            Type sourceType = typeof(TaskCompletionSource<>).MakeGenericType(requestType);
            object source = Activator.CreateInstance(sourceType);
            object task = sourceType.GetProperty("Task").GetValue(source);
            typeof(SearchReconnectSignal).GetField("pendingRead", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(f.Signal, task);
            var clock = Stopwatch.StartNew();
            bool triggered = false;
            for (int i = 0; i < 10000; i++) triggered |= f.Signal.Poll(false, false, DateTime.UtcNow);
            Require(!triggered && clock.ElapsedMilliseconds < 500, "Pending disk IO must never block the frame thread");
            sourceType.GetMethod("SetResult").Invoke(source, new object[] { null });
        }
        Console.WriteLine("PASS " + assertions + " reconnect signal assertions: fresh/duplicate/connected, TTL/version/size, startup/F12 barriers, asynchronous IO");
    }
}
