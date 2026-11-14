using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace Instrumentation
{
    // Local notification only. All filesystem work stays off the game thread;
    // Poll returns true only for a fresh request to reconnect the current scene.
    public sealed class SearchReconnectSignal
    {
        public const string RequestName = "AssistReconnect.request.json";
        public const string StatusName = "AssistReconnect.status.json";
        public const string RequestSchema = "celeste-assist-reconnect-v1";
        private readonly string directory;
        private readonly Dictionary<string, DateTime> seen = new Dictionary<string, DateTime>(StringComparer.Ordinal);
        private readonly Queue<string> seenOrder = new Queue<string>();
        private DateTime nextPoll, rejectThrough;
        private bool detachedByUser;
        private long generation;
        private Task<Request> pendingRead;
        private Task pendingWrite;

        private sealed class Request
        {
            public string Id, Error;
            public DateTime RequestedAt, WrittenAt;
            public long Generation;
        }

        public SearchReconnectSignal(string directory, DateTime startedAtUtc)
        {
            this.directory = directory;
            rejectThrough = startedAtUtc;
        }

        // Call on F12 before Poll. A request already in flight must not undo an
        // explicit detach, even when its sender's clock is a little ahead.
        public void UserDetached(DateTime nowUtc)
        {
            rejectThrough = nowUtc;
            detachedByUser = true;
            generation++;
        }

        public bool Poll(bool connected, bool userDetachHeld, DateTime nowUtc)
        {
            bool reconnect = false;
            if (pendingRead != null && pendingRead.IsCompleted)
            {
                Request request = pendingRead.Result; // ReadRequest catches IO/parse failures.
                pendingRead = null;
                if (request != null)
                {
                    string status = Decide(request, connected, userDetachHeld, nowUtc);
                    reconnect = status == "accepted_pending";
                    pendingWrite = Task.Run(() => WriteStatus(request.Id, status, nowUtc));
                }
            }
            // At most one small read/write operation at a time. Never wait for it.
            if (pendingRead == null && (pendingWrite == null || pendingWrite.IsCompleted) && nowUtc >= nextPoll)
            {
                nextPoll = nowUtc.AddMilliseconds(200);
                long readGeneration = generation;
                pendingRead = Task.Run(() => ReadRequest(readGeneration));
            }
            return reconnect;
        }

        private string Decide(Request request, bool connected, bool userDetachHeld, DateTime nowUtc)
        {
            if (request.Error != null) return request.Error;
            if (userDetachHeld || request.Generation != generation ||
                (detachedByUser && (request.WrittenAt <= rejectThrough || request.RequestedAt <= rejectThrough)))
                return "rejected_user_detached";
            if (request.WrittenAt <= rejectThrough || (nowUtc - request.RequestedAt).TotalSeconds > 30)
                return "rejected_stale";
            if ((request.RequestedAt - nowUtc).TotalSeconds > 5) return "rejected_future";
            while (seenOrder.Count > 0 && (seen.Count >= 256 || (nowUtc - seen[seenOrder.Peek()]).TotalSeconds > 35))
                seen.Remove(seenOrder.Dequeue());
            if (seen.ContainsKey(request.Id)) return "ignored_duplicate";
            seen.Add(request.Id, nowUtc);
            seenOrder.Enqueue(request.Id);
            return connected ? "ignored_connected" : "accepted_pending";
        }

        private Request ReadRequest(long readGeneration)
        {
            string claimed = Path.Combine(directory, RequestName + ".reading." + Guid.NewGuid().ToString("N"));
            Request result = new Request { Generation = readGeneration };
            try
            {
                string path = Path.Combine(directory, RequestName);
                // Claim by rename: a concurrently replaced/new request is never
                // accidentally deleted when this request finishes processing.
                try { File.Move(path, claimed); }
                catch (FileNotFoundException) { return null; }
                catch (DirectoryNotFoundException) { return null; }
                result.WrittenAt = File.GetLastWriteTimeUtc(claimed);
                using (var stream = new FileStream(claimed, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    if (stream.Length <= 0 || stream.Length > 4096) throw new FormatException("Request size");
                    byte[] bytes = new byte[(int)stream.Length];
                    int offset = 0;
                    while (offset < bytes.Length)
                    {
                        int count = stream.Read(bytes, offset, bytes.Length - offset);
                        if (count == 0) throw new EndOfStreamException();
                        offset += count;
                    }
                    var data = new JavaScriptSerializer().DeserializeObject(new UTF8Encoding(false, true).GetString(bytes)) as Dictionary<string, object>;
                    object value;
                    string id = data != null && data.TryGetValue("request_id", out value) ? value as string : null;
                    if (ValidId(id)) result.Id = id;
                    if (data == null || result.Id == null || !data.TryGetValue("schema", out value) || !RequestSchema.Equals(value))
                        throw new FormatException("Request schema or id");
                    if (!data.TryGetValue("requested_at_utc", out value) || !TryUtc(value, out result.RequestedAt))
                        throw new FormatException("Request UTC timestamp");
                }
                return result;
            }
            catch (Exception) { result.Error = "rejected_invalid"; return result; }
            finally
            {
                try { if (File.Exists(claimed)) File.Delete(claimed); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }

        private static bool ValidId(string id)
        {
            if (String.IsNullOrEmpty(id) || id.Length > 80) return false;
            foreach (char c in id)
                if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') && c != '-' && c != '_')
                    return false;
            return true;
        }

        private static bool TryUtc(object value, out DateTime utc)
        {
            utc = default(DateTime);
            // FNA's existing JSON adapter returns ISO date tokens as DateTime;
            // System.Web on Windows returns the original timestamp string.
            if (value is DateTime)
            {
                DateTime date = (DateTime)value;
                if (date.Kind == DateTimeKind.Unspecified) return false;
                utc = date.ToUniversalTime();
                return true;
            }
            string text = value as string;
            DateTimeOffset parsed;
            if (text == null || !(text.EndsWith("Z", StringComparison.Ordinal) || text.EndsWith("+00:00", StringComparison.Ordinal)) ||
                !DateTimeOffset.TryParse(text, CultureInfo.InvariantCulture, DateTimeStyles.None, out parsed)) return false;
            utc = parsed.UtcDateTime;
            return true;
        }

        private void WriteStatus(string id, string status, DateTime nowUtc)
        {
            string temporary = Path.Combine(directory, StatusName + ".tmp." + Guid.NewGuid().ToString("N"));
            try
            {
                var value = new Dictionary<string, object> {
                    { "schema", "celeste-assist-reconnect-status-v1" }, { "request_id", id },
                    { "status", status }, { "processed_at_utc", nowUtc.ToString("o", CultureInfo.InvariantCulture) },
                    { "bridge_process_id", Process.GetCurrentProcess().Id }
                };
                File.WriteAllText(temporary, new JavaScriptSerializer().Serialize(value), new UTF8Encoding(false));
                string destination = Path.Combine(directory, StatusName);
                if (File.Exists(destination)) File.Replace(temporary, destination, null);
                else File.Move(temporary, destination);
            }
            catch (Exception) { } // An acknowledgement failure must not stop native input.
            finally
            {
                try { if (File.Exists(temporary)) File.Delete(temporary); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }
}
