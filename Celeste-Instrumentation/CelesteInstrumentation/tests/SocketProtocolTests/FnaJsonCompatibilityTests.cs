using System;
using System.Collections.Generic;

static class FnaJsonCompatibilityTests
{
    public static void Run()
    {
#if CELESTE_FNA
        var serializer = new System.Web.Script.Serialization.JavaScriptSerializer();
        var source = new SortedDictionary<string, object>(StringComparer.Ordinal) {
            { "room", "测试\n\"\\" }, { "timer", 0.0166667f }, { "available", true },
            { "missing", null }, { "ids", new [] { "1:2", "1:3" } }
        };
        using (var parsed = System.Text.Json.JsonDocument.Parse(serializer.Serialize(source)))
        {
            var root = parsed.RootElement;
            if (root.GetProperty("room").GetString() != (string)source["room"] ||
                root.GetProperty("missing").ValueKind != System.Text.Json.JsonValueKind.Null ||
                !root.GetProperty("available").GetBoolean() ||
                Math.Abs(root.GetProperty("timer").GetDouble() - 0.0166667) > 1e-8 ||
                root.GetProperty("ids").GetArrayLength() != 2)
                throw new Exception("FNA JSON telemetry changed value semantics.");
        }
        var plan = (Dictionary<string, object>)serializer.DeserializeObject("{\"waypoints\":[{\"x\":12.5,\"y\":-4,\"action\":\"route\"}]}");
        var waypoints = plan["waypoints"] as object[];
        var point = waypoints == null ? null : waypoints[0] as Dictionary<string, object>;
        if (point == null || Convert.ToSingle(point["x"]) != 12.5f || Convert.ToSingle(point["y"]) != -4f)
            throw new Exception("FNA JSON must preserve the legacy dictionary/object-array plan shape.");
        Console.WriteLine("PASS production FNA JSON escaping, numeric/null/bool fields, and legacy waypoint shape");
#endif
    }
}
