using System;
using System.Collections.Generic;
using Instrumentation;
using Microsoft.Xna.Framework;

#pragma warning disable 0414, 0649, 0169
namespace Monocle
{
    public class Scene
    {
        public List<Entity> Entities = new List<Entity>();
        public Tracker Tracker;
        private Dictionary<int, double> actualDepthLookup = new Dictionary<int, double>();
        public Scene() { Tracker = new Tracker(Entities); }
        public void SetDepthCounter(int depth, double value) { actualDepthLookup[depth] = value; }
    }
    public class Tracker
    {
        private List<Entity> entities;
        public Tracker(List<Entity> entities) { this.entities = entities; }
        public List<Entity> CassetteOrder;
        public List<Entity> GetEntities<T>() where T : class
        {
            if (typeof(T) == typeof(Celeste.CassetteBlock) && CassetteOrder != null) return CassetteOrder;
            return entities.FindAll(entity => entity is T);
        }
        public T GetEntity<T>() where T : class
        {
            foreach (Entity entity in entities) if (entity is T) return entity as T;
            return null;
        }
    }
    public class Component { public bool Active = true; }
    public class Coroutine : Component { public bool Finished; private float waitTimer; private System.Collections.Generic.Stack<System.Collections.IEnumerator> enumerators = new System.Collections.Generic.Stack<System.Collections.IEnumerator>(); }
    public class Entity
    {
        public Scene Scene;
        public float X, Y, Width = 16, Height = 16;
        public float Left { get { return X - 8; } }
        public float Top { get { return Y - 8; } }
        public object Collider = new object();
        public bool Collidable = true, Active = true;
        public int Depth;
        private double actualDepth;
        public void SetActualDepth(double value) { actualDepth = value; }
        public List<Component> Components = new List<Component>();
    }
}
namespace Celeste
{
    public class LevelData { public string Name = "room-a"; }
    public class EntityData { public LevelData Level = new LevelData(); public int ID; public string Name; public Vector2 Position; }
    public class Platform : Monocle.Entity { public Vector2 LiftSpeed; private Vector2 movementCounter = new Vector2(0.25f, 0.5f); }
    public class Refill : Monocle.Entity { private float respawnTimer = 1.25f; private bool oneUse = true, twoDashes; }
    public class DashBlock : Platform { private bool permanent = true, canDash = true; }
    public class ZipMover : Platform { private float percent = 0.5f; private Vector2 target = new Vector2(100, 120); }
    public class FallingBlock : Platform { public bool Triggered; public float FallDelay; public bool HasStartedFalling { get; set; } }
    public class CrumblePlatform : Platform { }
    public class CassetteBlock : Platform
    {
        public enum Modes { Solid, Leaving, Disabled, Returning }
        public int Index = 1;
        public float Tempo = 1.25f;
        public bool Activated = true;
        public Modes Mode = Modes.Returning;
        public EntityID ID = new EntityID();
        private int blockHeight = 1;
        private bool groupLeader;
        private List<CassetteBlock> group;
        public void SetGroup(bool leader, params CassetteBlock[] blocks) { groupLeader = leader; group = new List<CassetteBlock>(blocks); }
    }
    public class CassetteBlockManager : Monocle.Entity
    {
        private int currentIndex = 1, beatIndex = 7, leadBeats = 9, maxBeat = 2, beatIndexOffset;
        private float beatTimer = .15625f, tempoMult = 1.25f;
        private bool isLevelMusic;
        private object sfx;
        public void SetMusic(bool levelMusic, bool ready) { isLevelMusic = levelMusic; sfx = ready ? new object() : null; }
        public void SetBeat(float timer, int index) { beatTimer = timer; beatIndex = index; }
    }
}

static class EnvironmentStateTests
{
    static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }
    public static void Run()
    {
        SearchEnvironmentState.Reset();
        Celeste.Level level = new Celeste.Level();
        Celeste.EntityData map = new Celeste.EntityData { ID = 12, Name = "refill", Position = new Vector2(8, 16) };
        Celeste.Refill refill = new Celeste.Refill { X = 108, Y = 216, Scene = level, Collidable = false };
        SearchEnvironmentState.Register(refill, map, new Vector2(100, 200));
        Celeste.ZipMover zip = new Celeste.ZipMover { Scene = level, X = 108, Y = 216 };
        SearchEnvironmentState.Register(zip, new Celeste.EntityData { ID = 13, Name = "zipMover", Position = new Vector2(8, 16) }, new Vector2(100, 200));
        refill.X = 120;
        var snapshot = SearchEnvironmentState.Capture(level);
        var entities = (List<Dictionary<string, object>>)snapshot["entities"];
        Require(entities.Count == 2 && (string)entities[0]["id"] == "room-a:12", "Overlapping entities need independent constructor IDs");
        Require((float)entities[0]["map_x"] == 108 && (float)entities[0]["x"] == 120, "Moving entities must preserve map origin");
        Require(!(bool)entities[0]["collidable"] && (float)entities[0]["respawn_timer"] == 1.25f, "Refill cooldown must retain actual seconds and collider state");
        Require((bool)entities[0]["present"] && (bool)entities[0]["one_use"], "Live refill state missing");
        Require((float)entities[1]["percent"] == 0.5f && (float)entities[1]["rem_x"] == 0.25f, "Platform subpixels and motion observation missing");
        Require(((List<string>)entities[1]["unavailable"]).Contains("coroutine_phase"), "Geometry must not fabricate coroutine position");
        refill.Scene = null;
        level.Session.DoNotLoad.Add(new Celeste.EntityID { Level = "room-a", ID = 12 });
        snapshot = SearchEnvironmentState.Capture(level);
        entities = (List<Dictionary<string, object>>)snapshot["entities"];
        Require(!(bool)entities[0]["present"], "Removed entity must remain distinguishable from a fresh entity");
        Require(((List<string>)snapshot["do_not_load"])[0] == "room-a:12", "Persistent removed IDs missing");
        Celeste.Refill replacement = new Celeste.Refill { Scene = level };
        SearchEnvironmentState.Register(replacement, map, new Vector2(100, 200));
        entities = (List<Dictionary<string, object>>)SearchEnvironmentState.Capture(level)["entities"];
        Require((bool)entities[0]["present"] && (bool)entities[0]["collidable"], "Native respawn must replace stale constructor observations");
        Console.WriteLine("PASS constructor entity identities, exact runtime timers/geometry, removed-state retention and native respawn replacement");
        CassetteState();
    }

    static void CassetteState()
    {
        SearchEnvironmentState.Reset();
        var level = new Celeste.Level();
        var first = new Celeste.CassetteBlock { X = 80, Y = 98, Scene = level, Collidable = false,
            ID = new Celeste.EntityID { Level = "room-a", ID = 30 } };
        var second = new Celeste.CassetteBlock { X = 96, Y = 98, Scene = level,
            ID = new Celeste.EntityID { Level = "room-a", ID = 31 } };
        first.SetGroup(true, second, first);
        second.SetGroup(false, second, first);
        var manager = new Celeste.CassetteBlockManager { Scene = level };
        level.Entities.Add(new Celeste.Player());
        level.Entities.Add(second);
        level.Entities.Add(manager);
        level.Entities.Add(first);
        // Scene order is the old update list. The upcoming BeforeUpdate sorts
        // actualDepth; Tracker order remains insertion order independently.
        level.Tracker.CassetteOrder = new List<Monocle.Entity> { first, second };
        first.Depth = 8990; first.SetActualDepth(8989.999998);
        second.Depth = -10; second.SetActualDepth(-10.000004);
        manager.Depth = 0; manager.SetActualDepth(-0.000008);
        level.Entities[0].Depth = 0; level.Entities[0].SetActualDepth(-0.000003);
        level.SetDepthCounter(8990, 0.000009);
        SearchEnvironmentState.Register(first, new Celeste.EntityData { ID = 30, Name = "cassetteBlock", Position = new Vector2(80, 96) }, new Vector2());
        SearchEnvironmentState.Register(second, new Celeste.EntityData { ID = 31, Name = "cassetteBlock", Position = new Vector2(96, 96) }, new Vector2());
        var snapshot = SearchEnvironmentState.Capture(level);
        var clock = (Dictionary<string, object>)snapshot["cassette_manager"];
        var blocks = (List<Dictionary<string, object>>)snapshot["entities"];
        var block = blocks[0];
        Require((int)snapshot["player_update_order"] == 0 && (int)clock["update_order"] == 2 && (int)block["update_order"] == 3, "Native entity update order must be preserved independently of ID sorting");
        Require((int)block["tracker_order"] == 0 && (int)blocks[1]["tracker_order"] == 1, "Cassette Tracker order must not be derived from the depth-sorted Scene list");
        Require((int)block["depth"] == 8990 && (double)block["actual_depth"] == 8989.999998 &&
                (double)clock["actual_depth"] == -0.000008 && (double)snapshot["player_actual_depth"] == -0.000003,
                "Upcoming update ordering needs exact double depth keys, not stale list indices or rounded offsets");
        Require(((Dictionary<string, double>)snapshot["depth_counters"])["8990"] == 0.000009,
                "Depth re-entry consumes the Scene's current per-depth counter");
        first.SetActualDepth(8989.999987); level.SetDepthCounter(8990, 0.000014);
        var refreshed = SearchEnvironmentState.Capture(level);
        Require((double)((List<Dictionary<string, object>>)refreshed["entities"])[0]["actual_depth"] == 8989.999987 &&
                ((Dictionary<string, double>)refreshed["depth_counters"])["8990"] == 0.000014,
                "Depth telemetry must refresh when a block returns to an earlier depth");
        Require((bool)clock["present"] && (bool)clock["active"] && !(bool)clock["music_ready"], "Private music's first non-advancing frame must remain observable");
        Require((float)clock["beat_timer"] == .15625f && (int)clock["beat_index"] == 7 && (int)clock["lead_beats"] == 9, "Cassette beat seconds/index/lead-in cannot be reconstructed or rounded");
        Require((int)clock["current_index"] == 1 && (int)clock["max_beat"] == 2 && (float)clock["tempo_mult"] == 1.25f, "Manager cycle parameters missing");
        Require((bool)block["activated"] && !(bool)block["collidable"] && (float)block["y"] == 98 && (float)block["map_y"] == 96, "Blocked activation and shifted geometry must not be conflated");
        Require((int)block["mode"] == 3 && (int)block["block_height"] == 1 && (bool)block["group_leader"], "Cassette block private/group state missing");
        var group = (List<string>)block["group_ids"];
        Require(group.Count == 2 && group[0] == "room-a:31" && group[1] == "room-a:30", "Native group order must not be sorted");
        manager.SetMusic(false, true);
        manager.SetBeat(.03125f, 8);
        clock = (Dictionary<string, object>)SearchEnvironmentState.Capture(level)["cassette_manager"];
        Require((bool)clock["music_ready"] && (float)clock["beat_timer"] == .03125f && (int)clock["beat_index"] == 8, "Manager observations must refresh every capture");
        manager.SetMusic(true, false);
        clock = (Dictionary<string, object>)SearchEnvironmentState.Capture(level)["cassette_manager"];
        Require((bool)clock["music_ready"] && (bool)clock["is_level_music"], "Level-music manager advances without a stored private music instance");
        first.SetGroup(true);
        level.Entities.Remove(manager);
        clock = (Dictionary<string, object>)SearchEnvironmentState.Capture(level)["cassette_manager"];
        Require(!(bool)clock["present"] && !clock.ContainsKey("beat_timer"), "Absent managers cannot fabricate a beat phase");
        Console.WriteLine("PASS cassette manager exact clocks, music readiness, physical activation/geometry, native group/update order and absence");
    }
}
