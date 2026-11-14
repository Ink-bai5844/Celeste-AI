using System;
using System.Collections.Generic;
using Instrumentation;
using Microsoft.Xna.Framework;

// Minimal data-only stand-ins. The production observation builder is compiled
// unchanged; these tests never load Celeste or call the game simulation.
#pragma warning disable 0414, 0649, 0169
namespace Microsoft.Xna.Framework
{
    public struct Vector2
    {
        public float X, Y;
        public Vector2(float x, float y) { X = x; Y = y; }
    }
}
#if !CELESTE_FNA
namespace System.Web.Script.Serialization
{
    public class JavaScriptSerializer
    {
        public string Serialize(object value) { return System.Text.Json.JsonSerializer.Serialize(value); }
    }
}
#endif
namespace Monocle
{
    public class VirtualButton
    {
        public bool Check { get; set; }
        private float bufferCounter;
        private bool consumed;
        public VirtualButton(float seconds, bool used) { bufferCounter = seconds; consumed = used; }
    }
}
namespace Celeste
{
    public class Session { public string Level = "room-a"; public List<EntityID> DoNotLoad = new List<EntityID>(); }
    public class Level : Monocle.Scene { public Session Session = new Session(); }
    public class StateMachine { public int State; }
    public enum Facings { Left = -1, Right = 1 }
    public class EntityID { public string Level; public int ID; }
    public class Follower { public object Entity; public float DelayTimer; public Leader Leader; }
    public class Leader { public List<Follower> Followers = new List<Follower>(); }
    public class Strawberry : Monocle.Entity
    {
        public EntityID ID = new EntityID { Level = "room-a", ID = 17 };
        public Follower Follower = new Follower { DelayTimer = 0.3f };
        private float collectTimer = -0.15f;
        public bool Winged, Golden, WaitingOnSeeds;
        private bool collected, flyingAway;
        private float flapSpeed;
    }
    public static class Input
    {
        public static int Gamepad;
        public static Monocle.VirtualIntegerAxis MoveX = new Monocle.VirtualIntegerAxis(), MoveY = new Monocle.VirtualIntegerAxis();
        public static Monocle.VirtualButton Grab = new Monocle.VirtualButton(0, false);
        public static Monocle.VirtualButton Jump = new Monocle.VirtualButton(0.08f, false);
        public static Monocle.VirtualButton Dash = new Monocle.VirtualButton(0.08f, true);
    }
    public class Actor : Monocle.Entity
    {
        private Vector2 movementCounter = new Vector2(0.25f, -0.375f);
        private float liftSpeedTimer = 0.1f;
        public Vector2 LiftSpeed = new Vector2(20, -30);
    }
    public class Player : Actor
    {
        public float Stamina = 110;
        public Player() { X = 24; Y = 160; }
        public Vector2 Speed = new Vector2(90, -105), DashDir = new Vector2(1, 0);
        public int Dashes = 1;
        public bool Dead, Ducking, AutoJump = true;
        public float AutoJumpTimer;
        public Facings Facing = Facings.Right;
        public StateMachine StateMachine = new StateMachine();
        public Leader Leader = new Leader();
        public bool OnGround(int distance) { return false; }
        private Vector2 beforeDashSpeed = new Vector2(123, 0);
        private float varJumpSpeed = -105, jumpGraceTimer = 0.1f, varJumpTimer = 0.2f;
        private float dashCooldownTimer = 0.2f, dashRefillCooldownTimer = 0.1f;
        private float forceMoveXTimer = 0.16666667f, wallSlideTimer = 1.2f, climbNoMoveTimer = 0.1f;
        private int forceMoveX = 1;
        private bool dashStartedOnGround = true;
        private float maxFall = 160;
        private float dashAttackTimer = 0.3f, wallSpeedRetentionTimer = 0.06f, wallSpeedRetained = 90, hopWaitXSpeed = 120;
        private int hopWaitX = 1;
    }
}

static class ModelStateTests
{
    static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    public static void Run()
    {
        Require(SearchModelState.TimerFrames(0.1f) == 6, "Float noise must not add a timer tick");
        Require(SearchModelState.TimerFrames(-0.15f, true) == -9, "Collection stagger must remain negative");
        Require(SearchModelState.TimerFrames(-0.15f) == 0, "Countdown timer must clamp at zero");
        InputsManager.Initialize();
        InputsManager.ApplyPlaybackInput(17);
        Celeste.Player player = new Celeste.Player();
        Celeste.Strawberry berry = new Celeste.Strawberry();
        player.Leader.Followers.Add(new Celeste.Follower { Entity = berry });
        SearchModelState model = SearchModelState.Capture(player, new Celeste.Level(), new List<string>(), new List<string> { "room-a:17" }, false);
        Require((float)model.Data["rem_x"] == 0.25f && (float)model.Data["rem_y"] == -0.375f, "Inherited private subpixels missing");
        Require((int)model.Data["prev_input"] == 17, "Held action mask missing");
        Require((int)model.Data["var_jump"] == 12, "Variable-jump timer mismatch");
        Require((int)model.Data["dash_attack"] == 18 && (int)model.Data["wall_retention"] == 4, "Collision grace timers missing");
        Require((float)model.Data["wall_retained"] == 90 && (int)model.Data["hop_wait_x"] == 1, "Retained wall/hop movement missing");
        Require((bool)model.Data["auto_jump_active"] && (int)model.Data["auto_jump"] == 0, "Indefinite AutoJump flag must survive its zero timer");
        Require(!model.Approximate.Exists(item => item.StartsWith("auto_jump:")), "AutoJump flag must not use an inferred expiry");
        Require((int)model.Data["jump_buffer"] == 5 && (int)model.Data["dash_buffer"] == 0, "Consumed buffer must not be rearmed");
        Require((int)model.Data["collect_ground"] == -9 && (int)model.Data["collect_delay"] == 18, "Collection timer units/order changed");
        Require(model.Unavailable.Contains("mechanisms"), "Missing moving-platform state must be explicit");
        string first = model.Fingerprint();
        model.Data["frame"] = 900L;
        Require(model.Fingerprint() == first, "Absolute command frame must not change anchor fingerprint");
        model.Data["x"] = 25f;
        Require(model.Fingerprint() != first, "Actual position must change anchor fingerprint");
        player.AutoJumpTimer = 0.1f;
        model = SearchModelState.Capture(player, new Celeste.Level(), new List<string>(), new List<string>(), false);
        Require((bool)model.Data["auto_jump_active"] && (int)model.Data["auto_jump"] == 6, "Timed AutoJump must preserve both fields");
        player.AutoJump = false;
        model = SearchModelState.Capture(player, new Celeste.Level(), new List<string>(), new List<string>(), false);
        Require(!(bool)model.Data["auto_jump_active"] && (int)model.Data["auto_jump"] == 6, "Inactive flag must not fabricate a cleared timer before the engine clears it");
        player.StateMachine.State = 2;
        model = SearchModelState.Capture(player, new Celeste.Level(), new List<string>(), new List<string>(), false);
        Require(!model.Data.ContainsKey("dash_frames") && model.Unavailable.Contains("dash_frames"), "Mid-dash coroutine cannot be fabricated");
        Console.WriteLine("PASS runtime anchor units, private/subpixel state, consumed inputs, collection staggering and fingerprint");
    }
}
