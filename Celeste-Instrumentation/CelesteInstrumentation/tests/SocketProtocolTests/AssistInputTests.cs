using System;
using System.Collections.Generic;
using Celeste;
using Instrumentation;
using Microsoft.Xna.Framework.Input;

namespace Monocle
{
    public class Binding
    {
        public List<Keys> Keyboard = new List<Keys>();
        public bool ControllerHeld;
        public Binding(params Keys[] keys) { Keyboard.AddRange(keys); }
        public bool Check(int gamepad, float threshold)
        {
            foreach (Keys key in Keyboard) if (MInput.Keyboard.Check(key)) return true;
            return ControllerHeld;
        }
    }
    public class VirtualIntegerAxis { public int Value; }
    public static class MInput
    {
        public static HashSet<Keys> PhysicalKeys = new HashSet<Keys>();
        public class KeyboardData
        {
            public bool Check(Keys key)
            {
                return InputsManager.PassThrough || InputsManager.ReadingPhysicalInput ? PhysicalKeys.Contains(key) : InputsManager.Check(key);
            }
        }
        public static KeyboardData Keyboard = new KeyboardData();
        public static object[] GamePads = new object[4];
    }
}

static class AssistInputTests
{
    static void Require(bool value, string message) { if (!value) throw new Exception(message); }
    public static void Run()
    {
        Settings loadedSettings = Settings.Instance;
        Settings.Instance = null;
        InputsManager.Initialize();
        Require(InputsManager.PassThrough && !InputsManager.GameplayBindingsAvailable, "Normal startup before settings load must remain safe and manual");
        Settings.Instance = loadedSettings;
        Require(InputsManager.GameplayBindingsAvailable, "Loaded game bindings must become available without reinitializing the type");
        Settings.Instance.Right.Keyboard[0] = Keys.Extra;
        Require(InputsManager.ByteToKeys(1)[0] == (int)Keys.Extra, "A settings remap must affect the next assisted command");
        Settings.Instance.Right.Keyboard[0] = Keys.Right;
        var jumpKeys = Settings.Instance.Jump.Keyboard;
        Settings.Instance.Jump.Keyboard = new List<Keys>();
        Require(!InputsManager.GameplayBindingsAvailable, "Unbound keyboard actions must be reported instead of crashing manual game startup");
        Settings.Instance.Jump.Keyboard = jumpKeys;
        InputsManager.Initialize();
        Require(InputsManager.PassThrough, "Initialization must leave human input in control");
        Monocle.MInput.PhysicalKeys.Add(Keys.Left);
        InputsManager.ApplyPlaybackInput(1 | 32);
        Require(!InputsManager.PassThrough, "A gameplay command must explicitly acquire control");
        AssistInputState.CaptureCurrent();
        Require(AssistInputState.HumanInputMask == 2 && !AssistInputState.HumanMenuInput, "Physical sampling must ignore injected right/dash input");
        Require(!InputsManager.PassThrough && !InputsManager.ReadingPhysicalInput, "Sampling must restore the input policy");
        long epoch = AssistInputState.HumanInputEpoch;
        AssistInputState.CaptureCurrent();
        Require(AssistInputState.HumanInputEpoch == epoch + 1, "Held physical input must invalidate an older anchor");
        Monocle.MInput.PhysicalKeys.Clear();
        Settings.Instance.Jump.ControllerHeld = true;
        AssistInputState.CaptureCurrent();
        Require(AssistInputState.HumanInputMask == 16, "Gamepad binding must be included in current physical input");
        Settings.Instance.Jump.ControllerHeld = false;
        Monocle.MInput.PhysicalKeys.Add(Keys.Escape);
        AssistInputState.CaptureCurrent();
        Require(AssistInputState.HumanInputMask == 0 && AssistInputState.HumanMenuInput, "Menu interruption must remain distinguishable from gameplay mask");
        Monocle.MInput.PhysicalKeys.Clear();
        epoch = AssistInputState.HumanInputEpoch;
        AssistInputState.CaptureCurrent();
        Require(AssistInputState.HumanInputEpoch == epoch, "Released input must not keep changing the human intervention epoch");

        InputsManager.PassPhysicalInput();
        Require(InputsManager.PassThrough && InputsManager.KeysDown.Count == 0 && InputsManager.PreviousKeysDown.Count == 0, "Releasing control must remove every synthetic key and edge");
        Input.MoveX.Value = -1;
        Input.MoveY.Value = -1;
        Input.Jump.Check = true;
        Input.Grab.Check = true;
        AssistInputState.ObserveEffectiveInput();
        Require(InputsManager.PlaybackInput == (2 | 4 | 16 | 64), "Previous input must describe actual virtual human input, never command 128");
        Require(InputsManager.PassThrough, "Observation must not acquire control");
        var previous = Input.MoveX;
        Input.MoveX = null;
        AssistInputState.CaptureCurrent();
        Require(!AssistInputState.Available, "Missing input initialization must be explicit, not fabricated neutral");
        Input.MoveX = previous;
        Input.MoveX.Value = Input.MoveY.Value = 0;
        Input.Jump.Check = Input.Grab.Check = false;
        InputsManager.Initialize();
        Console.WriteLine("PASS manual default/release, separate physical and virtual masks, controller/menu interference, and human-input epoch");
    }
}
