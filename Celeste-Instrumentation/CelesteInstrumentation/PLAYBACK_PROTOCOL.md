# Celeste assistant and playback protocol

## Passive assistant startup

Set `<AssistMode>true</AssistMode>` in `InstrumentationParameters.xml` for the
human-operated assistant. The game starts through its original menus and save
selection. Instrumentation does not select debug play mode, call SaveData.Start,
replace the overworld with WaitingScene, skip native chapter introductions, or
load a chapter. With the first live player, a background connection offers
`LIVEINFO` for the existing native scene. The controller must acknowledge byte 0;
the first fresh `STATE` is the authoritative planning observation.
Native timing, render paths, spotlight wipes, cassette counts and autosplitter
updates are also preserved. The model must validate the reported native dt;
the legacy constant-delta patch is not applied to AssistMode.

Both `LIVEINFO` and `STATE` advertise `assist_protocol_version: 1`, and
`capabilities` containing `physical_input_passthrough`, `explicit_takeover`,
`native_room_respawn`, and `current_scene_attach`. A GUI must check these before
sending 128; older bridge builds reject that value. `assist_mode` identifies this
normal-startup configuration. The assistant server may use a separate configured
port, such as 25102, from the legacy autonomous controller.

New connections start in `control_mode: "manual"`. Reply 128 to keep physical
keyboard and controller input unchanged while receiving observations. Only a
confirmed user target may cause the controller to send a 0–127 action and acquire
control (`control_mode: "assist"`). Reaching the target, canceling, or failing
must return 128. Death releases synthetic input immediately and uses the native
respawn. Cutscenes, overlays and menus return manual control; they are not
automatically confirmed. Native room transitions can retain an admitted airborne
route. Both STATE replies and DONE acknowledgments have a 50 ms watchdog in
AssistMode; disconnect releases control and retries connection in the background.

`human_input_mask` is the **current hardware-sampled** gameplay binding mask,
including keyboard, controller, move-only, dash-only and demo-dash bindings.
`human_menu_input` separately reports menu/cancel/talk input. Null means that
input is not initialized and admission must fail. Any physical gameplay/menu
input wins over a requested assisted action on that frame. `human_input_epoch`
increments on every sampled frame with such input, allowing a prepared plan to
be rejected even if the human subsequently returns to the same coordinates.
`control_event` remains the most recent event and is not a current-key indicator.

`model_state.prev_input` is distinct: it records the effective MoveX/MoveY and
Jump/Dash/Grab held values immediately after the preceding virtual-input update.
It never contains command 128. Settings bindings are resolved lazily, including
remapping in the normal settings menu; missing keyboard gameplay bindings are
reported as `input_bindings_supported: false` and block assisted actions without
disabling manual play. `map_name` comes from the native area's mode Path, and
`camera_x`, `camera_y`, and `level_bounds` use game world coordinates. The GUI's
dedicated map view does not rely on screen scaling or letterboxing.

## Legacy isolated playback startup

This protocol executes routes found by an independent background simulator.
The live controller must respond promptly to every frame, using neutral or
already prepared input while its worker searches. Searching must not pause the
game. Deaths use the game's normal current-room respawn; they do not restart the
chapter or replay a prefix. Candidate evaluation remains offline. The bridge
does not train a model, teleport the player, or award berries.

The game connects as a TCP client to the configured Python server. The original
`LOADINFO` handshake is retained: the server responds with one UTF-8
`SessionParameters` XML document, including `<SearchMode>true</SearchMode>`.
The new field defaults to false, preserving the existing 137-float protocol.
Set `EndOnDeath=false` and `EndOnLevelExit=false` for continuous live play.
SearchMode plus EndOnDeath=false selects the live lifecycle. LOADINFO is used
once for initial chapter selection, not after each death.
`TimeoutSeconds` counts controllable updates at 60 updates per second within a
room, reset on a room change or a new player spawn; zero disables that limit.

With SearchMode enabled, no MAPINFO message or legacy float frame is sent.
Each message is one of:

| Message | Wire format | Required response |
| --- | --- | --- |
| Observation | ASCII `STATE`, 4-byte unsigned little-endian UTF-8 payload length, JSON | Exactly one action byte |
| Terminal | ASCII `DONE`, 4-byte unsigned little-endian UTF-8 payload length, JSON | Exactly one acknowledgment byte |

Action values 0–127 are bit masks: bit 0 right, 1 left, 2 up, 3 down,
4 jump, 5 dash, 6 grab. Values 129–252 are rejected. Special commands are:

| Byte | Live behavior |
| --- | --- |
| 128 | Keep the connection and observations, release synthetic keys, pass physical inputs through unchanged. |
| 253 | Send DONE with reason detached, release inputs/socket, leave the current game running for human control. |
| 254 | Request native current-room death/respawn via Player.Die; only for an explicitly detected stuck state. No DONE or LOADINFO follows. |
| 255 | In live mode, detach exactly as 253. Reserved older non-live behavior still stops its isolated playback session. |

254 uses the game's death action, including its real checkpoint. It is rejected
while carrying a non-winged golden berry because that item changes the native
death action into a chapter restart. `control_event` reports that rejection.
Ordinary deaths retain stock game behavior. TCP readers must read exact lengths;
a single receive call is not a message boundary.

The JSON object has schema_version `celeste-playback-state-v1`, room, x, y,
vx, vy, dashes, stamina, on_ground, on_safe_ground, state, dead, berries,
following, frame, and input_phase. Positions use the game's world-space player position;
positive Y points down. `state` is the game's Player.StateMachine.State integer.
`berries` is the sorted set of committed `Session.Strawberries` IDs, each
formatted `room:id`. `following` contains carried Strawberry IDs separately;
touching or carrying a berry does not confirm collection. `spawn_id` increments
for each distinct Player instance added, including the first spawn; room
transitions that retain the same Player do not increment it. `deaths` increments
only when the actual Player.Die method returns a non-null dead body, not merely
when death is attempted. `respawn_x`/`respawn_y` expose Session.RespawnPoint
(null if absent). `control_event` identifies spawn/death/respawn-request events.

DONE additionally includes a reason: detached, death, transition, timeout, stopped,
chapter_complete, unsupported_runtime, or ended. Terminal observations read current runtime values
rather than reusing a stale float frame. The chapter_complete hook runs after
the actual Level.CompleteArea method and its RegisterAreaComplete bookkeeping,
including carried-berry collection; unrelated cutscenes are not completion.
In live mode, DONE detaches in place after acknowledgment. chapter_complete
preserves the game's original completion scene/wipe. Natural death sends no
DONE: the same connection resumes STATE after the new player spawns. Disconnect,
invalid input, or controller timeout also releases inputs in place. Original
keyboard behavior and normal overworld/level loading resume after detachment;
the bridge does not force WaitingScene, load a chapter, or close the process.

**F12 is the physical emergency-detach key during live control.** It is read
directly from the keyboard before gameplay gating, bypassing the injected keys.
The bridge sends DONE with reason `user_detached`, releases the controller, and
leaves the game running for the user. It does not reconnect automatically. This
also works during natural death/respawn using the most recent player observation;
before the initial player exists it simply releases the connection.

## Input timing and verification limits

STATE is emitted in a prefix to `MInput.UpdateVirtualInputs`, after the engine
sets delta time and before virtual inputs or scene physics update. Its
`input_phase` is `before_virtual_inputs`; DONE uses `terminal`. The received
mask installs the keyboard state immediately, then the original virtual-input
update runs once. Thus movement axes, aim, button presses, and buffering all see
the same new command in the upcoming controllable update. SearchMode bypasses
the older Player.Update input swap and the mid-Player.Update tracker exchange.
No virtual input is manually updated a second time.

`frame` counts accepted commands (including 128 manual frames), starting at zero and continuing across rooms
and deaths. A 254 native-respawn request also advances this command counter once.
STATE frame N observes the result after N commanded controllable updates and
requests command N. It is not an absolute engine-frame counter. Engine freezes,
room transitions, pauses, overlays, inactive/dead players, and cutscenes do not
consume route commands. Held keys persist during these ticks; their previous
state advances so Pressed is not repeated. A private unpause timer is checked;
an incompatible runtime without that field stops as unsupported_runtime.

Prepared action segments compare expected state directly at the documented
pre-input boundary; no scheduling-lag window or neutral flush is needed. A new
spawn must cancel predictions and pending routes from the previous player.
When the controller finishes or is interrupted it sends 253, acknowledges DONE,
and leaves the game process running. Live detachment does not suppress the
remaining native scene update.

This ordering follows the inspected decompilation:
[Engine.Update](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Monocle/Engine.cs),
[MInput.Update](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Monocle/MInput.cs),
[Player.Update](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Player.cs),
and [Level.Update](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Level.cs).

Input lists are reset at session start/end. RNG state is intentionally unchanged;
this bridge does not promise deterministic replay or exact simulator parity.
SearchMode uses the pre-existing isolated session setup and SaveData slot 666;
it does not merge results into the user's ordinary save slot. Build/protocol
checks do not establish successful in-game playback, collision parity, room
coverage, or collectible completion.

## Runtime model anchors and native respawn

Each observation includes `model_state`, a dictionary using the offline
`SimState` names. It contains actual room/position/velocity, inventory, confirmed
and carried berries, facing, ducking, previous action mask, actor movement
remainders (`rem_x`, `rem_y`), jump/variable-jump/force-movement/climb/dash-refill
timers, current dash direction and pre-dash horizontal speed, maximum fall speed,
lift speed/grace, and follower order/collection timers. The bridge reads these
values and never writes them into the game.

Countdown timers use integer nominal 60 Hz updates:
`max(0, ceil(seconds * 60 - 0.0001))`. `collect_ground` preserves its sign because
later berries can have a negative collection stagger. `runtime_timers_seconds`
also preserves the original floating-point seconds. Jump/dash buffers already
consumed by the game export as zero. `auto_jump_active` directly exports
`Player.AutoJump`, and `auto_jump` independently exports `AutoJumpTimer` in frames.
A true flag with a zero timer means indefinite AutoJump; it must not expire on
the next model step or be replaced by the variable-jump timer. A positive timer
expires according to the game's update logic, while gameplay events such as
ground contact can clear the flag separately.

`model_state_unavailable` lists omitted state instead of inventing defaults.
In particular a running dash's coroutine position cannot be derived from its
attack timer: state 2 omits `dash_frames`, so a controller must not treat it as a
complete dash anchor. Moving-platform phases are not inferred from geometry.
Entity observations are available separately in `room_entities`, described
below; these are not silently converted into fabricated fresh model states.
A controller must distinguish a fresh room from a revisit or a mid-room anchor.
`model_state_approximate` describes conversions
that are not exact engine state, including aggregate first-follower counters.

`state_fingerprint` is SHA-256 over sorted model fields, raw timer values and the
unavailable list, excluding absolute command frame. The declared
`fingerprint_scope` excludes entity coroutines and RNG. It is useful for comparing
replayed anchors, but neither a savestate nor proof of complete engine equality.

STATE's short network round trip occurs on the engine update thread, so the
controller must keep the reply path lightweight and perform all search in a
background worker. It must never hold a reply while waiting for search results.
TCP_NODELAY avoids batching delays. Live read/write operations have a one-second
watchdog; an unresponsive client is detached rather than blocking indefinitely.
This is a recovery deadline, not an allowed search pause.

Diagnostics expose `engine_frame`, `delta_time`, `raw_delta_time`,
`previous_round_trip_ms`, and `max_round_trip_ms`. Round-trip times measure the
preceding successful STATE exchange, excluding JSON capture. The bridge keeps
the existing fixed update/input boundary and constant nominal 1/60 raw delta;
an engine scheduler can catch up after wall-clock stalls, so command frames
must not be inferred from elapsed wall-clock time.

After natural death, wait for STATE with a new spawn_id, then re-anchor the
background model at the actual checkpoint. If stuck, 254 can request the same
native death/respawn path. There is no chapter reload, prefix replay, coordinate
reset, private-state write, or simulated savestate restoration in live mode.

## Current-scene reconnection

A lost live socket releases playback inputs and enables a background connection
attempt once per second. Game updates and human input continue while it waits.
The worker uses bounded connect/handshake deadlines, and never accesses engine
objects. The main thread captures one advisory snapshot before starting it.

The reconnect handshake is eight ASCII bytes `LIVEINFO`, uint32 little-endian
UTF-8 JSON length, then JSON using the STATE shape, including `area_id` and
`area_mode`. The controller must reply with byte 0 to accept; any other reply
declines. After acceptance the game thread adopts the socket, and the next
message is a fresh STATE. The handshake snapshot may have aged while connecting;
only the fresh STATE should anchor a plan. Playback command/death/spawn counters
are preserved, and the handshake never calls `BeginSession`, `SaveData.Start`,
`LevelLoader`, a pause method, or a respawn method.

In live mode, cutscenes/overlays clear gameplay keys. When an open native Textbox
specifically reports `waitingForInput`, the bridge waits 15 updates and pulses
the ordinary Confirm binding for one update, repeating only while the textbox
continues waiting. This honors Textbox's initial 0.1-second input gate. It never
sends Cancel, skips a scene, or confirms a generic menu; paused levels are
excluded, and Confirm bindings shared with Pause are rejected. These native
dialogue updates do not consume gameplay command frames.

Explicit detach commands 253/255 and physical F12 cancel automatic reconnect.
Physical F11 opts back into attaching to the unchanged current scene. An F12
press also cancels a pending handshake and closes its result when it finishes.
`reconnect_attempts` and `reconnects` are diagnostic counters. `LOADINFO` retains
its original initial-session semantics and must not be used for reattachment.

## Runtime entity observations

`room_entities` has schema `celeste-room-entities-v1`, `entities`,
`do_not_load`, `coverage`, and `unavailable`. Constructor Harmony postfixes
capture actual `EntityData.ID`, `EntityData.Level.Name`, kind and world-space map
origin for strawberries, refills, dash blocks, zip movers, falling blocks,
crumble platforms, spikes, springs, and jump-throughs where that runtime exposes
an EntityData constructor. Positions are never used to guess identity. Dynamic
entities that lack those constructors are explicitly outside coverage.

Every identified entity exports `id` (`room:id`), `room`, `entity_id`, `kind`,
`runtime_type`, `map_x`, `map_y`, current `x`, `y`, `left`, `top`, `width`,
`height`, `collidable`, `active`, and `present`. Removed entities remain recorded
with `present=false`; native respawn replaces records when constructors run
again. `retained_observation=true` means the object was garbage-collected and
its last available observation is retained. `do_not_load` is the actual session
ID set and is not itself interpreted as a list of broken dash blocks.

Refills include actual `respawn_timer`, `one_use`, and `two_dashes`. Strawberries
include `collected`, `has_leader`, `collect_timer`, `follow_delay_timer`, `winged`,
`golden`, `flying_away`, `flap_speed`, and `waiting_on_seeds`. Zip movers include
`percent` and target coordinates; falling blocks include `triggered`,
`fall_delay`, and `has_started_falling`. Platforms include movement remainders
and lift speed. All entity timer values are raw seconds, unlike the frame
counters in `model_state`. Per-entity `unavailable` lists missing reflected data.

The staged cassette telemetry addition registers `CassetteBlock` EntityData
constructors. Each block also exports `index`, `tempo`, `activated`, numeric
`mode`, `block_height`, `group_leader`, and `group_ids`. The group IDs preserve
the native list order: activation can wiggle the player for one member before
a later member blocks the group. `Activated` is a requested state and can differ
from actual `collidable`; `mode` is retained for auditing, not assumed to be the
physical phase. Current `x`, `y` and collider bounds must be used because native
`ShiftSize` physically moves the block, including the initial two-pixel shift.

`room_entities.cassette_manager` is captured directly from the scene tracker,
even though the manager has no EntityData constructor. It always includes
`present` and `unavailable`; when present it adds `active`, `current_index`,
`beat_timer` (raw Single seconds), `beat_index`, `tempo_mult`, `lead_beats`,
`max_beat`, `is_level_music`, `beat_index_offset`, and `music_ready`.
`music_ready` means `isLevelMusic || sfx != null`: the first frame creating a
private music instance does not advance the native clock. A missing manager
does not export invented zero-valued clocks. Reflection failures are named in
`unavailable`, not replaced by guessed phase values.

Each observed entity and the cassette manager export `update_order` in the
native `Scene.Entities` sequence; `room_entities.player_update_order` locates
the player in that same sequence. These indices are independent of the sorted
output entity IDs and can change when the scene changes. Missing or removed
entities have a null index. Consumers must retain actual ordering when modeling
activation and player displacement. `Celeste.Freeze(time)` also calls
`CassetteBlockManager.AdvanceMusic(time)`, so player-update frame counts alone
cannot determine the music phase.

These new fields require loading the rebuilt C# bridge. The current bridge
has no assembly-loading, reflection-query, or hot-reload socket command;
restarting the Python assistant cannot add fields to an already-running game.
The cassette build is isolated in `artifacts/bridge-cassette-staging` and was
not copied into the running game. Its compilation and data-only protocol tests
do not constitute native runtime validation. Cassette pickup/cutscene behavior
remains unsupported and is not made executable by observing its blocks.

Main mechanism/refill/fly-away `coroutines` export `active`, `finished`,
`wait_timer` in seconds, and `stack` in top-first order. Each stack item provides
`iterator_type` and its primitive/enum/vector `fields` under the actual compiler
field names. Object references and decorative tile/outline coroutines are
excluded. These raw observations are exact read-only values, but interpreting
compiler states is runtime-version dependent: `coroutine_phase` remains marked
unavailable until the independent model verifies that mapping.

The additional player anchor fields `dash_attack`, `wall_retention` are nominal
frame counters; `wall_retained`, `hop_wait_x`, and `hop_wait_x_speed` export the
corresponding movement state. All observation additions are read-only.

Source mappings follow [EntityData](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/EntityData.cs),
[Refill](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Refill.cs),
[Strawberry](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Strawberry.cs),
[ZipMover](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/ZipMover.cs),
[FallingBlock](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/FallingBlock.cs),
and [Coroutine](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Monocle/Coroutine.cs).

## Building and checks

The project defaults `CelesteGameDir` to the repository's `GAME/Celeste-1` and
`XnaAssemblyRoot` to the Windows GAC_32 directory. Both MSBuild properties can be
overridden, for example `-p:CelesteGameDir=D:\Games\Celeste`. The output stays
`bin/Release/net4.5/CelesteInstrumentation.dll`.

Run `dotnet run --project tests/SocketProtocolTests/SocketProtocolTests.csproj
-c Release` from this directory for independent loopback/framing and input-edge
tests. They compile the production socket/input classes with minimal stubs and
do not load or launch Celeste.
