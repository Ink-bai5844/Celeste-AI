# Cassette mechanics handoff — 2026-11-12

## Delivery status

`celeste_search/cassette.py` is a **standalone, unintegrated reference-derived
physics module**. It does not change `physics.py`, `runtime_state.py`, planner
capability flags, or the running game. Eleven targeted offline tests pass.
No native cassette-room trajectory has been verified: the running Windows
bridge does not export the new clock/group fields and was deliberately not
replaced or restarted. Missing telemetry must continue to reject planning.

The C# telemetry work is separate. See
`../../artifacts/bridge-cassette-staging/BUILD-VERIFICATION.md`. Its staged binary
is a Windows XNA/.NET 4.5 build, **not a tested Mac binary**. Rebuild/port the
instrumentation for the destination game's actual runtime before deploying it.

The last 11→11z selected-boundary crossing succeeded, but the later neutral
tail died. This is evidence that boundary clearance alone does not certify a
safe handoff; it is not evidence that cassette collision simulation works.
The controller's terminal-entry/handoff work is independent of this module.

## What the module implements

* Native Single rounding, 1/6-second subdivision, eight subdivisions per color,
  pre-toggle at subdivision 7, music startup's skipped first tick, lead-in reset,
  and the native single-if behavior for a large delta (not a catch-up loop).
* Actual per-block `Activated` versus `Collidable`, current world position,
  platform remainder/lift speed, and `blockHeight`.
* Atomic group activation only after every member's obstruction check. Checks
  run in the observed native group-list order; earlier successful actor wiggles
  remain even if a later member blocks activation. Activation retries on later
  block updates without waiting for another musical beat.
* The special group overlap check four pixels below another member; player
  upward wiggle of 1–4 pixels; real one-pixel pre-toggle/activation/deactivation
  platform shifts; rider carry, overlap push, crouch/squish escape and native
  0.16-second lift grace representation (10 simulation ticks).
* Ordered `shift`, `enable`, and `disable` events for attached static movers.
  Emitting events is not the same as applying attached spike/spring physics.
* Frozen/hashable clock and block values, strict dictionary round trips, and
  rejection of missing phase/group/order fields rather than invented phase 0.

## Interface and integration contract

```python
from celeste_search.cassette import (
    CassetteClock, CassetteBlockState, CassetteActor, step_cassette,
)

clock = CassetteClock.from_dict(room_entities["cassette_manager"])
blocks = tuple(CassetteBlockState.from_dict(e) for e in cassette_entities)
# Construct CassetteActor from the exact current player's corresponding fields.
before = step_cassette(clock, blocks, actor,
    solid_rects=other_solids, jumpthru_rects=jump_throughs,
    before_order=player_update_order)
# Perform the existing Player.Update against before.blocks' current collisions.
# Reconstruct actor from that resulting full SimState; never reset player state.
after = step_cassette(before.clock, before.blocks, updated_actor,
    solid_rects=updated_other_solids, jumpthru_rects=jump_throughs,
    after_order=player_update_order)
```

`step_cassette` returns `CassetteStep(clock, blocks, actor, events)`. Order bounds
are **open**; preserve the returned clock/blocks between calls so the manager
does not tick twice. `dt` defaults to 1/60 and is rounded as native Single.
`advance_clock(clock, dt)` is also exposed for isolated reference tests.

Required `SimState` additions (not made in this delivery):

* `cassette_clock: CassetteClock | None = None`
* `cassette_blocks: tuple[CassetteBlockState, ...] = ()`

Include complete values in state serialization, cloning, frontier keys, caches,
replay comparisons, drift checks and live re-anchoring. Do not quantize the
subdivision phase away. Use `None` for unknown, never an initialized clock.

Required manager telemetry: `present`, `current_index`, `beat_timer`,
`beat_index`, `tempo_mult`, `lead_beats`, `max_beat`, `is_level_music`,
`beat_index_offset`, `active`, `music_ready`, `update_order`.

Required block telemetry: stable `id`, `index`, `tempo`, actual `x/y`,
`width/height`, `activated`, `collidable`, `active`, `block_height`,
`group_leader`, native-order `group_ids`, `update_order`, `rem_y`, `lift_x/y`.
The bridge also exports `mode` for audit; this native version does not use it
in the block's physical update. Native collider size does **not** shrink with
`blockHeight`; `ShiftSize` calls `MoveV` and changes a visual side height.

`CassetteActor` contains feet `x/y`, `ducking`, player `state`, `facing`,
`climb_trigger_dir`, `dead`, `rem_y`, `lift_vx/vy`, `lift_timer`,
`allow_pushing` and `ignore_jumpthrus`. Merge only these affected values back
into the preserved full SimState. Speeds, dash resources, stamina, collection,
and unrelated mechanisms must not be reinitialized. Actor y wiggles are direct
position changes; collision-blocked exact movement clears the Y remainder.

## Remaining integration boundaries

1. Add strict runtime ingestion of the new telemetry and observe the actual
   player/manager/block update order. Unknown order must not be guessed.
2. Insert collidable cassette rectangles into all relevant body, wall, dash,
   ground, climb, hazard and moving-solid queries. Existing exact-collision
   caches must include the complete cassette geometry/phase state.
3. Supply all other current solid and jump-through rectangles to this module.
   Other dynamic entities may be interleaved in native update order; the module
   does not reorder or simulate them. Initially reject unsupported interleavings
   rather than assume every mechanism updates after the player.
4. Apply `shift`/`enable`/`disable` to real attached spike/spring state in order,
   including possible immediate hazard effects. Reject these attachments until
   integrated; do not render inactive attached spikes as live hazards.
5. Theo/other actor obstruction is not implemented. The native block checks
   Theo before Player. Reject rooms/anchors containing relevant unsupported
   actors instead of treating the player-only update as complete.
6. Native manager survives transitions, and its OnLevelStart may silently
   choose a color from existing beat phase. Camera/transition updates are not
   one-to-one with action frames. This module neither resets nor extrapolates
   that time. Re-anchor from fresh complete telemetry after a transition;
   internal multi-room continuation into an unknown phase remains unsupported.
7. Cassette pickup is **not implemented**: the native fixed 16×16 collider
   starts a freeze/cutscene, changes respawn/save state, stops blocks and later
   removes the manager. Its visible hover moves only the sprite, not collider.
   An implementation may allow verified avoidance and terminate candidates as
   `unsupported_interaction` on contact; do not treat pickup as an ordinary
   strawberry or assume successful collection.
8. Add end-to-end simulator/controller tests and native per-frame parity after
   loading a compatible bridge. The current eleven tests exercise the isolated
   module against source-derived cases, not actual game telemetry recordings.

## Verification

From `Celeste-Search`:

```sh
python -m unittest discover -s tests -p test_cassette.py -v
```

Eleven tests cover strict telemetry/hash round trips, beat boundaries and
float32 behavior, startup/lead-in, actual pre-toggle carry, activation wiggle,
whole-group deferral/retry, group overlap veto, split update order, real disable
motion, crouch escape and rejection of incomplete groups.

Native reference was decompiled locally from `GAME/Celeste-2/Celeste.exe` into
`artifacts/cassette-audit-native/`: `Celeste.CassetteBlock`,
`Celeste.CassetteBlockManager`, `Celeste.Cassette`, `Celeste.Platform`,
`Celeste.Solid`, `Celeste.Actor`, `Celeste.Player`. These are audit references;
the game binary is not required in the handoff package and must be excluded.
There is no `Celeste.CassetteListener` type in this native assembly; the empty
failed-decompilation file, if present, is not an implementation reference.
