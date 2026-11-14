"""Guidance-only resource groups; these never decide physical reachability."""


def grip_resource_band(stamina):
    """Distinguish ordinary stamina ranges and further paid climb jumps.

    The vanilla model charges 27.5 for a climb jump and requires 20 to
    acquire a new grip. Waiting/climbing can spend more, while ordinary wall
    jumps can be free, so this is diversity guidance rather than a guarantee.
    """
    return (int(max(0., stamina)/20), max(0, int((stamina-20)//27.5)))
