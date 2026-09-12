"""Driving MK64's menus, with eyes.

Menus are not part of the learning problem — they are the toll you pay once to
reach the start line. The output of this module is a committed savestate; after
that, every episode begins with load_state() and no menu is ever touched again.
"""


def tap(env, hold=4, wait=24, shot=None, **button):
    """Press something, release it, let the menu animate, optionally look."""
    env.step(frames=hold, **button)
    env.pad.neutral()
    env.step(frames=wait)
    if shot:
        return env.screenshot(name=shot)
    return None


def wait(env, frames=60, shot=None):
    env.pad.neutral()
    env.step(frames=frames)
    if shot:
        return env.screenshot(name=shot)
    return None
