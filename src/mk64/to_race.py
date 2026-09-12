"""Drive the menus from power-on to the start line of a Time Trial.

Time Trials on purpose: no opponents, no items, and a staff ghost to be measured
against — which is what SPEC.md calls "solved".

Nothing here is learned. It runs once and its output is a savestate; after that
every episode starts with load_state() and no menu is ever touched again.

The press count is the whole trick: MK64 puts an "OK ?" confirm after *each*
selection, so it is five A presses from PLAYER SELECT to the track, not four.
Four leaves you on MAP SELECT looking at a screen that appears to ignore input.
"""

from .menu import tap

#: A presses from PLAYER SELECT to the start line, in order.
_A_PRESSES = (
    "choose MARIO",          # -> OK ? prompt
    "confirm MARIO",         # -> MAP SELECT
    "choose LUIGI RACEWAY",  # -> OK ? prompt
    "confirm course",        # -> OK ? again
    "start the time trial",  # -> on the track
)


def to_player_select(env):
    env.step(frames=400)                                     # boot -> title
    tap(env, hold=6, wait=90, start=True)
    tap(env, hold=6, wait=90, start=True)                    # -> GAME SELECT
    tap(env, hold=4, wait=40, down=True)                     # MARIO GP -> T.TRIALS
    tap(env, hold=4, wait=60, accel=True)                    # confirm mode
    tap(env, hold=4, wait=60, accel=True)                    # focus BEGIN
    tap(env, hold=4, wait=90, accel=True)                    # enter BEGIN
    return env


def to_start_line(env, countdown=260, verbose=False):
    """Power-on -> Mario on the Luigi Raceway start line, countdown finished."""
    to_player_select(env)
    env.pad.neutral()
    env.step(frames=60)
    for what in _A_PRESSES:
        tap(env, hold=5, wait=150, accel=True)
        if verbose:
            print(f"  A: {what} (frame {env.frame})")
    # Let Lakitu finish the lights, so the saved state is one the agent can
    # simply drive out of.
    env.pad.neutral()
    env.step(frames=countdown)
    return env
