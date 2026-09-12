"""MK64Env — a frame-stepped Mario Kart 64 with working rollback.

The contract this file exists to provide:

    env.step(steer=..., accel=...)   advance exactly one emulated frame
    s = env.save_state()             snapshot  (~microseconds, in RAM)
    env.load_state(s)                restore   == env.clone() for search

Everything here is game-agnostic: no lap counters, no kart addresses. What a
position *is* gets decided in an adapter on top, once ram_hunt has found it.

Threading note: CoreDoCommand(M64CMD_EXECUTE) blocks for the lifetime of the
emulator, so it runs on a worker thread while the main thread drives it with
PAUSE/ADVANCE_FRAME. The frame callback signals an Event so step() returns only
once a frame has genuinely been emulated, rather than guessing with sleeps.
"""

import os
import threading
import time
from pathlib import Path

from . import core as m64
from .pad import ShmPad

# MK64's framebuffers live inside RDRAM (origins seen around 0x31fa00, 0x345200,
# 0x36aa00). glide64mk2 renders through an FBO and does not write those pages
# back byte-identically, so ANY state comparison that includes them reports a
# working emulator as non-deterministic. Compare game state over this window
# instead -- it is below every framebuffer the game uses.
STATE_WINDOW = (0x000000, 0x300000)

DEFAULT_ROM = os.environ.get(
    "MK64_ROM", "/home/shiamjuniorchuttoo/Lets Burn Tokens/MK64/Mario Kart 64 (USA).z64")


class MK64Env:
    def __init__(self, rom=DEFAULT_ROM, headless=True, verbose=False,
                 state_dir=None):
        self.rom = str(rom)
        if not Path(self.rom).is_file():
            raise FileNotFoundError(f"ROM not found: {self.rom}")
        self.headless = headless
        self.frame = 0
        self._frame_evt = threading.Event()
        self._exec_thread = None
        self._running = False
        self._states = {}
        self._next_id = 0

        self.state_dir = Path(state_dir or "/dev/shm/mk64_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # The pad must exist before the plugin starts, or the plugin will
        # initialise the segment itself and clear `present`.
        self.pad = ShmPad()

        self.core = m64.Core(verbose=verbose)
        # Turn off the 60 fps limiter: training wants frames as fast as the CPU
        # will give them. (Set after EXECUTE starts; see _start.)
        self.core.config_set("Core", "SaveStatePath", str(self.state_dir))
        self.core.config_set("Core", "OnScreenDisplay", False)
        # The Game Boy Camera capture backend defaults to "opencv", which
        # blocks on a camera that is not there and hangs ROM startup before
        # the CPU ever runs. MK64 has no transfer pak; switch it off.
        self.core.config_set("Core", "GbCameraVideoCaptureBackend1", "")
        self.core.open_rom(self.rom)

        plugin_dir = Path(__file__).resolve().parents[2] / "build"
        gfx_dir = "/usr/lib/x86_64-linux-gnu/mupen64plus"

        # ORDER MATTERS, and the core only whispers about it: plugin_connect()
        # warns "Front-end bug: plugins are attached in wrong order" and then
        # carries on with a half-wired RSP. It must be GFX -> AUDIO -> INPUT ->
        # RSP, because plugin_connect_rsp() hands the RSP the *current* gfx
        # function pointers. Attach RSP first and it binds to dummy_gfx: MK64
        # then boots, renders framebuffer 0x27f forever, never fires a second VI
        # and never reads the controller again. Cost me an afternoon.
        #
        # The same fact kills true headless operation: with no video plugin the
        # RSP gets dummy_gfx and MK64 wedges identically. The RDP has to be real.
        # So "headless" here means offscreen-ish (a window we ignore), not
        # plugin-less; observations still come from RDRAM, never from pixels.
        self.core.attach_plugin(m64.M64PLUGIN_GFX,
                                f"{gfx_dir}/mupen64plus-video-glide64mk2.so")
        # (audio deliberately left unattached -> core's dummy_audio; silent and
        #  fast, and unlike video, nothing in the game waits on it.)
        self.core.attach_plugin(m64.M64PLUGIN_INPUT,
                                str(plugin_dir / "mupen64plus-input-shm.so"))
        self.core.attach_plugin(m64.M64PLUGIN_RSP,
                                f"{gfx_dir}/mupen64plus-rsp-hle.so")

        self.core.set_frame_callback(self._on_frame)
        self._start()

    # --- lifecycle -------------------------------------------------------
    def _on_frame(self, idx):
        self.frame = idx
        self._frame_evt.set()

    def _start(self):
        self._exec_thread = threading.Thread(target=self.core.execute,
                                             name="m64-execute", daemon=True)
        self._exec_thread.start()
        # Wait for the emulator to actually be running before pausing it.
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if self.core.emu_state() == m64.M64EMU_RUNNING:
                break
            time.sleep(0.01)
        else:
            raise m64.M64Error("emulator did not reach RUNNING state; core log:\n"
                               + "\n".join(self.core.log[-15:]))
        self._running = True
        try:
            self.core.set_speed_limiter(False)
        except m64.M64Error:
            pass                      # not fatal, just slower
        self.core.pause()

    def close(self):
        if self._running:
            try:
                self.core.stop()
            except m64.M64Error:
                pass
            if self._exec_thread:
                self._exec_thread.join(timeout=5.0)
            self._running = False
        self.core.detach_all()
        self.pad.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- stepping --------------------------------------------------------
    def step(self, frames=1, **action):
        """Apply `action` and advance `frames` emulated frames."""
        if action:
            self.pad.set(**action)
        for _ in range(frames):
            self._frame_evt.clear()
            self.core.advance_frame()
            if not self._frame_evt.wait(timeout=5.0):
                raise m64.M64Error(
                    f"frame did not advance (frame={self.frame}, polls={self.pad.polls}); "
                    "core log:\n" + "\n".join(self.core.log[-10:]))
        return self.frame

    def run(self, frames, **action):
        """Hold one action for N frames — the usual way to skip menus."""
        return self.step(frames=frames, **action)

    # --- rollback --------------------------------------------------------
    def save_state(self):
        """Snapshot to /dev/shm and return a handle. This is env.clone()."""
        sid = self._next_id
        self._next_id += 1
        path = self.state_dir / f"s{sid:06d}.st"
        before = self.frame
        self.core.save_state_to(path)
        # A savestate is written by the emulator thread on the next frame edge,
        # so step once to make sure it has actually landed on disk.
        self.step(1)
        if not path.exists():
            raise m64.M64Error(f"savestate did not appear at {path}")
        # The pad is part of the environment's state even though it lives
        # outside the emulator: restoring RAM but not the controller means the
        # settling frame replays a different input and the rollback is not exact.
        self._states[sid] = (path, before, self.pad.raw)
        return sid

    def load_state(self, sid):
        path, _, pads = self._states[sid]
        self.pad.restore(pads)
        self.core.load_state_from(path)
        self.step(1)       # the load is applied on the next frame edge
        return self.frame

    def drop_state(self, sid):
        path, _, _ = self._states.pop(sid, (None, None, None))
        if path and path.exists():
            path.unlink()

    # --- memory ----------------------------------------------------------
    def ram(self):
        return self.core.rdram()

    def ram_bytes(self, addr=0, length=None):
        r = self.core.rdram()
        return bytes(r[addr:addr + length] if length else r)

    def game_state_bytes(self):
        """RDRAM minus video memory — the right thing to hash or diff."""
        lo, hi = STATE_WINDOW
        return self.ram_bytes(lo, hi - lo)
