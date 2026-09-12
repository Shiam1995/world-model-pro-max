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

import gzip
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
                 state_dir=None, timeout=60.0):
        self.rom = str(rom)
        if not Path(self.rom).is_file():
            raise FileNotFoundError(f"ROM not found: {self.rom}")
        self.headless = headless
        # How long to wait for a single frame. Generous on purpose: this box
        # shares a CPU with other work (an Unreal shader compile pushed the load
        # average past 20 and starved the emulator thread), and a training run
        # that dies because a neighbour got busy is not worth having. A frame
        # taking seconds is slow, not broken.
        self.timeout = timeout
        #: How often a pause had to be re-asserted. Non-zero is fine; growing
        #: fast means something else is resuming the core.
        self.pause_nudges = 0
        self.frame = 0
        self._frame_evt = threading.Event()
        self._exec_thread = None
        self._running = False
        self._states = {}
        self._next_id = 0

        # Per-process savestate directory for the same reason as the pad: two
        # environments writing s000001.st into one directory will load each
        # other's worlds.
        self.state_dir = Path(state_dir or f"/dev/shm/mk64_states_{os.getpid()}")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # The pad must exist before the plugin starts, or the plugin will
        # initialise the segment itself and clear `present`. The plugin reads
        # MK64_INPUT_SHM, so point it at this process's segment before the core
        # loads it.
        self.pad = ShmPad()
        os.environ["MK64_INPUT_SHM"] = self.pad.name

        self.core = m64.Core(verbose=verbose)
        # Turn off the 60 fps limiter: training wants frames as fast as the CPU
        # will give them. (Set after EXECUTE starts; see _start.)
        self.core.config_set("Core", "SaveStatePath", str(self.state_dir))
        self.shot_dir = Path(__file__).resolve().parents[2] / "runs" / "shots"
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        self.core.config_set("Core", "ScreenshotPath", str(self.shot_dir) + "/")
        self.core.config_set("Core", "OnScreenDisplay", False)
        # The core jitters PI/SI interrupt timing on purpose ("RandomizeInterrupt",
        # default on) to shake out games that depend on exact timing. For us it
        # means the same actions from the same savestate do not replay
        # identically -- which would quietly poison every segment search.
        self.core.config_set("Core", "RandomizeInterrupt", False)
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
        self.pad.unlink()
        for f in self.state_dir.glob("*.st"):
            f.unlink(missing_ok=True)

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
            # Wait until the core has genuinely parked before asking for the
            # next frame. The frame callback runs INSIDE new_frame(), before
            # the core does its own `if (l_FrameAdvance) { g_rom_pause = 1;
            # l_FrameAdvance = 0; }` bookkeeping. Fire advance_frame() in that
            # window and the emulator thread immediately eats the flag we just
            # set, pauses anyway, and nothing ever resumes it — an intermittent
            # hang that looks like a slow frame. So the callback is a hint, not
            # the completion signal; PAUSED is the completion signal.
            self._await_paused()
            self._frame_evt.clear()
            self.core.advance_frame()
            if not self._frame_evt.wait(timeout=self.timeout):
                raise m64.M64Error(
                    f"frame did not advance (frame={self.frame}, "
                    f"polls={self.pad.polls}, state={self.core.emu_state()}); "
                    "core log:\n" + "\n".join(self.core.log[-10:]))
        return self.frame

    def _await_complete_state_file(self, path, max_frames=240):
        """Block until `path` is a whole, decompressable savestate."""
        last = -1
        for _ in range(max_frames):
            if path.exists():
                size = path.stat().st_size
                if size == last and size > 0:
                    try:
                        with gzip.open(path, "rb") as fh:
                            while fh.read(1 << 20):
                                pass
                        return size
                    except (OSError, EOFError):
                        # A truncated gzip raises EOFError, which is NOT an
                        # OSError — catching only OSError lets the "still being
                        # written" case escape as a hard failure.
                        pass
                last = size
            self.step(1)
        raise m64.M64Error(
            f"savestate at {path} never finished writing "
            f"(stuck at {last} bytes)")

    def _pump(self, event, what, max_frames=120):
        """Advance frames until the core signals `event`."""
        for _ in range(max_frames):
            if event.is_set():
                return
            self.step(1)
        if not event.is_set():
            raise m64.M64Error(f"{what} never completed within {max_frames} frames")

    def _await_paused(self, timeout=None):
        """Make the core paused, rather than hoping it becomes paused.

        Waiting passively is not enough. A PAUSE issued at the wrong moment can
        be swallowed — the emulator then free-runs, and the next step() sits
        there until it times out. It cost a whole search run: the core had run
        12,178 frames on its own before anyone noticed. So re-assert the pause
        while waiting; M64CMD_PAUSE is idempotent and takes effect at the next
        VI, which is microseconds away.
        """
        deadline = time.time() + (timeout or self.timeout)
        nudged = 0
        next_nudge = time.time()
        while time.time() < deadline:
            st = self.core.emu_state()
            if st == m64.M64EMU_PAUSED:
                self.pause_nudges += nudged
                return
            if st == m64.M64EMU_STOPPED:
                raise m64.M64Error(
                    f"emulator stopped mid-run at frame {self.frame}; core log:\n"
                    + "\n".join(self.core.log[-8:]))
            if time.time() >= next_nudge:
                try:
                    self.core.pause()
                except m64.M64Error:
                    pass
                nudged += 1
                next_nudge = time.time() + 0.05
            time.sleep(0.0005)
        raise m64.M64Error(
            f"core never reported PAUSED after {nudged} pause requests "
            f"(state={self.core.emu_state()}, frame={self.frame})")

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
        # Step frames until the core reports the write finished. Checking only
        # that the file exists is not enough: it appears early and half-written
        # (a 16 KiB stub for an 8 MiB machine), and loading it later silently
        # gives you the wrong world.
        self._pump(self.core.save_done, "savestate write")
        if not self.core.save_ok:
            raise m64.M64Error(f"core reported savestate failure for {path}")
        # M64CORE_STATE_SAVECOMPLETE fires BEFORE the bytes are on disk, so the
        # file is still a truncated gzip at this point ("Compressed file ended
        # before the end-of-stream marker"). A stub like that loads as garbage
        # or not at all, which is a horrible bug to meet later. Wait for the
        # file to actually decompress.
        self._await_complete_state_file(path)
        # The pad is part of the environment's state even though it lives
        # outside the emulator: restoring RAM but not the controller means the
        # settling frame replays a different input and the rollback is not exact.
        self._states[sid] = (path, before, self.pad.raw)
        return sid

    def load_state(self, sid):
        path, _, pads = self._states[sid]
        self.pad.restore(pads)
        self.core.load_state_from(path)
        self._pump(self.core.load_done, "savestate load")
        if not self.core.load_ok:
            raise m64.M64Error(f"core reported load failure for {path}")
        return self.frame

    def load_state_file(self, path, settle=2):
        """Restore a savestate from disk — the way every episode begins.

        The menus are walked exactly once (src/mk64/to_race.py); after that the
        start line is just a file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.pad.neutral()
        self.core.load_state_from(path)
        self._pump(self.core.load_done, f"load of {path.name}")
        if not self.core.load_ok:
            raise m64.M64Error(f"core refused savestate {path}")
        self.step(settle)
        return self.frame

    def drop_state(self, sid):
        path, _, _ = self._states.pop(sid, (None, None, None))
        if path and path.exists():
            path.unlink()

    # --- eyes ------------------------------------------------------------
    def screenshot(self, name=None, settle=3):
        """Grab what is on screen, as a PNG path.

        Needed because the menus cannot be navigated blind, and it is the same
        capture path stage 2's confidence overlay will draw on top of.
        """
        before = set(self.shot_dir.glob("*.png"))
        self.core.take_screenshot()
        for _ in range(settle):
            self.step(1)
            new = set(self.shot_dir.glob("*.png")) - before
            if new:
                shot = new.pop()
                if name:
                    dest = self.shot_dir / name
                    shot.replace(dest)
                    return dest
                return shot
        raise m64.M64Error("no screenshot appeared — does the video plugin "
                           "support ReadScreen?")

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
