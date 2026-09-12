"""ctypes binding to libmupen64plus.so.2.

The emulator is loaded *into this process* rather than spawned as a subprocess.
That is the whole point: in-process means save_state()/load_state() are function
calls on memory we can see, so `env.clone()` costs microseconds and the
savestate becomes a usable world model. A subprocess + screen scraping could
never do segment search.

Only the core is bound here. Anything that knows what a lap or a kart is
belongs in env.py or an adapter, never in this file.
"""

import ctypes
import os
import struct
import threading

# --- m64p_types.h -----------------------------------------------------------

M64ERR = {
    0: "SUCCESS", 1: "NOT_INIT", 2: "ALREADY_INIT", 3: "INCOMPATIBLE",
    4: "INPUT_ASSERT", 5: "INPUT_INVALID", 6: "INPUT_NOT_FOUND",
    7: "NO_MEMORY", 8: "FILES", 9: "INTERNAL", 10: "INVALID_STATE",
    11: "PLUGIN_FAIL", 12: "SYSTEM_FAIL", 13: "UNSUPPORTED", 14: "WRONG_TYPE",
}

(M64CMD_NOP, M64CMD_ROM_OPEN, M64CMD_ROM_CLOSE, M64CMD_ROM_GET_HEADER,
 M64CMD_ROM_GET_SETTINGS, M64CMD_EXECUTE, M64CMD_STOP, M64CMD_PAUSE,
 M64CMD_RESUME, M64CMD_CORE_STATE_QUERY, M64CMD_STATE_LOAD, M64CMD_STATE_SAVE,
 M64CMD_STATE_SET_SLOT, M64CMD_SEND_SDL_KEYDOWN, M64CMD_SEND_SDL_KEYUP,
 M64CMD_SET_FRAME_CALLBACK, M64CMD_TAKE_NEXT_SCREENSHOT, M64CMD_CORE_STATE_SET,
 M64CMD_READ_SCREEN, M64CMD_RESET, M64CMD_ADVANCE_FRAME) = range(21)

M64PLUGIN_RSP, M64PLUGIN_GFX, M64PLUGIN_AUDIO, M64PLUGIN_INPUT, M64PLUGIN_CORE = 1, 2, 3, 4, 5

# m64p_core_param
M64CORE_EMU_STATE = 1
M64CORE_VIDEO_MODE = 2
M64CORE_SAVESTATE_SLOT = 3
M64CORE_SPEED_FACTOR = 4
M64CORE_SPEED_LIMITER = 5
M64CORE_STATE_LOADCOMPLETE = 10
M64CORE_STATE_SAVECOMPLETE = 11

M64EMU_STOPPED, M64EMU_RUNNING, M64EMU_PAUSED = 1, 2, 3

# m64p_type for config
M64TYPE_INT, M64TYPE_FLOAT, M64TYPE_BOOL, M64TYPE_STRING = 1, 2, 3, 4

CORE_API_VERSION = 0x020001
CONFIG_API_VERSION = 0x020000

DEBUG_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
STATE_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
FRAME_CB = ctypes.CFUNCTYPE(None, ctypes.c_uint)

MSG_LEVEL = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "STATUS", 5: "VERBOSE"}


class M64Error(RuntimeError):
    pass


def _check(rc, what):
    if rc != 0:
        raise M64Error(f"{what} failed: M64ERR_{M64ERR.get(rc, rc)}")


class Core:
    """A loaded mupen64plus core. One per process — the core holds global state."""

    def __init__(self, lib="libmupen64plus.so.2", data_dir="/usr/share/mupen64plus",
                 config_dir=None, verbose=False):
        self.verbose = verbose
        self.lib = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        self._plugins = {}
        self._log = []
        # Keep strong refs: ctypes callbacks are garbage collected otherwise and
        # the core is left calling freed memory.
        self._debug_cb = DEBUG_CB(self._on_debug)
        self._state_cb = STATE_CB(self._on_state)
        self._frame_cb = None
        self.frame = 0
        # Savestates are written by the core's worker thread ("m64pwq"), not
        # inline, so the file on disk is incomplete for a while after the
        # command returns. The core tells us when it is done — listen, rather
        # than sleeping and hoping.
        self.save_done = threading.Event()
        self.load_done = threading.Event()
        self.save_ok = None
        self.load_ok = None

        if config_dir is None:
            config_dir = os.path.expanduser("~/.config/mupen64plus")

        _check(self.lib.CoreStartup(
            ctypes.c_int(CORE_API_VERSION),
            ctypes.c_char_p(config_dir.encode()),
            ctypes.c_char_p(data_dir.encode()),
            ctypes.c_void_p(0), self._debug_cb,
            ctypes.c_void_p(0), self._state_cb), "CoreStartup")

    # --- logging ---------------------------------------------------------
    def _on_debug(self, ctx, level, message):
        msg = f"[core/{MSG_LEVEL.get(level, level)}] {message.decode(errors='replace')}"
        self._log.append(msg)
        if self.verbose and level <= 3:
            print(msg)

    def _on_state(self, ctx, param, value):
        if param == M64CORE_STATE_SAVECOMPLETE:
            self.save_ok = bool(value)
            self.save_done.set()
        elif param == M64CORE_STATE_LOADCOMPLETE:
            self.load_ok = bool(value)
            self.load_done.set()

    @property
    def log(self):
        return list(self._log)

    # --- config ----------------------------------------------------------
    def config_section(self, name):
        h = ctypes.c_void_p()
        _check(self.lib.ConfigOpenSection(ctypes.c_char_p(name.encode()), ctypes.byref(h)),
               f"ConfigOpenSection({name})")
        return h

    def config_set(self, section, key, value):
        h = self.config_section(section)
        if isinstance(value, bool):
            t, v = M64TYPE_BOOL, ctypes.byref(ctypes.c_int(1 if value else 0))
        elif isinstance(value, int):
            t, v = M64TYPE_INT, ctypes.byref(ctypes.c_int(value))
        elif isinstance(value, float):
            t, v = M64TYPE_FLOAT, ctypes.byref(ctypes.c_float(value))
        elif isinstance(value, str):
            t, v = M64TYPE_STRING, ctypes.c_char_p(value.encode())
        else:
            raise TypeError(f"cannot set config value of type {type(value)}")
        _check(self.lib.ConfigSetParameter(h, ctypes.c_char_p(key.encode()),
                                          ctypes.c_int(t), v),
               f"ConfigSetParameter({section}.{key})")

    # --- rom / plugins ---------------------------------------------------
    def open_rom(self, path):
        with open(path, "rb") as fh:
            data = fh.read()
        buf = ctypes.create_string_buffer(data, len(data))
        _check(self.lib.CoreDoCommand(ctypes.c_int(M64CMD_ROM_OPEN),
                                      ctypes.c_int(len(data)), buf), "ROM_OPEN")
        self.rom_size = len(data)

    def attach_plugin(self, kind, so_path):
        handle = ctypes.CDLL(so_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        startup = handle.PluginStartup
        startup.argtypes = [ctypes.c_void_p, ctypes.c_void_p, DEBUG_CB]
        # Pass the core's own handle so the plugin can resolve core symbols.
        core_handle = ctypes.cast(self.lib._handle, ctypes.c_void_p)
        _check(startup(core_handle, None, self._debug_cb), f"PluginStartup({so_path})")
        _check(self.lib.CoreAttachPlugin(ctypes.c_int(kind),
                                        ctypes.c_void_p(handle._handle)),
               f"CoreAttachPlugin({kind})")
        self._plugins[kind] = handle
        return handle

    def detach_all(self):
        for kind in (M64PLUGIN_GFX, M64PLUGIN_AUDIO, M64PLUGIN_INPUT, M64PLUGIN_RSP):
            if kind in self._plugins:
                self.lib.CoreDetachPlugin(ctypes.c_int(kind))
                try:
                    self._plugins[kind].PluginShutdown()
                except Exception:
                    pass
        self._plugins.clear()

    # --- run control -----------------------------------------------------
    def set_frame_callback(self, fn):
        """Register a per-frame callback. Required for frame-stepped control."""
        def _wrap(idx):
            self.frame = idx
            fn(idx)
        self._frame_cb = FRAME_CB(_wrap)
        _check(self.lib.CoreDoCommand(ctypes.c_int(M64CMD_SET_FRAME_CALLBACK),
                                      ctypes.c_int(0), self._frame_cb),
               "SET_FRAME_CALLBACK")

    def execute(self):
        """Blocks until the emulator stops — call from a worker thread."""
        return self.lib.CoreDoCommand(ctypes.c_int(M64CMD_EXECUTE), 0, None)

    def _cmd(self, cmd, i=0, p=None, what=""):
        _check(self.lib.CoreDoCommand(ctypes.c_int(cmd), ctypes.c_int(i),
                                      p if p is not None else None), what or str(cmd))

    def pause(self):          self._cmd(M64CMD_PAUSE, what="PAUSE")
    def resume(self):         self._cmd(M64CMD_RESUME, what="RESUME")
    def advance_frame(self):  self._cmd(M64CMD_ADVANCE_FRAME, what="ADVANCE_FRAME")
    def stop(self):           self._cmd(M64CMD_STOP, what="STOP")
    def reset(self, hard=True): self._cmd(M64CMD_RESET, 1 if hard else 0, what="RESET")

    def emu_state(self):
        out = ctypes.c_int(0)
        _check(self.lib.CoreDoCommand(ctypes.c_int(M64CMD_CORE_STATE_QUERY),
                                      ctypes.c_int(M64CORE_EMU_STATE),
                                      ctypes.byref(out)), "CORE_STATE_QUERY")
        return out.value

    def set_speed_limiter(self, on):
        v = ctypes.c_int(1 if on else 0)
        _check(self.lib.CoreDoCommand(ctypes.c_int(M64CMD_CORE_STATE_SET),
                                      ctypes.c_int(M64CORE_SPEED_LIMITER),
                                      ctypes.byref(v)), "SPEED_LIMITER")

    # --- savestates ------------------------------------------------------
    def save_state_to(self, path):
        self.save_done.clear()
        self.save_ok = None
        _check(self.lib.CoreDoCommand(ctypes.c_int(M64CMD_STATE_SAVE), ctypes.c_int(1),
                                      ctypes.c_char_p(str(path).encode())), "STATE_SAVE")

    def load_state_from(self, path):
        self.load_done.clear()
        self.load_ok = None
        _check(self.lib.CoreDoCommand(ctypes.c_int(M64CMD_STATE_LOAD), ctypes.c_int(0),
                                      ctypes.c_char_p(str(path).encode())), "STATE_LOAD")

    def take_screenshot(self):
        """Queue a screenshot; the core writes it on the next rendered frame."""
        _check(self.lib.CoreDoCommand(ctypes.c_int(M64CMD_TAKE_NEXT_SCREENSHOT),
                                      ctypes.c_int(0), None), "TAKE_NEXT_SCREENSHOT")

    # --- memory ----------------------------------------------------------
    def rdram(self):
        """The emulated RDRAM as a writable ctypes array (8 MiB with expansion pak).

        This is how positions get found: snapshot it, diff it against a
        savestate taken a moment later, and correlate. No disassembly.
        """
        self.lib.DebugMemGetPointer.restype = ctypes.POINTER(ctypes.c_uint8)
        M64P_DBG_PTR_RDRAM = 4
        ptr = self.lib.DebugMemGetPointer(ctypes.c_int(M64P_DBG_PTR_RDRAM))
        if not ptr:
            raise M64Error("DebugMemGetPointer(RDRAM) returned NULL "
                           "— was the core built without the debugger?")
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * (8 << 20))).contents

    def read_u32(self, addr):
        """Read a big-endian word at an RDRAM offset (N64 is big-endian)."""
        raw = bytes(self.rdram()[addr:addr + 4])
        return struct.unpack(">I", raw)[0]
