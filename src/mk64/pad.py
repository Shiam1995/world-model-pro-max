"""The N64 controller, as bits, and the shared-memory block the core reads.

Bit layout is taken from vendor/m64p/m64p_plugin.h (BUTTONS) rather than from
memory — the union packs two signed 8-bit axes into the high half of a u32, and
guessing that wrong produces a kart that mysteriously only turns left.
"""

import ctypes
import mmap
import os
import struct

# Bit positions within BUTTONS.Value, low half.
R_DPAD, L_DPAD, D_DPAD, U_DPAD = 0, 1, 2, 3
START, Z_TRIG, B_BUTTON, A_BUTTON = 4, 5, 6, 7
R_CBUTTON, L_CBUTTON, D_CBUTTON, U_CBUTTON = 8, 9, 10, 11
R_TRIG, L_TRIG = 12, 13
X_AXIS_SHIFT, Y_AXIS_SHIFT = 16, 24

SHM_MAGIC = 0x4D363453  # 'M64S', must match input_shm.c
SHM_NAME = os.environ.get("MK64_INPUT_SHM", "/mk64_input")
# struct shm_pad: magic, version, present, polls, pad[4]
SHM_FMT = "<4I4I"
SHM_SIZE = struct.calcsize(SHM_FMT)


def buttons(steer=0.0, accel=False, brake=False, hop=False, item=False,
            start=False, y=0.0, up=False, down=False, left=False, right=False):
    """Pack a kart action into a raw BUTTONS word.

    steer/y are -1.0..+1.0 and map onto the signed 8-bit axes. The N64 stick
    saturates around +-80 in practice, not 127; full-scale 127 is outside what
    the hardware reports and some games clamp or misread it, so we scale to 80.
    """
    v = 0
    if accel: v |= 1 << A_BUTTON
    if brake: v |= 1 << B_BUTTON
    if hop:   v |= 1 << R_TRIG
    if item:  v |= 1 << Z_TRIG
    if start: v |= 1 << START
    # D-pad: menus are far more reliable on the pad than on the analog stick,
    # which needs to cross a deadzone and can register as two steps.
    if up:    v |= 1 << U_DPAD
    if down:  v |= 1 << D_DPAD
    if left:  v |= 1 << L_DPAD
    if right: v |= 1 << R_DPAD

    def axis(f):
        a = int(round(max(-1.0, min(1.0, f)) * 80))
        return a & 0xFF            # two's complement into 8 bits

    v |= axis(steer) << X_AXIS_SHIFT
    v |= axis(y) << Y_AXIS_SHIFT
    return v


def unpack(value):
    """Inverse of buttons(), for logging a human's lap back into actions."""
    def signed(b):
        return b - 256 if b > 127 else b
    return {
        "steer": signed((value >> X_AXIS_SHIFT) & 0xFF) / 80.0,
        "y":     signed((value >> Y_AXIS_SHIFT) & 0xFF) / 80.0,
        "accel": bool(value >> A_BUTTON & 1),
        "brake": bool(value >> B_BUTTON & 1),
        "hop":   bool(value >> R_TRIG & 1),
        "item":  bool(value >> Z_TRIG & 1),
        "start": bool(value >> START & 1),
        "up":    bool(value >> U_DPAD & 1),
        "down":  bool(value >> D_DPAD & 1),
        "left":  bool(value >> L_DPAD & 1),
        "right": bool(value >> R_DPAD & 1),
    }


class ShmPad:
    """Writer side of the controller shared memory.

    Created before the core loads the plugin, so the plugin finds the segment
    already stamped and does not zero our `present` mask.
    """

    def __init__(self, name=SHM_NAME, present=0b0001):
        self.name = name
        path = f"/dev/shm/{name.lstrip('/')}"
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        os.ftruncate(self._fd, SHM_SIZE)
        self._mm = mmap.mmap(self._fd, SHM_SIZE)
        self._pads = [0, 0, 0, 0]
        self._write(magic=SHM_MAGIC, version=0x000100, present=present, polls=0)

    def _write(self, magic=SHM_MAGIC, version=0x000100, present=None, polls=None):
        cur = struct.unpack(SHM_FMT, self._mm[:SHM_SIZE])
        present = cur[2] if present is None else present
        polls = cur[3] if polls is None else polls
        self._mm[:SHM_SIZE] = struct.pack(SHM_FMT, magic, version, present, polls,
                                          *self._pads)

    def set(self, port=0, **kwargs):
        """set(steer=-0.4, accel=True) — takes effect on the next emulated frame."""
        self._pads[port] = buttons(**kwargs)
        self._write()

    def set_raw(self, value, port=0):
        self._pads[port] = value & 0xFFFFFFFF
        self._write()

    @property
    def raw(self):
        """Current controller words — part of the env state a savestate must carry."""
        return tuple(self._pads)

    def restore(self, pads):
        self._pads = list(pads)
        self._write()

    def neutral(self):
        self._pads = [0, 0, 0, 0]
        self._write()

    @property
    def polls(self):
        """How many times the core has asked for controller 1's state.

        If this stops rising, the emulator is not running frames — which looks
        exactly like 'the agent learned to do nothing'. Assert on it.
        """
        return struct.unpack(SHM_FMT, self._mm[:SHM_SIZE])[3]

    def close(self):
        try:
            self._mm.close()
        finally:
            os.close(self._fd)

    def unlink(self):
        try:
            os.unlink(f"/dev/shm/{self.name.lstrip('/')}")
        except FileNotFoundError:
            pass
