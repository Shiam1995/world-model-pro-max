/* input_shm — a mupen64plus input plugin whose controller state lives in
 * shared memory, so a Python trainer can set it exactly, frame by frame.
 *
 * Why not drive mupen64plus-input-sdl by injecting fake SDL key events
 * (M64CMD_SEND_SDL_KEYDOWN)? Because keys are binary. MK64 steering is an
 * 8-bit signed axis, and a policy that can only ever steer full-lock cannot
 * hold a racing line. This plugin passes the axis through untouched.
 *
 * Layout of /dev/shm/<name> is struct shm_pad below: the writer (Python) fills
 * in `pad[i]`, the core reads it in GetKeys(). `polls` lets the reader verify
 * the core is actually consuming frames, which is how we catch a silently
 * stalled emulator instead of training against a frozen screen.
 */

/* POSIX prototypes (ftruncate, shm_open) are hidden under a strict -std=c11. */
#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define M64P_PLUGIN_PROTOTYPES 1
#include "m64p_types.h"
#include "m64p_common.h"
#include "m64p_plugin.h"

#define PLUGIN_NAME       "input-shm"
#define PLUGIN_VERSION    0x000100
#define INPUT_PLUGIN_API  0x020100   /* core rejects anything below 0x020100 */
#define SHM_DEFAULT       "/mk64_input"
#define N_PADS            4

struct shm_pad {
    uint32_t magic;          /* 'M64S' — guards against a stale/foreign mapping */
    uint32_t version;
    uint32_t present;        /* bitmask of connected controllers */
    uint32_t polls;          /* ++ on every GetKeys, so Python can see frames move */
    uint32_t pad[N_PADS];    /* raw BUTTONS.Value per controller */
};

#define SHM_MAGIC 0x4D363453u

static struct shm_pad *g_shm = NULL;
static int   g_fd = -1;
static void (*g_debug)(void *, int, const char *) = NULL;
static void  *g_debug_ctx = NULL;
static CONTROL *g_controls = NULL;

static void logmsg(int level, const char *msg)
{
    if (g_debug) g_debug(g_debug_ctx, level, msg);
}

static void shm_open_map(void)
{
    const char *name = getenv("MK64_INPUT_SHM");
    if (!name || !*name) name = SHM_DEFAULT;

    g_fd = shm_open(name, O_RDWR | O_CREAT, 0600);
    if (g_fd < 0) { logmsg(M64MSG_ERROR, "input-shm: shm_open failed"); return; }
    if (ftruncate(g_fd, sizeof(struct shm_pad)) != 0)
        logmsg(M64MSG_WARNING, "input-shm: ftruncate failed (already sized?)");

    void *p = mmap(NULL, sizeof(struct shm_pad), PROT_READ | PROT_WRITE,
                   MAP_SHARED, g_fd, 0);
    if (p == MAP_FAILED) { logmsg(M64MSG_ERROR, "input-shm: mmap failed"); g_shm = NULL; return; }

    g_shm = (struct shm_pad *)p;
    /* Only initialise if Python has not already staked the segment: it may have
     * been created and filled before the core ever loaded this plugin. */
    if (g_shm->magic != SHM_MAGIC) {
        memset(g_shm, 0, sizeof(*g_shm));
        g_shm->magic   = SHM_MAGIC;
        g_shm->version = PLUGIN_VERSION;
        g_shm->present = 1;            /* controller 1 only, by default */
    }
    logmsg(M64MSG_INFO, "input-shm: mapped controller state");
}

EXPORT m64p_error CALL PluginStartup(m64p_dynlib_handle CoreLibHandle, void *Context,
                                     void (*DebugCallback)(void *, int, const char *))
{
    (void)CoreLibHandle;
    g_debug_ctx = Context;
    g_debug = DebugCallback;
    shm_open_map();
    return M64ERR_SUCCESS;
}

EXPORT m64p_error CALL PluginShutdown(void)
{
    if (g_shm) { munmap(g_shm, sizeof(struct shm_pad)); g_shm = NULL; }
    if (g_fd >= 0) { close(g_fd); g_fd = -1; }
    g_debug = NULL;
    return M64ERR_SUCCESS;
}

EXPORT m64p_error CALL PluginGetVersion(m64p_plugin_type *PluginType, int *PluginVersion,
                                        int *APIVersion, const char **PluginNamePtr,
                                        int *Capabilities)
{
    if (PluginType)     *PluginType     = M64PLUGIN_INPUT;
    if (PluginVersion)  *PluginVersion  = PLUGIN_VERSION;
    if (APIVersion)     *APIVersion     = INPUT_PLUGIN_API;
    if (PluginNamePtr)  *PluginNamePtr   = PLUGIN_NAME;
    if (Capabilities)   *Capabilities   = 0;
    return M64ERR_SUCCESS;
}

EXPORT void CALL InitiateControllers(CONTROL_INFO ControlInfo)
{
    g_controls = ControlInfo.Controls;
    unsigned present = (g_shm ? g_shm->present : 1u);
    for (int i = 0; i < N_PADS; i++) {
        g_controls[i].Present = (present >> i) & 1u;
        g_controls[i].RawData = 0;
        g_controls[i].Plugin  = PLUGIN_MEMPAK;   /* MK64 saves ghosts to a mempak */
    }
}

EXPORT void CALL GetKeys(int Control, BUTTONS *Keys)
{
    if (!Keys) return;
    if (!g_shm || g_shm->magic != SHM_MAGIC) { Keys->Value = 0; return; }
    if (Control < 0 || Control >= N_PADS)    { Keys->Value = 0; return; }
    Keys->Value = g_shm->pad[Control];
    if (Control == 0) g_shm->polls++;
}

/* No mempak/rumble emulation here: the core handles PLUGIN_MEMPAK itself when
 * RawData is 0, which is all MK64 needs for staff-ghost saves. */
EXPORT void CALL ControllerCommand(int Control, unsigned char *Command) { (void)Control; (void)Command; }
EXPORT void CALL ReadController(int Control, unsigned char *Command)    { (void)Control; (void)Command; }
EXPORT void CALL SDL_KeyDown(int keymod, int keysym) { (void)keymod; (void)keysym; }
EXPORT void CALL SDL_KeyUp(int keymod, int keysym)   { (void)keymod; (void)keysym; }
EXPORT void CALL RenderCallback(void) { }
EXPORT int  CALL RomOpen(void)  { return 1; }
EXPORT void CALL RomClosed(void) { }
