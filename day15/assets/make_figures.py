"""Generate day15/assets/d15_ray_system.svg (the actor-learner loop on Ray, GUIDE Block 1).
Usage: python day15/assets/make_figures.py day15/assets
Same drawing helpers and palette as day14/assets/make_figures.py.

Facts drawn here (Ray 2.60.0):
  - ray.put always writes the payload to the shared-memory (plasma) store (core_worker.cc PutInLocalPlasmaStore).
  - A task / actor-method return value below max_direct_call_object_size (100 KiB, ray_config_def.h) is returned
    inline to the caller and kept in the caller's in-process memory store; larger returns go to plasma.
  - A CartPole fragment (128 steps x 4 envs) is about 21.5 KB, so it comes back inline to the driver and is
    inlined again into learner.update; the 36,620-byte weight dict is ray.put once and read zero-copy by each actor.
"""
import html
import re
import sys
from pathlib import Path

FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"
INK, GREY, MUTED = "#1f2937", "#6b7280", "#94a3b8"
C = {
    "drv": ("#e8f0fe", "#3b6fd4"),    # the driver process
    "act": ("#e3f5e9", "#2f9e5b"),    # rollout actors (CPU)
    "lrn": ("#f1eafd", "#7048c8"),    # the learner actor (GPU)
    "ray": ("#fff1dc", "#e3891b"),    # Ray's own processes (GCS, raylet)
    "obj": ("#ddf4f5", "#0b7f86"),    # object stores and objects
    "note": ("#f8fafc", "#cbd5e1"),
    "white": ("#ffffff", "#cbd5e1"),
}
CALL, DATA = "#3b6fd4", "#0b7f86"


def rich(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"_\{([^}]*)\}", r'<tspan baseline-shift="sub" font-size="75%">\1</tspan>', s)
    return s


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.p = w, h, []
        self.p.append(f'<rect width="{w}" height="{h}" rx="16" fill="#ffffff"/>')

    def text(self, x, y, s, size=12.5, fill=INK, anchor="start", weight="normal", mono=False, italic=False):
        fam = MONO if mono else FONT
        st = ' font-style="italic"' if italic else ""
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}" '
                      f'text-anchor="{anchor}" font-weight="{weight}" xml:space="preserve"{st}>{rich(s)}</text>')

    def rect(self, x, y, w, h, kind="note", rx=8, dash=False, sw=1.5):
        f, s = C[kind]
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{f}" '
                      f'stroke="{s}" stroke-width="{sw}"{d}/>')

    def block(self, x, y, lines, gap=17):
        """Left-aligned lines: each item is (text, style dict)."""
        for i, (t, kw) in enumerate(lines):
            if t:
                self.text(x, y + i * gap, t, **kw)

    def path(self, d, color=GREY, sw=1.7, dash=False, head=True):
        mid = "ah" + color.strip("#")
        da = ' stroke-dasharray="6 4"' if dash else ""
        m = f' marker-end="url(#{mid})"' if head else ""
        self.p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{m}/>')

    def badge(self, x, y, n, color=CALL):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#ffffff" stroke="{color}" stroke-width="1.6"/>')
        self.text(x, y + 4.3, str(n), size=11.5, fill=color, anchor="middle", weight="700")

    def save(self, path):
        colors = {GREY, INK, MUTED, CALL, DATA} | {v[1] for v in C.values()}
        defs = "".join(
            f'<marker id="ah{c.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in sorted(colors))
        body = "\n".join(self.p)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}"><defs>{defs}</defs>\n{body}\n</svg>\n', encoding="utf-8", newline="\n")


def T(t, size=12.5, weight="normal", fill=INK, italic=False):
    return (t, dict(size=size, weight=weight, fill=fill, italic=italic))


def B(t, size=14, fill=INK):
    return T(t, size=size, weight="700", fill=fill)


def M(t, size=12, fill=INK):
    return (t, dict(size=size, mono=True, fill=fill))


def G(t, size=11.5):
    return T(t, size=size, fill=GREY)


def ray_system(out: Path) -> None:
    W, H = 1280, 960
    s = Svg(W, H)
    s.text(30, 36, "One iteration of the Day 15 loop on Ray: who calls what, and where the bytes go", size=19,
           weight="700")
    s.text(30, 58, "Sync mode, CartPole, 2 rollout actors + 1 GPU learner on one node. Solid blue: a remote method "
           "call (it returns an ObjectRef at once). Dashed teal: object data. Numbers: order within an iteration.",
           size=12.5, fill=GREY)

    # ---------------------------------------------------------------- driver
    dx, dy, dw, dh = 30, 78, 610, 340
    s.rect(dx, dy, dw, dh, "drv")
    s.block(dx + 16, dy + 26, [
        B("Driver process: train.py"),
        G("Owns every object returned by a call it submits."),
        M("ray.init(object_store_memory=2 GiB)        # CPU 7, GPU 1 detected"),
        M("RW  = ray.remote(num_cpus=1)(RolloutWorker)"),
        M("actors  = [RW.remote(i, cfg) for i in range(2)]   # ActorHandle"),
        M("learner = ray.remote(num_cpus=1, num_gpus=1)(Learner).remote(cfg)"),
        M("for it in range(iters):"),
        M("      refs  = [a.sample.remote(128) for a in actors]  # ObjectRefs"),
        M("      stats = ray.get(learner.update.remote(*refs))    # blocks"),
        M("      ray.get([a.set_weights.remote(stats['pub'])"),
        M("               for a in actors])  # sync: before the next sample"),
        M("wait mode:  ready, rest = ray.wait(pending, num_returns=1)"),
        M("after:      ray.timeline(path); ray.shutdown()"),
    ], gap=17.5)
    for i, n in ((7, 1), (8, 2), (9, 4)):
        s.badge(dx + 36, dy + 26 + i * 17.5 - 4, n)
    # the driver's in-process memory store
    mx, my, mw, mh = dx + 12, dy + 258, dw - 24, 70
    s.rect(mx, my, mw, mh, "white", rx=7)
    s.text(mx + 10, my + 18, "in-process memory store (small objects owned by the driver)", size=11.5, fill=DATA,
           weight="700")
    chips = [("F0: fragment of actor 0", "21.5 KB, inline"), ("F1: fragment of actor 1", "21.5 KB, inline"),
             ("stats (incl. pub)", "a few KB, inline")]
    cw = 182
    for i, (a, b) in enumerate(chips):
        cx = mx + 10 + i * (cw + 8)
        s.rect(cx, my + 27, cw, 36, "obj", rx=6, sw=1.2)
        s.text(cx + cw / 2, my + 42, a, size=11.5, anchor="middle", weight="700")
        s.text(cx + cw / 2, my + 56, b, size=10.5, anchor="middle", fill=GREY)
    f0x, f1x, stx = mx + 10 + cw / 2, mx + 10 + cw + 8 + cw / 2, mx + 10 + 2 * (cw + 8) + cw / 2
    chip_bottom = my + 63

    # ---------------------------------------------------------------- Ray runtime
    rx, ry, rw, rh = 670, 78, 580, 200
    s.rect(rx, ry, rw, rh, "ray")
    s.block(rx + 16, ry + 26, [
        B("Ray's own processes on the node (ray.init starts them, ray stop ends them)"),
        T("GCS: cluster metadata, the actor directory"),
        T("raylet: scheduler + worker pool"),
        T("  · resources: CPU 7.0 (cgroup cpu.max 7.68, truncated), GPU 1.0"),
        T("  · places each actor by its num_cpus / num_gpus; too big = pending forever"),
        T("  · sets OMP_NUM_THREADS = floor(num_cpus) in each worker process"),
        T("  · sets CUDA_VISIBLE_DEVICES only when num_gpus > 0"),
        G("Method calls and small objects go straight between processes (gRPC);"),
        G("the raylet places actors and manages the shared-memory store."),
    ], gap=19)

    # ---------------------------------------------------------------- workers
    wy, wh = 500, 250
    a0x, a0w = 30, 400
    s.rect(a0x, wy, a0w, wh, "act")
    s.block(a0x + 16, wy + 26, [
        B("RolloutWorker 0: actor process, CPU"),
        G("num_cpus=1, num_gpus=0: OMP 1, and it still sees the GPU"),
        M("set_weights(pub):"),
        M("  w = ray.get(pub['ref'][0])  # read-only numpy view"),
        M("  copy w into the torch params; SHA-256 digest"),
        M("sample(T=128):  4 envs x 128 steps, CPU forward"),
        M("  return fragment dict (21.5 KB):"),
        M("    obs, actions, logp, values, next_values,"),
        M("    rewards, terminated, truncated,"),
        M("    meta {policy_version v, digest, episodes}"),
        G("Calls from one caller run one at a time, in order."),
    ], gap=19.5)
    a1x, a1w = 450, 200
    s.rect(a1x, wy, a1w, wh, "act")
    s.block(a1x + 14, wy + 26, [
        B("RolloutWorker 1"),
        G("actor process, CPU"),
        T("the same two methods,"),
        T("its own 4 envs, its own"),
        T("seed, its own params"),
        T(""),
        T("One fragment per sample():"),
        T("tagged with the version"),
        T("it really sampled with."),
    ], gap=19.5)
    lx, lw = 670, 580
    s.rect(lx, wy, lw, wh, "lrn")
    s.block(lx + 16, wy + 26, [
        B("Learner: actor process, GPU"),
        G("num_cpus=1, num_gpus=1: CUDA_VISIBLE_DEVICES='0'"),
        M("update(*fragments):   # Ray resolved both refs before the call"),
        M("  lag = version - v;  digest == published[v] ?"),
        M("  GAE on CPU -> one H2D copy -> PPO on the GPU"),
        M("     4 epochs x 4 minibatches = 16 optimizer steps"),
        M("  version += 1;  publish():"),
        M("     params -> numpy (D2H) -> ref = ray.put(weights)"),
        M("  return stats {pub: {version, digest, ref: [ref]},"),
        M("                lags, ratio, clip, KL, timings}"),
        G("ray.put once for all actors; the actors fetch it themselves."),
    ], gap=19.5)

    # ---------------------------------------------------------------- plasma store
    px, py, pw, ph = 30, 812, 1220, 112
    s.rect(px, py, pw, ph, "obj")
    s.block(px + 16, py + 26, [
        B("Shared-memory object store", size=13.5),
        T("plasma, in /dev/shm, 2 GiB"),
        G("immutable objects; any process on"),
        G("the node reads them without a copy"),
    ], gap=19)
    wcx, wcy, wcw, wch = 330, py + 22, 440, 68
    s.rect(wcx, wcy, wcw, wch, "white", rx=7)
    s.text(wcx + wcw / 2, wcy + 26, "weights of version v: dict of numpy arrays, 36,620 B", size=12.5,
           anchor="middle", weight="700")
    s.text(wcx + wcw / 2, wcy + 47, "ray.put always writes here, whatever the size", size=11.5, anchor="middle",
           fill=GREY)
    ocx, ocw = 800, 430
    s.rect(ocx, wcy, ocw, wch, "white", rx=7, dash=True)
    s.text(ocx + ocw / 2, wcy + 22, "returns of 100 KiB or more land here, not inline", size=12.5, anchor="middle",
           weight="700")
    s.text(ocx + ocw / 2, wcy + 40, "limit: max_direct_call_object_size = 100 KiB", size=11.5, anchor="middle",
           fill=GREY)
    s.text(ocx + ocw / 2, wcy + 57, "e.g. Exp 3 fragments with 1 or 2 actors", size=11.5, anchor="middle",
           fill=GREY)

    # ---------------------------------------------------------------- arrows
    top, dbot = wy - 2, dy + dh
    # (1) sample calls and (4) set_weights calls: driver -> each actor
    for x, n in ((80, 1), (140, 4), (595, 1), (630, 4)):
        s.path(f"M{x},{dbot} L{x},{top}", color=CALL)
        s.badge(x - 15, dbot + 40 + (0 if n == 1 else 22), n)
    # fragments come back inline into the driver's memory store
    s.path(f"M{300},{wy} L{f0x:.1f},{chip_bottom + 2}", color=DATA, dash=True)
    s.path(f"M{500},{wy} L{f1x:.1f},{chip_bottom + 2}", color=DATA, dash=True)
    s.text(196, dbot + 58, "fragment returned", size=11, fill=DATA, anchor="middle", italic=True)
    s.text(196, dbot + 72, "inline (< 100 KiB)", size=11, fill=DATA, anchor="middle", italic=True)
    # (2) update call: driver -> learner, refs in; Ray copies the small values into the call
    yc = 330
    s.path(f"M{dx + dw},{yc} L{1180},{yc} L{1180},{top}", color=CALL)
    s.badge(668, yc - 18, 2)
    s.text(684, yc - 14, "learner.update.remote(F0_ref, F1_ref): refs in; Ray copies the 2 × 21.5 KB values "
           "into the call", size=11.5, fill=CALL)
    # stats return to the driver (inline) -> ray.get
    ys = my + 45
    s.path(f"M{900},{wy} L{900},{ys} L{mx + mw - 6:.1f},{ys}", color=DATA, dash=True)
    s.text(912, ys + 40, "stats returned inline;", size=11.5, fill=DATA, italic=True)
    s.text(912, ys + 56, "ray.get in the driver unblocks", size=11.5, fill=DATA, italic=True)
    # (3) the learner puts the new weights once
    s.path(f"M{720},{wy + wh} L{720},{wcy - 2}", color=DATA, dash=True)
    s.badge(740, wy + wh + 30, 3, color=DATA)
    s.text(756, wy + wh + 34, "ray.put(weights): once per version, from the learner", size=11.5, fill=DATA,
           italic=True)
    # (4) each actor fetches the same object, zero-copy
    s.path(f"M{380},{wcy} L{380},{wy + wh + 2}", color=DATA, dash=True)
    s.path(f"M{550},{wcy} L{550},{wy + wh + 2}", color=DATA, dash=True)
    s.badge(150, wy + wh + 30, 4, color=DATA)
    s.text(166, wy + wh + 34, "ray.get(pub['ref'][0]) in", size=11.5, fill=DATA, italic=True)
    s.text(166, wy + wh + 50, "each actor: zero-copy, read-only", size=11.5, fill=DATA, italic=True)

    s.text(30, 948, "Async mode: ① for iteration k+1 is submitted before ② for iteration k. Calls to one actor run in "
           "submission order, so that sample runs after set_weights(v_{k}) and is trained at version k+1: lag 1.",
           size=12, fill=GREY)
    s.save(out / "d15_ray_system.svg")


if __name__ == "__main__":
    o = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    o.mkdir(parents=True, exist_ok=True)
    ray_system(o)
    print("wrote", o / "d15_ray_system.svg")
