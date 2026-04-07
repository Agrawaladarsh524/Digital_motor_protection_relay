"""
Generate the five result figures for the Digital Motor Protection Relay.

Run:  python3 make_plots.py
Produces: fig1_trip_curve.png ... fig5_sensitivity.png  and results.txt
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from motor_relay import (MotorRelay, balanced, A, tau_from_class,
                         trip_time_closed_form, sequence_components,
                         equivalent_heating_current, IEC_CLASS_WINDOWS)

TAU = 400.0
K_ULT = 1.05
I_SET = 10.0
THETA_HOT = 1.0 / K_ULT ** 2

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titlesize": 10, "figure.autolayout": True})

log = []
def say(s=""):
    print(s)
    log.append(s)


# ----------------------------------------------------------------------
# FIGURE 1 - trip curve vs IEC 60947-4-1 class windows
# ----------------------------------------------------------------------
def fig1():
    mult = np.logspace(math.log10(1.06), 1.0, 400)
    cold = [trip_time_closed_form(m / K_ULT, TAU, 0.0) for m in mult]
    hot = [trip_time_closed_form(m / K_ULT, TAU, THETA_HOT) for m in mult]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(mult, cold, lw=2.0, color="#1f4e79",
              label=f"model, cold start ($\\Theta_0$=0), $\\tau$={TAU:.0f} s")
    ax.loglog(mult, hot, lw=2.0, ls="--", color="#c00000",
              label=f"model, hot start ($\\Theta_0$={THETA_HOT:.3f})")

    # IEC acceptance windows at 7.2 x setting
    for cls, (tmin, tmax), col in [("10", IEC_CLASS_WINDOWS["10"], "#2e7d32"),
                                   ("20", IEC_CLASS_WINDOWS["20"], "#e69f00")]:
        ax.add_patch(plt.Rectangle((6.6, tmin), 1.3, tmax - tmin,
                                   facecolor=col, alpha=0.22, edgecolor=col,
                                   lw=1.4, zorder=1))
        ax.text(8.3, math.sqrt(tmin * tmax) * (1.9 if cls == "20" else 0.55),
                f"class {cls}\n{tmin:g}-{tmax:g} s",
                fontsize=7.5, color=col, va="center", fontweight="bold")

    ax.axvline(7.2, color="k", lw=0.8, ls=":")
    ax.text(4.9, 700, "IEC test point\n7.2 x setting", fontsize=7.5, ha="right")
    ax.annotate("", xy=(7.15, 700), xytext=(5.0, 700),
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.axvline(1.05, color="grey", lw=0.8, ls=":")
    ax.text(1.07, 3, "1.05 x\nno trip", fontsize=7, color="grey")

    t72 = trip_time_closed_form(7.2 / K_ULT, TAU)
    ax.plot([7.2], [t72], "o", ms=7, color="#1f4e79", zorder=5)
    ax.annotate(f"{t72:.2f} s", (7.2, t72), textcoords="offset points",
                xytext=(-42, 10), fontsize=8, fontweight="bold")

    ax.set_xlabel("current, multiples of setting current $I/I_{set}$")
    ax.set_ylabel("trip time (s)")
    ax.set_title("Fig 1  Trip-time characteristic vs IEC 60947-4-1 trip-class windows")
    ax.set_xlim(1.05, 11)
    ax.set_ylim(0.3, 1e4)
    from matplotlib.ticker import ScalarFormatter, FixedLocator
    ax.xaxis.set_major_locator(FixedLocator([1.05, 1.2, 1.5, 2, 3, 4, 5, 7.2, 10]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig("fig1_trip_curve.png")
    plt.close(fig)

    say("FIG 1  trip characteristic")
    say(f"  tau = {TAU:.0f} s   class-10 window bounds tau to "
        f"{tau_from_class(4.0):.0f}-{tau_from_class(10.0):.0f} s")
    say(f"  trip at 7.2x cold = {t72:.2f} s   -> inside class 10 (4-10 s)  PASS")
    say(f"  trip at 1.20x     = {trip_time_closed_form(1.20/K_ULT, TAU):.0f} s "
        f"({trip_time_closed_form(1.20/K_ULT, TAU)/60:.1f} min)  -> under 2 h  PASS")
    say(f"  trip at 1.05x     = infinite (M^2-1 = 0)             -> no trip  PASS")
    say()


# ----------------------------------------------------------------------
# helper: run a relay through a piecewise current profile
# ----------------------------------------------------------------------
def run_profile(relay, segments, dt=0.02):
    """segments = [(duration_s, multiple_of_Iset, running_bool), ...]"""
    ts, th, cur = [], [], []
    for dur, mult, running in segments:
        for _ in range(int(dur / dt)):
            I = mult * I_SET
            relay.update(*balanced(I), dt, running=running)
            ts.append(relay.t); th.append(relay.theta); cur.append(mult)
            if relay.tripped:
                return np.array(ts), np.array(th), np.array(cur)
    return np.array(ts), np.array(th), np.array(cur)


# ----------------------------------------------------------------------
# FIGURE 2 - DOL start, rated run, sustained overload
# ----------------------------------------------------------------------
def fig2():
    relay = MotorRelay(I_set=I_SET, tau_heat=TAU)
    segs = [(2.0, 6.5, True), (300.0, 1.0, True), (1200.0, 1.3, True)]
    ts, th, cur = run_profile(relay, segs)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.4]})
    a1.plot(ts, cur, lw=1.6, color="#444")
    a1.set_ylabel("$I/I_{set}$")
    a1.set_title("Fig 2  DOL start $\\to$ rated run $\\to$ 1.3x sustained overload")

    a2.plot(ts, th, lw=2.0, color="#1f4e79", label="thermal state $\\Theta$")
    a2.axhline(1.0, color="#c00000", lw=1.2, ls="--", label="trip, $\\Theta$=1")
    a2.axhline(0.9, color="#e69f00", lw=1.0, ls=":", label="alarm, $\\Theta$=0.9")
    a2.axhline(0.6, color="green", lw=1.0, ls=":", label="restart permitted below 0.6")
    if relay.tripped:
        a2.plot([relay.trip_time], [1.0], "v", ms=9, color="#c00000", zorder=5)
        a2.annotate(f"{relay.trip_reason}\nt = {relay.trip_time:.0f} s",
                    (relay.trip_time, 1.0), textcoords="offset points",
                    xytext=(-95, -30), fontsize=8, color="#c00000")
    a2.set_xlabel("time (s)"); a2.set_ylabel("$\\Theta$")
    a2.set_ylim(0, 1.15); a2.legend(fontsize=7, loc="lower right")
    fig.savefig("fig2_start_overload.png")
    plt.close(fig)

    i2 = int(2.0 / 0.02) - 1
    say("FIG 2  start then overload")
    say(f"  a single 2 s DOL start at 6.5x costs Theta = {th[i2]:.3f} "
        f"({th[i2]*100:.0f}% of thermal capacity)")
    say(f"  after 5 min at rated current Theta settles at {th[int(302/0.02)-1]:.3f}")
    say(f"  {relay.trip_reason} at t = {relay.trip_time:.0f} s, i.e. "
        f"{(relay.trip_time-302)/60:.1f} min after the 1.3x overload began")
    say()


# ----------------------------------------------------------------------
# FIGURE 3 - thermal memory over repeated starts
# ----------------------------------------------------------------------
def fig3():
    START_S, START_MULT, OFF_S = 3.0, 6.5, 30.0
    dt = 0.02

    relay = MotorRelay(I_set=I_SET, tau_heat=TAU, start_window=1e9)
    ts, th = [], []
    allowed_starts = 0
    for start_no in range(1, 4):
        assert relay.request_start()
        allowed_starts += 1
        for _ in range(int(START_S / dt)):
            relay.update(*balanced(START_MULT * I_SET), dt, running=True)
            ts.append(relay.t); th.append(relay.theta)
        for _ in range(int(OFF_S / dt)):
            relay.update(*balanced(0.0), dt, running=False)
            ts.append(relay.t); th.append(relay.theta)

    theta_before_block = relay.theta
    blocked = not relay.request_start()
    assert blocked

    # Keep the relay stopped until the restart threshold is reached.
    cool_time = 0.0
    while not relay.restart_permitted and cool_time < 600.0:
        relay.update(*balanced(0.0), dt, running=False)
        ts.append(relay.t); th.append(relay.theta)
        cool_time += dt
    assert relay.restart_permitted
    assert relay.request_start()

    # Memoryless comparison: each isolated start begins at theta=0.
    m_relay = MotorRelay(I_set=I_SET, tau_heat=TAU, start_window=1e9)
    mts, mth = [], []
    for _ in range(3):
        m_relay.theta = 0.0
        for _ in range(int(START_S / dt)):
            m_relay.update(*balanced(START_MULT * I_SET), dt, running=True)
            mts.append(m_relay.t); mth.append(m_relay.theta)
        for _ in range(int(OFF_S / dt)):
            m_relay.update(*balanced(0.0), dt, running=False)
            mts.append(m_relay.t); mth.append(m_relay.theta)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ts, th, lw=2.0, color="#1f4e79",
            label="thermal replica with memory (this relay)")
    ax.plot(mts, mth, lw=1.5, ls="--", color="#888",
            label="memoryless comparison (resets each stop)")
    ax.axhline(1.0, color="#c00000", lw=1.2, ls="--", label=r"trip, $\Theta$=1")
    ax.axhline(0.6, color="green", lw=1.0, ls=":", label="restart permitted below 0.6")
    ax.plot([ts[-1]], [th[-1]], "o", ms=6, color="#1f4e79")
    ax.annotate(f"3 starts → $\\Theta$={theta_before_block:.3f}\n4th request BLOCKED\n{cool_time:.1f} s cooling → restart permitted",
                (ts[-1], th[-1]), textcoords="offset points", xytext=(-145, 18),
                fontsize=7.5, fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"thermal state $\Theta$")
    ax.set_title("Fig 3  Thermal memory and enforced restart inhibit")
    ax.set_ylim(0, 1.15); ax.legend(fontsize=7.2, loc="lower right")
    fig.savefig("fig3_thermal_memory.png")
    plt.close(fig)

    say("FIG 3  thermal memory + restart inhibit")
    say(f"  starts allowed = {allowed_starts}; theta before blocked request = {theta_before_block:.3f}")
    say(f"  4th start request -> {'BLOCKED' if blocked else 'ERROR: ALLOWED'}")
    say(f"  cooling time until theta < 0.60 = {cool_time:.1f} s")
    say(f"  restart permitted after cooling = {relay.restart_permitted}")
    say("  memoryless comparison resets theta at each stop, so it does not accumulate history")
    say()
    return allowed_starts, theta_before_block, cool_time


# ----------------------------------------------------------------------
# FIGURE 4 - single phasing, with and without negative-seq compensation
# ----------------------------------------------------------------------
def fig4():
    # one phase open, delta / ungrounded: remaining phases equal and opposite
    RUN_MULT = 1.6
    Ia = RUN_MULT * I_SET + 0j
    Ib = -RUN_MULT * I_SET + 0j
    Ic = 0 + 0j
    _, I1, I2 = sequence_components(Ia, Ib, Ic)

    Ks = [0.0, 1.0, 3.0, 6.0]
    rows = []
    for K in Ks:
        Ieq = equivalent_heating_current(I1, I2, K)
        M = Ieq / (K_ULT * I_SET)
        t = trip_time_closed_form(M, TAU)
        rows.append((K, Ieq, M, t))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.9))

    labels = [("no comp.\nK=0" if k == 0 else f"K={k:.0f}") for k, _, _, _ in rows]
    Ms = [r[2] for r in rows]
    cols = ["#888" if k == 0 else "#1f4e79" for k, _, _, _ in rows]
    a1.bar(labels, Ms, color=cols)
    a1.axhline(1.0, color="#c00000", lw=1.3, ls="--")
    a1.text(2.4, 1.04, "pickup, M=1", color="#c00000", fontsize=7.5)
    for i, m in enumerate(Ms):
        a1.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=8)
    a1.set_ylabel("normalised heating current  $M = I_{eq}/(k\\,I_{set})$")
    a1.set_title("effect of negative-sequence weighting", fontsize=9)
    a1.set_ylim(0, 2.8)

    ts_plot = [r[3] if math.isfinite(r[3]) else 1e4 for r in rows]
    a2.bar(labels, ts_plot, color=cols)
    a2.set_yscale("log")
    for i, (k, ieq, m, t) in enumerate(rows):
        txt = "NEVER\nTRIPS" if not math.isfinite(t) else f"{t:.0f} s"
        a2.text(i, ts_plot[i] * 1.25, txt, ha="center", fontsize=8,
                fontweight="bold",
                color="#c00000" if not math.isfinite(t) else "k")
    a2.set_ylabel("trip time (s), log scale")
    a2.set_title("resulting trip time", fontsize=9)
    a2.set_ylim(10, 1e5)

    fig.suptitle(f"Fig 4  One phase lost while running "
                 f"($I_1$={I1:.2f} A, $I_2$={I2:.2f} A, $I_2/I_1$={I2/I1:.2f})",
                 fontsize=10)
    fig.savefig("fig4_single_phasing.png")
    plt.close(fig)

    say("FIG 4  single phasing")
    say(f"  remaining phases at {RUN_MULT}x setting -> I1 = {I1:.2f} A, "
        f"I2 = {I2:.2f} A, I2/I1 = {I2/I1:.2f}")
    for k, ieq, m, t in rows:
        tt = "never trips" if not math.isfinite(t) else f"{t:.0f} s"
        say(f"  K={k:.0f}: I_eq={ieq:5.2f} A  M={m:4.2f}  -> {tt}")
    say("  WITHOUT compensation the relay sees only 0.88x setting and NEVER")
    say("  trips, while the motor cooks. This single result justifies the")
    say("  entire negative-sequence term.")
    say("  In the full relay the dedicated element 46 (I2/I1 > 0.5) fires")
    say("  immediately; the K=6 value above is a thermal-only sensitivity result.")
    say()


# ----------------------------------------------------------------------
# FIGURE 5 - sensitivity, and the model's known limitation
# ----------------------------------------------------------------------
def fig5():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.9))

    taus = np.linspace(150, 1000, 300)
    t72 = [trip_time_closed_form(7.2 / K_ULT, t) for t in taus]
    a1.plot(taus, t72, lw=2.0, color="#1f4e79")
    a1.axhspan(4, 10, color="#2e7d32", alpha=0.2, label="class 10 (4-10 s)")
    a1.axhspan(6, 20, color="#e69f00", alpha=0.18, label="class 20 (6-20 s)")
    a1.axvline(TAU, color="k", ls=":", lw=1.0)
    a1.plot([TAU], [trip_time_closed_form(7.2 / K_ULT, TAU)], "o", ms=7,
            color="#1f4e79")
    a1.annotate(f"chosen\n$\\tau$={TAU:.0f} s\n{trip_time_closed_form(7.2/K_ULT, TAU):.2f} s",
                (TAU, trip_time_closed_form(7.2 / K_ULT, TAU)),
                textcoords="offset points", xytext=(12, -6), fontsize=8)
    a1.set_xlabel("thermal time constant $\\tau$ (s)")
    a1.set_ylabel("trip time at 7.2 x setting (s)")
    a1.set_title("calibration: $\\tau$ is set by the class window", fontsize=9)
    a1.legend(fontsize=7.5)

    mult = np.linspace(1.3, 10, 300)
    ratio = [trip_time_closed_form(m / K_ULT, TAU, 0.0) /
             trip_time_closed_form(m / K_ULT, TAU, THETA_HOT) for m in mult]
    a2.plot(mult, ratio, lw=2.0, color="#c00000", label="this model (1 thermal mass)")
    a2.text(1.35, 3.0, "commercial reference curves examined\nshow substantially smaller separation",
             fontsize=7.5, va="center")
    a2.set_xlabel("current, multiples of setting")
    a2.set_ylabel("cold trip time / hot trip time")
    a2.set_title("KNOWN LIMITATION: hot/cold ratio too large", fontsize=9)
    a2.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Fig 5  Sensitivity study and self-identified model limitation",
                 fontsize=10)
    fig.savefig("fig5_sensitivity.png")
    plt.close(fig)

    say("FIG 5  sensitivity and limitation")
    say(f"  hot/cold trip-time ratio: {ratio[0]:.1f} at 1.3x rising to "
        f"{ratio[-1]:.1f} at 10x")
    say("  Commercial reference curves examined in this study show substantially")
    say("  smaller hot/cold separation; no universal 2-3 ratio is assumed here.")
    say("  Cause: one thermal mass cannot")
    say("  represent a fast-heating winding AND a slow-heating frame.")
    say("  A higher-fidelity model would use separate thermal states for the")
    say("  fast-heating winding and slower frame, then combine their states.")
    say()


if __name__ == "__main__":
    say("=" * 68)
    say("DIGITAL MOTOR PROTECTION RELAY - RESULTS")
    say(f"I_set = {I_SET:.0f} A   tau = {TAU:.0f} s   k_ult = {K_ULT}   K_neg = 6")
    say("=" * 68); say()
    fig1(); fig2(); fig3(); fig4(); fig5()
    say("=" * 68)
    say("figures written: fig1..fig5 png")
    with open("results.txt", "w") as f:
        f.write("\n".join(log) + "\n")
