"""
Digital Motor Protection Relay - thermal replica overload element
=================================================================
Elements implemented:
    49   thermal overload (first-order thermal replica, IEC 60255-style)
    46   phase unbalance / phase loss (negative sequence)
    51   stall / locked rotor (definite time, after start window)
    37   loss of load (undercurrent alarm)
         restart inhibit based on thermal state; thermal alarm hysteresis

Model
-----
    dTheta/dt = (1/tau) * (M^2 - Theta),      M = I_eq / (k_ult * I_set)
    trip when Theta >= 1

    closed form:  t_trip = tau * ln[ (M^2 - Theta0) / (M^2 - 1) ]
    cold start :  t_trip = tau * ln[  M^2         / (M^2 - 1) ]

Equivalent heating current (negative sequence weighted):
    I_eq^2 = I1^2 + K * I2^2          K ~ 3..6

Run this file directly to (a) reproduce the tau calibration table,
(b) unit-test the integrator against the closed form, and (c) run the
duty-cycle scenarios.

The current implementation intentionally uses analytically generated phase
current profiles and a single equivalent thermal state. Industrial extensions
are discussed in the README rather than hidden behind unimplemented claims.
"""

import math
import cmath

# ----------------------------------------------------------------------
# 1. Calibration: derive tau from the IEC 60947-4-1 trip class window
# ----------------------------------------------------------------------

def tau_from_class(t_trip_at_test, test_multiple=7.2, k_ult=1.05, theta0=0.0):
    """Thermal time constant required to trip in t_trip_at_test seconds
    at `test_multiple` x setting current, from an initial thermal state theta0.

    IEC 60947-4-1 defines trip class by a cold-start test at 7.2 x setting:
        class 10A -> 2..10 s,  class 10 -> 4..10 s,
        class 20  -> 6..20 s,  class 30 -> 9..30 s
    """
    M2 = (test_multiple / k_ult) ** 2
    return t_trip_at_test / math.log((M2 - theta0) / (M2 - 1.0))


def trip_time_closed_form(M, tau, theta0=0.0):
    """Analytic trip time. Returns math.inf if the model never trips."""
    M2 = M * M
    if M2 <= 1.0 or M2 <= theta0:
        return math.inf
    return tau * math.log((M2 - theta0) / (M2 - 1.0))


# ----------------------------------------------------------------------
# 2. Symmetrical components
# ----------------------------------------------------------------------

A = cmath.exp(2j * math.pi / 3.0)          # 1 angle 120 deg


def sequence_components(Ia, Ib, Ic):
    """Ia, Ib, Ic are complex phasors. Returns (I0, I1, I2) magnitudes."""
    I0 = (Ia + Ib + Ic) / 3.0
    I1 = (Ia + A * Ib + A * A * Ic) / 3.0
    I2 = (Ia + A * A * Ib + A * Ic) / 3.0
    return abs(I0), abs(I1), abs(I2)


def balanced(I):
    """Balanced POSITIVE-sequence set for a magnitude I.
    Positive sequence means b lags a by 120 deg: (I, I*a^2, I*a).
    Getting this backwards gives a pure negative-sequence set, which the
    relay correctly reads as a severe fault - a useful sanity check.
    """
    return complex(I), complex(I) * A * A, complex(I) * A


def equivalent_heating_current(I1, I2, K=6.0):
    """I_eq^2 = I1^2 + K*I2^2.

    K is a lumped modelling factor representing the relative heating effect
    of negative-sequence current. Running at low slip, the negative-sequence
    rotor resistance can be much higher than the positive-sequence value, so
    the same current magnitude can produce substantially more rotor heating.
    This implementation uses K=6 as a conservative literature-informed
    assumption; it is not an IEC-prescribed constant.
    """
    return math.sqrt(I1 * I1 + K * I2 * I2)


# ----------------------------------------------------------------------
# 3. The relay
# ----------------------------------------------------------------------

class MotorRelay:
    def __init__(self,
                 I_set,                  # A, motor full load current
                 tau_heat,               # s, heating time constant
                 tau_cool_ratio=4.0,     # cooling is slower: fan stops with motor
                 k_ult=1.05,             # ultimate trip current factor
                 K_neg=6.0,              # negative sequence heating weight
                 alarm_level=0.90,       # thermal alarm ON threshold
                 alarm_hysteresis=0.05,  # thermal alarm OFF band (5%)
                 restart_level=0.60,     # restart inhibit clears below this
                 stall_multiple=3.0,     # definite-time stall pickup
                 stall_delay=1.0,        # s
                 start_window=8.0,       # s, start current tolerated this long
                 underload_multiple=0.5):
        self.I_set = I_set
        self.tau_heat = tau_heat
        self.tau_cool = tau_heat * tau_cool_ratio
        self.k_ult = k_ult
        self.K_neg = K_neg
        self.alarm_level = alarm_level
        self.alarm_hysteresis = alarm_hysteresis
        if not 0.0 <= self.alarm_hysteresis < 1.0:
            raise ValueError("alarm_hysteresis must be in [0, 1)")
        self.alarm_off_level = alarm_level * (1.0 - self.alarm_hysteresis)
        self.restart_level = restart_level
        self.stall_multiple = stall_multiple
        self.stall_delay = stall_delay
        self.start_window = start_window
        self.underload_multiple = underload_multiple

        self.reset_state()

    # -- state ---------------------------------------------------------
    def reset_state(self):
        self.theta = 0.0            # thermal state, trip at 1.0
        self.tripped = False
        self.trip_reason = None
        self.trip_time = None
        self.alarm = False
        self.thermal_alarm = False
        self.t = 0.0
        self.run_time = 0.0         # time since the motor was energised
        self.stall_timer = 0.0
        self.history = []           # (t, theta, I_eq, running)

    @property
    def restart_permitted(self):
        """True when a new motor start is allowed by thermal state."""
        return (not self.tripped) and self.theta < self.restart_level

    def request_start(self):
        """Request a motor start; returns False when thermal restart is blocked."""
        if self.restart_permitted:
            self.run_time = 0.0
            self.stall_timer = 0.0
            return True
        return False

    def clear_thermal_capacity(self):
        """Force thermal state to zero for an emergency restart demonstration.

        This is intentionally separate from normal restart permission and should
        only be used as a simulation of an emergency override."""
        self.theta = 0.0

    # -- main loop -----------------------------------------------------
    def update(self, Ia, Ib, Ic, dt, running=True):
        """Ia,Ib,Ic: complex phasors (or floats for a balanced set).
        Call once per measurement cycle."""
        self.t += dt

        if isinstance(Ia, (int, float)):
            Ia, Ib, Ic = complex(Ia), complex(Ib), complex(Ic)

        _, I1, I2 = sequence_components(Ia, Ib, Ic)
        I_eq = equivalent_heating_current(I1, I2, self.K_neg)

        # --- thermal replica (element 49) ---
        M = I_eq / (self.k_ult * self.I_set)
        tau = self.tau_heat if running else self.tau_cool
        self.theta += (dt / tau) * (M * M - self.theta)
        self.theta = max(self.theta, 0.0)

        if running:
            self.run_time += dt
        else:
            self.run_time = 0.0
            self.stall_timer = 0.0

        self.history.append((self.t, self.theta, I_eq, running))

        if self.tripped:
            return

        # Thermal alarm hysteresis: turn on at alarm_level and remain on
        # until theta falls below the lower release threshold.
        if self.theta >= self.alarm_level:
            self.thermal_alarm = True
        elif self.theta < self.alarm_off_level:
            self.thermal_alarm = False
        self.alarm = self.thermal_alarm

        if self.theta >= 1.0:
            self._trip("49 thermal overload")
            return

        # --- phase loss / unbalance (element 46) ---
        if I1 > 0.1 * self.I_set and (I2 / I1) > 0.5:
            self._trip("46 phase loss / severe unbalance")
            return

        # --- stall / locked rotor (element 51, definite time) ---
        # only armed after the start window, so a healthy DOL start is ignored
        I_max = max(abs(Ia), abs(Ib), abs(Ic))
        if running and self.run_time > self.start_window and \
           I_max > self.stall_multiple * self.I_set:
            self.stall_timer += dt
            if self.stall_timer >= self.stall_delay:
                self._trip("51 stall / locked rotor")
                return
        else:
            self.stall_timer = 0.0

        # --- loss of load (element 37, alarm only) ---
        if running and self.run_time > self.start_window and \
           I_max < self.underload_multiple * self.I_set:
            self.alarm = True

    def _trip(self, reason):
        self.tripped = True
        self.trip_reason = reason
        self.trip_time = self.t


# ----------------------------------------------------------------------
# 4. Curve generator: trip time vs current multiple
# ----------------------------------------------------------------------

def trip_curve(tau, k_ult=1.05, theta0=0.0, lo=1.06, hi=10.0, n=200):
    """Return (multiples_of_setting, trip_times). Uses the closed form."""
    xs, ys = [], []
    for i in range(n):
        x = lo * (hi / lo) ** (i / (n - 1))        # log spacing
        t = trip_time_closed_form(x / k_ult, tau, theta0)
        xs.append(x)
        ys.append(t)
    return xs, ys


IEC_CLASS_WINDOWS = {      # class : (t_min, t_max) at 7.2 x setting, cold
    "10A": (2.0, 10.0),
    "10":  (4.0, 10.0),
    "20":  (6.0, 20.0),
    "30":  (9.0, 30.0),
}


# ----------------------------------------------------------------------
# 5. Self-test and scenarios
# ----------------------------------------------------------------------

def _print_calibration_table(k_ult=1.05):
    print(f"\nTAU CALIBRATION  (test point 7.2 x setting, cold, k_ult={k_ult})")
    print("-" * 62)
    print(f"{'class':<6}{'t_min':>7}{'tau_min':>10}{'t_max':>8}{'tau_max':>10}")
    for cls, (tmin, tmax) in IEC_CLASS_WINDOWS.items():
        print(f"{cls:<6}{tmin:>7.1f}{tau_from_class(tmin, k_ult=k_ult):>10.0f}"
              f"{tmax:>8.1f}{tau_from_class(tmax, k_ult=k_ult):>10.0f}")


def _unit_test_integrator(tau, k_ult=1.05):
    """Numerically integrated trip time must match the closed form."""
    print(f"\nUNIT TEST: integrator vs closed form   (tau={tau:.0f} s)")
    print("-" * 62)
    print(f"{'I/Iset':>8}{'closed form':>14}{'integrated':>13}{'error':>10}")
    I_set = 10.0
    for mult in (1.2, 1.5, 2.0, 4.0, 7.2, 10.0):
        analytic = trip_time_closed_form(mult / k_ult, tau)
        relay = MotorRelay(I_set=I_set, tau_heat=tau, k_ult=k_ult, K_neg=6.0,
                           start_window=1e9)   # disable stall element here
        dt, I = 0.02, mult * I_set
        while not relay.tripped and relay.t < 4 * 3600:
            relay.update(*balanced(I), dt)
        num = relay.trip_time
        err = 100.0 * (num - analytic) / analytic
        print(f"{mult:>8.2f}{analytic:>14.2f}{num:>13.2f}{err:>9.3f}%")


def _unit_test_alarm_hysteresis(tau, k_ult=1.05):
    """Verify the thermal alarm's ON/OFF thresholds are genuinely hysteretic."""
    print("\nUNIT TEST: thermal alarm hysteresis")
    print("-" * 62)
    relay = MotorRelay(I_set=10.0, tau_heat=tau, k_ult=k_ult, start_window=1e9)
    dt = 0.02

    # Force the state just below the ON threshold, then use a high current
    # sample to cross it; the alarm must latch while theta remains in the band.
    relay.theta = relay.alarm_level - 0.001
    relay.update(*balanced(6.5 * relay.I_set), dt, running=True)
    assert relay.theta >= relay.alarm_level and relay.thermal_alarm
    theta_after_on = relay.theta

    # A zero-current sample cools the state but remains above the OFF threshold;
    # the alarm must stay ON rather than chatter OFF.
    relay.update(*balanced(0.0), dt, running=False)
    assert relay.theta > relay.alarm_off_level and relay.thermal_alarm
    theta_mid_band = relay.theta

    # Below the release threshold the alarm must turn OFF.
    relay.theta = relay.alarm_off_level - 0.001
    relay.update(*balanced(0.0), dt, running=False)
    assert relay.theta < relay.alarm_off_level and not relay.thermal_alarm
    print(f"  ON threshold  = {relay.alarm_level:.3f}")
    print(f"  OFF threshold = {relay.alarm_off_level:.3f}")
    print(f"  crossed ON at theta = {theta_after_on:.4f} -> alarm ON")
    print(f"  cooled within hysteresis band to theta = {theta_mid_band:.4f} -> alarm remains ON")
    print(f"  below OFF threshold -> alarm OFF")
    print("  -> PASS")


def _scenario_no_trip_at_105(tau, k_ult=1.05):
    print(f"\nSCENARIO: sustained 1.05 x setting  (standard: must NOT trip)")
    print("-" * 62)
    t = trip_time_closed_form(1.05 / k_ult, tau)
    print(f"  closed form trip time = {t}   -> M^2-1 = 0, asymptote at pickup")
    print(f"  and at 1.20 x setting = {trip_time_closed_form(1.20/k_ult, tau):.0f} s "
          f"({trip_time_closed_form(1.20/k_ult, tau)/60:.1f} min), "
          "standard requires trip within 2 h  -> OK")


def _scenario_hot_vs_cold(tau, k_ult=1.05):
    theta0_hot = 1.0 / (k_ult ** 2)     # previously running at rated current
    print(f"\nSCENARIO: cold vs hot curves  (theta0_hot={theta0_hot:.3f})")
    print("-" * 62)
    print(f"{'I/Iset':>8}{'cold (s)':>12}{'hot (s)':>11}{'ratio':>9}")
    for mult in (1.5, 2.0, 4.0, 7.2):
        c = trip_time_closed_form(mult / k_ult, tau, 0.0)
        h = trip_time_closed_form(mult / k_ult, tau, theta0_hot)
        print(f"{mult:>8.2f}{c:>12.1f}{h:>11.2f}{c/h:>9.1f}")
    print("  NOTE: the single-state model produces a large hot/cold separation.")
    print("  One thermal mass cannot fully represent a fast-heating winding and")
    print("  a slower frame. A multi-state thermal model would reduce this limitation.")


def _scenario_repeated_starts(tau):
    """Thermal memory plus an enforced thermal restart lockout."""
    print("\nSCENARIO: repeated DOL starts (thermal memory + restart inhibit)")
    print("-" * 62)
    I_set = 10.0
    relay = MotorRelay(I_set=I_set, tau_heat=tau, start_window=1e9)
    dt = 0.02

    for start_no in range(1, 4):
        allowed = relay.request_start()
        print(f"  start request {start_no}: {'ALLOWED' if allowed else 'BLOCKED'} "
              f"(theta before = {relay.theta:.3f})")
        if not allowed:
            break
        for _ in range(int(3.0 / dt)):
            I = 6.5 * I_set
            relay.update(*balanced(I), dt, running=True)
            if relay.tripped:
                break
        if relay.tripped:
            break
        for _ in range(int(30.0 / dt)):
            relay.update(*balanced(0.0), dt, running=False)
        print(f"    after start {start_no}: theta = {relay.theta:.3f} "
              f"{'[ALARM]' if relay.thermal_alarm else ''}")

    if not relay.tripped:
        blocked = not relay.request_start()
        print(f"  4th start request at theta={relay.theta:.3f}: "
              f"{'BLOCKED by restart inhibit' if blocked else 'ALLOWED'}")
        if blocked:
            cool_time = 0.0
            while not relay.restart_permitted and cool_time < 600.0:
                relay.update(*balanced(0.0), dt, running=False)
                cool_time += dt
            print(f"  cooled for {cool_time:.1f} s -> theta = {relay.theta:.3f}, "
                  f"restart {'PERMITTED' if relay.restart_permitted else 'STILL BLOCKED'}")

    if relay.tripped:
        print(f"  -> {relay.trip_reason} at t={relay.trip_time:.1f} s")
    else:
        print("  -> no thermal trip in this lockout demonstration")
    print("  Thermal memory accumulates heat across starts; the implemented")
    print("  restart lockout prevents a new start while theta is above 0.60.")


def _scenario_single_phasing(tau):
    print("\nSCENARIO: one phase lost while running")
    print("-" * 62)
    I_set = 10.0
    # delta / ungrounded: I0 = 0, remaining two phases carry equal & opposite
    Ia = 1.6 * I_set + 0j
    Ib = -1.6 * I_set + 0j
    Ic = 0 + 0j
    _, I1, I2 = sequence_components(Ia, Ib, Ic)
    for K in (0.0, 3.0, 6.0):
        I_eq = equivalent_heating_current(I1, I2, K)
        M = I_eq / (1.05 * I_set)
        t = trip_time_closed_form(M, tau)
        label = "no compensation" if K == 0 else f"K={K:.0f}"
        print(f"  {label:<18} I1={I1:6.2f} I2={I2:6.2f} "
              f"I_eq={I_eq:6.2f} A  M={M:5.2f}  trip in {t:8.1f} s")
    print("  I2/I1 = 1.0 here, so the full relay's element 46 phase-loss test")
    print("  fires first. The K=6, ~82 s value above is a thermal-only sensitivity")
    print("  result, not the full relay trip time.")


def _scenario_start_then_overload(tau):
    print("\nSCENARIO: DOL start -> rated run -> 1.3 x overload")
    print("-" * 62)
    I_set = 10.0
    relay = MotorRelay(I_set=I_set, tau_heat=tau)
    dt = 0.02

    def run(seconds, mult, running=True):
        for _ in range(int(seconds / dt)):
            I = mult * I_set
            relay.update(*balanced(I), dt, running=running)
            if relay.tripped:
                return True
        return False

    run(2.0, 6.5);   print(f"  end of 2 s start at 6.5x : theta = {relay.theta:.3f}")
    run(300.0, 1.0); print(f"  after 5 min at rated     : theta = {relay.theta:.3f}")
    tripped = run(3600.0, 1.3)
    print(f"  1.3x overload applied at t = {2.0+300.0:.0f} s")
    if tripped:
        print(f"  -> {relay.trip_reason} at t = {relay.trip_time:.0f} s "
              f"({(relay.trip_time - 302.0)/60:.1f} min after the overload began)")
    else:
        print(f"  -> no trip within 1 h, theta = {relay.theta:.3f}")


if __name__ == "__main__":
    K_ULT = 1.05
    _print_calibration_table(K_ULT)

    TAU = 400.0        # chosen inside the class 10 window
    print(f"\nCHOSEN tau = {TAU:.0f} s  -> class 10 "
          f"(window {tau_from_class(4.0):.0f}..{tau_from_class(10.0):.0f} s), "
          f"trip at 7.2x = {trip_time_closed_form(7.2/K_ULT, TAU):.2f} s")

    _unit_test_integrator(TAU, K_ULT)
    _unit_test_alarm_hysteresis(TAU, K_ULT)
    _scenario_no_trip_at_105(TAU, K_ULT)
    _scenario_hot_vs_cold(TAU, K_ULT)
    _scenario_start_then_overload(TAU)
    _scenario_repeated_starts(TAU)
    _scenario_single_phasing(TAU)
    print()
