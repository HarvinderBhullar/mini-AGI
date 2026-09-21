"""
The learning rate, governed by held-out loss rather than by a horizon.

A cosine schedule asserts that the run ends. For a model that reads
continually that assertion is false, and it fails in a specific way: when the
regime changes - a new corpus, a change of shape - a count of characters
already read says the model is finished when it has in fact just started
again. 

TWO RULES, AND THEY POINT IN OPPOSITE DIRECTIONS. That symmetry is the whole
point; a cosine can only ever go down.

  EASING    the rate is nudged at EVERY evaluation by an amount that varies
            smoothly with the evidence, rather than stepping every eighth one.
            The evidence is an exponentially weighted least-squares fit of
            held-out against evaluation index - no window, so nothing ever
            falls off an edge and produces a jump in the signal that becomes a
            jump in the rate.

            THE EVIDENCE IS AN EFFECT SIZE, NOT A t-STATISTIC. A t-statistic
            is the wrong variable here: se(slope) shrinks as the fit
            accumulates weight, so given enough observations ANY downward drift
            becomes significant. It measures how long the controller has been
            watching, not how much has improved, and pins the rate at the
            ceiling.

            So the deciding number is
                e = -slope / sigma          improvement per evaluation, in
                                            units of the residual scatter
            which is scale-free and does NOT grow as evidence accumulates. The
            significance test is kept as a CAP rather than as the signal, since
            acting on a slope that could be noise is still wrong, so the rate
            moves on min(t, EFFECT * e).

            A second fit at LAM_FAST caps it again. The slow fit remembers ~66
            evaluations, so when improvement stops it goes on reporting the old
            decline for tens of evaluations. The fast fit notices in about 12,
            and min() takes whichever is less impressed.

            The nudge itself is unchanged: exp(NUDGE * tanh((t - T_MID)/T_W)),
            about x0.991 per evaluation where there is no evidence, x1.009
            where improvement is provable, and continuous everywhere between.

  REGIME    held-out jumped by several standard errors and stayed there ->
            step the rate back up. New material, or a change of shape. It has
            to persist to count, so a single noisy evaluation cannot trigger
            it. This is the same detector `tools/plot_progress.py` uses to
            decide where to fit its trend.

Nothing here is a hyperparameter the user has to set. Both thresholds are read
off the measured standard error of the evaluation itself, so the controller
tightens automatically as the evaluation gets less noisy.
"""

import math
from collections import deque


def _sums():
    return dict(w=0.0, w2=0.0, x=0.0, y=0.0, xx=0.0, xy=0.0, yy=0.0)


class Plasticity:
    FLOOR = 0.05        # the rate is never allowed to reach zero
    CEIL = 1.0
    LAM = 0.97          # decay per evaluation; n_eff -> (1+L)/(1-L) = 65.7
    LAM_FAST = 0.85     # the second, shorter fit; n_eff -> 12.3. Not lower:
                        # below MIN_EFF the fast fit never engages at all and
                        # the cap silently stops existing.
    EFFECT = 45.0       # converts e into the same units as t, so the two can
                        # be compared by min(). Calibrated so that typical
                        # real-run evidence sits just BELOW T_MID, making
                        # gentle decay the resting posture.
    MIN_EFF = 12.0      # effective observations before it will act at all
    NUDGE = 0.005       # log-scale gain per evaluation, going UP
    NUDGE_DOWN = 0.025  # ...and coming down. Deliberately larger - see below.
    T_W_DOWN = 6.0      # the width on the way down, over the range the verdict
                        # actually reaches (it runs to about -12)
    T_MID = 2.2         # the verdict at which it neither eases up nor down.
                        # DELIBERATELY ABOVE THE MIDDLE, for two reasons.
                        #
                        # The errors are not symmetric. Overshooting the rate
                        # costs a fraction of a nat and tens of millions of
                        # characters to repair; undershooting only costs time.
                        # So slow-but-real progress reads as slight decay, and
                        # only clearly better progress buys more rate.
                        #
                        # And `e` has a bias to correct. It is slope over
                        # RESIDUAL SCATTER, and that scatter is the model's own
                        # checkpoint-to-checkpoint wobble rather than the
                        # evaluation's error. The wobble falls over a run as
                        # the pool stops churning, while the evaluation error
                        # barely moves - so a run that merely gets QUIETER
                        # starts scoring as a run that is improving. Sitting
                        # the neutral point high is what absorbs that.
                        #
                        # This does not disable raising: real evidence still
                        # clears it, a few evaluations in ten rather than a
                        # third of them.
    T_W = 0.75          # how sharply it responds ABOVE that
    JUMP_SE = 4.0       # standard errors that count as a regime change
    HOLD_SE = 2.0       # ...and it has to still be up here next time
    FLOOR_JUMP = 0.25   # a jump this large is a regime change whatever the noise
    UP = 2.0            # what a confirmed regime change restores
    WARMUP = 100        # optimiser steps, in case the moments are not restored

    def __init__(self, scale=1.0, best=None):
        self.scale = float(scale)
        # Two exponentially weighted least-squares fits of held-out against
        # evaluation index, kept as running sums. There is no window: every
        # past reading still counts, weighted by LAM ** age, so nothing ever
        # falls off an edge. A hard window of length N steps whenever its
        # oldest point drops out, which is a jump in the SIGNAL that then
        # becomes a jump in the rate - the staircase this replaced.
        #
        # The slow fit is the evidence. The fast one exists only to notice
        # sooner when improvement has stopped, and can only ever lower the
        # verdict - see observe().
        self.S = _sums()
        self.F = _sums()
        self.i = 0.0                   # evaluation counter, the x axis
        self.se_hist = deque(maxlen=64)
        self.prev = None
        self.jump_from = None          # a candidate regime change, unconfirmed
        self.events = []
        self.step = 0
        self.last_t = 0.0
        self.last_e = 0.0

    # ---------------------------------------------------------------- lr
    def factor(self):
        """What to multiply every group's base rate by, right now."""
        w = min(1.0, (self.step + 1) / self.WARMUP) if self.step < self.WARMUP \
            else 1.0
        return self.scale * w

    def tick(self):
        self.step += 1

    # ------------------------------------------------------------ evidence
    def _se(self):
        if not self.se_hist:
            return 0.0
        s = sorted(self.se_hist)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def _accumulate(self, y):
        """Add one reading to BOTH fits. They share the x axis."""
        self.i += 1.0
        x = self.i
        for S, lam in ((self.S, self.LAM), (self.F, self.LAM_FAST)):
            for k in ("w", "x", "y", "xx", "xy", "yy"):
                S[k] *= lam
            S["w2"] *= lam * lam
            S["w"] += 1.0
            S["w2"] += 1.0
            S["x"] += x
            S["y"] += y
            S["xx"] += x * x
            S["xy"] += x * y
            S["yy"] += y * y

    @staticmethod
    def _fit(S):
        """
        (t, n_eff, e) for one set of running sums.

        t is the slope over its own standard error - the significance. e is
        the slope over the RESIDUAL SCATTER - the effect size, improvement per
        evaluation in units of the noise it has to be seen through. Both are
        signed so that positive means held-out is falling.

        The difference between them is the whole point of this file. t carries
        a sqrt(sxx) that grows until the window fills and then stays large, so
        t rewards patience rather than progress. e carries no such factor.
        """
        if S["w"] <= 2:
            return 0.0, 0.0, 0.0
        n_eff = S["w"] ** 2 / max(S["w2"], 1e-12)
        sxx = S["xx"] - S["x"] ** 2 / S["w"]
        sxy = S["xy"] - S["x"] * S["y"] / S["w"]
        syy = S["yy"] - S["y"] ** 2 / S["w"]
        if sxx <= 0 or n_eff <= 2:
            return 0.0, n_eff, 0.0
        slope = sxy / sxx
        resid = max(syy - slope * sxy, 0.0)
        sigma = math.sqrt(resid / (n_eff - 2)) if n_eff > 2 else 0.0
        if sigma <= 0:
            # A perfectly straight run has no residual, so neither statistic
            # is defined. Report nothing rather than infinity; real held-out
            # is never this clean, and a run that is has no noise to see
            # through and needs no help from here.
            return 0.0, n_eff, 0.0
        t = -slope / (sigma / math.sqrt(sxx))
        e = -slope / sigma
        return max(-50.0, min(50.0, t)), n_eff, e

    def _trend(self):
        """(t, effective observations) for the slow fit. Kept for callers."""
        t, n_eff, _ = self._fit(self.S)
        return t, n_eff

    def _verdict(self):
        """
        The number the rate actually moves on, and the two it is built from.

        min() of three readings, because each can only ever say "less than you
        think" and none may say "more":

          t_slow          is it distinguishable from noise at all - the
                          significance guard, taken from the better-powered
                          fit because that is what it is for
          EFFECT * e_slow is it big enough to be worth a higher rate
          EFFECT * e_fast ...and is it still happening now

        Only the SLOW t appears. t depends on how much window it was measured
        over - se(slope) shrinks as sqrt(sxx) - so a deliberately short fit has
        a mechanically small t and would veto everything regardless of what the
        loss is doing. The fast fit contributes its EFFECT SIZE instead, which
        does not know how long the window was.
        """
        t_s, n_s, e_s = self._fit(self.S)
        _, n_f, e_f = self._fit(self.F)
        t = min(t_s, self.EFFECT * e_s)
        if n_f >= self.MIN_EFF:
            t = min(t, self.EFFECT * e_f)
        return max(-50.0, min(50.0, t)), n_s, e_s

    def observe(self, val, se=None):
        """
        One held-out evaluation. Returns a note if the rate moved notably.

        The rate is nudged EVERY time, by an amount that varies smoothly with
        the evidence: exp(NUDGE * tanh((t - T_MID) / T_W)). Deep in "no
        evidence" that is about x0.9917 per evaluation, compounding to x0.935
        over eight - a drift rather than a staircase, so nothing the rate does
        is ever a shock to the run.
        """
        if val is None or not math.isfinite(float(val)):
            return None
        val = float(val)
        if se is not None and math.isfinite(float(se)) and float(se) > 0:
            self.se_hist.append(float(se))
        s = self._se()
        note = None

        # ---- REGIME: a jump, confirmed on the following evaluation --------
        if self.jump_from is not None:
            if val > self.jump_from + self.HOLD_SE * max(s, 1e-9):
                before = self.scale
                self.scale = min(self.CEIL, self.scale * self.UP)
                note = (f"held-out moved to {val:.4f} from {self.jump_from:.4f} "
                        f"and stayed - new regime, rate {before:.3f} -> "
                        f"{self.scale:.3f}")
                self.events.append({"at": self.step, "kind": "regime",
                                    "val": val, "scale": self.scale})
                self.S = _sums()
                self.F = _sums()
                self.i = 0.0
            self.jump_from = None
        elif self.prev is not None:
            bar = max(self.JUMP_SE * s, self.FLOOR_JUMP)
            if val - self.prev > bar:
                self.jump_from = self.prev      # confirm or discard next time

        self.prev = val
        self._accumulate(val)
        t, n_eff, e = self._verdict()
        self.last_t = t
        self.last_e = e

        # ---- the nudge, every time, sized by the evidence ------------------
        #
        # ASYMMETRIC, AND THAT IS THE POINT. Going up and coming down are not
        # the same question. A rate that is too high SHOWS you - held-out
        # turns and keeps turning - so deterioration is direct evidence of
        # overshoot and should be corrected in proportion to how bad it is. A
        # rate that is merely working tells you nothing about whether a higher
        # one would work better, so upward is a slow probe and stays at NUDGE.
        #
        # The two halves therefore need different WIDTHS, not just different
        # gains. tanh saturates by |arg| = 2, so a single narrow width would
        # make every verdict below about -0.3 produce an identical step: the
        # controller would see a large fall and a small one and answer both at
        # the same speed. T_W_DOWN spreads the response across the range the
        # verdict actually reaches (down to roughly -12), so how bad the
        # deterioration is reaches the rate.
        if note is None and n_eff >= self.MIN_EFF:
            before = self.scale
            up = t >= self.T_MID
            gain = self.NUDGE if up else self.NUDGE_DOWN
            width = self.T_W if up else self.T_W_DOWN
            f = math.exp(gain * math.tanh((t - self.T_MID) / width))
            self.scale = max(self.FLOOR, min(self.CEIL, self.scale * f))
            # One line per evaluation would be noise. Record only when the
            # rate has drifted a full 2% since the last thing recorded.
            last = self.events[-1]["scale"] if self.events else 1.0
            if abs(math.log(self.scale / max(last, 1e-9))) > 0.02:
                kind = ("improving" if t > self.T_MID else
                        "deteriorating" if t < -self.T_MID else "settling")
                note = (f"{kind}: t={t:+.2f} (effect {e:+.3f} per evaluation) "
                        f"over {n_eff:.0f} effective evaluations, rate "
                        f"{before:.3f} -> {self.scale:.3f}")
                self.events.append({"at": self.step, "kind": kind,
                                    "val": val, "scale": self.scale,
                                    "t": round(t, 2)})
        return note

    # ------------------------------------------------------------- restart
    def state(self):
        t, n_eff, e = self._verdict()
        return {"scale": self.scale, "prev": self.prev,
                "S": dict(self.S), "F": dict(self.F), "i": self.i,
                "se": list(self.se_hist), "step": self.step,
                "n": round(n_eff, 1), "t": round(t, 3), "e": round(e, 4),
                "events": self.events[-40:]}

    @classmethod
    def restore(cls, d):
        p = cls()
        if not d:
            return p
        p.scale = float(d.get("scale", 1.0))
        p.prev = d.get("prev")
        if isinstance(d.get("S"), dict):
            p.S.update({k: float(v) for k, v in d["S"].items() if k in p.S})
            p.i = float(d.get("i", 0.0) or 0.0)
            if isinstance(d.get("F"), dict):
                p.F.update({k: float(v) for k, v in d["F"].items()
                            if k in p.F})
            # A checkpoint that carries no F leaves the fast fit empty, and
            # that is correct. Seeding it from the slow fit would copy the slow
            # fit's n_eff with it, so the "fast" reading would just be the slow
            # one again until it decayed and the cap would never bind. Empty
            # costs nothing: MIN_EFF gates only this cap, not the controller,
            # so the slow fit keeps steering throughout.
        elif d.get("hist"):
            # an older checkpoint kept a plain window; replay it so the fit
            # starts from the evidence that was already gathered
            for v in d["hist"]:
                p._accumulate(float(v))
        # The warmup is there in case the moments were lost. They ARE restored
        # here, so re-running it on every resume only puts a notch in the rate
        # that nothing asked for.
        p.step = int(d.get("step", 0) or 0)
        for v in d.get("se") or []:
            p.se_hist.append(float(v))
        p.events = list(d.get("events") or [])
        return p

    def describe(self):
        s = self._se()
        return (f"learning rate is governed by held-out, not by a horizon: "
                f"rate x{self.scale:.3f}, floor x{self.FLOOR}, "
                f"noise {s:.4f}" if s else
                f"learning rate is governed by held-out, not by a horizon: "
                f"rate x{self.scale:.3f}, floor x{self.FLOOR}")
