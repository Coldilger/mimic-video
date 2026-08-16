"""Experiment 3 (cost per decision) instrumentation for mimic-video. Does
not modify video_action_model.py or any of the Experiment 1 wrapper
subclasses -- wraps VAMInference.step (and subclasses') with a timer via a
class decorator applied at import time (monkey-patch on the class, not the
shared source file), recording per-replan wall-clock latency into a list on
the instance.

Why wrap `step` and gate on action_buffer, rather than a narrower method:
mimic's world-model + action-decoder forward pass is inlined directly in
step() (video_action_model.py:172-199, the `if self.action_buffer is None`
block) -- there's no separate "_predict_new_chunk"-style method to wrap the
way F1-VLA's experiment3_cost/timing_wrapper.py does. Checking whether
action_buffer is None *before* calling the original step() cleanly
identifies calls that trigger a real forward pass (a replan) vs. calls that
just pop the next action off an already-computed chunk -- matching
../README's "one full chunk-worth decision" semantics, not every single 5Hz
control tick.
"""

from __future__ import annotations

import time

import torch


def add_timing(cls):
    """Class decorator: wraps cls.step with a timer, active only on calls
    that trigger a real replan. Records into self._chunk_latencies_s
    (created lazily), does not change return value or behavior."""
    original = cls.step

    def timed(self, *args, **kwargs):
        if not hasattr(self, "_chunk_latencies_s"):
            self._chunk_latencies_s = []
        will_replan = self.action_buffer is None
        if will_replan:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        if will_replan:
            torch.cuda.synchronize()
            self._chunk_latencies_s.append(time.perf_counter() - t0)
        return result

    cls.step = timed
    return cls
