# `pl.system`

Synchronization, cache and cross-core primitives. These are side-effecting statements
rather than value-producing operators — most return nothing.

`pl.system.signal_wait(signal, [offset], expected, backoff_ticks=150)` optionally
delays unsuccessful polling of a local GM `INT32` signal on A3. The compile-time
interval is measured in device counter ticks, not microseconds. The first read
is immediate; each read still invalidates the signal cache line. This preserves
non-consuming `>=` waiting and does not replace payload publication or acquisition.
Zero (the default) retains the existing TWAIT lowering. Positive values require
PTOAS support for `pto.comm.twait`'s `backoffTicks` attribute and an advancing
hardware counter; they do not support simulators whose counter remains zero.

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
