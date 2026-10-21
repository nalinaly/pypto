# `pl.system`

同步、cache 与跨核原语。它们是产生副作用的语句而非产值算子 —— 多数没有返回值。

`pl.system.signal_wait(signal, [offset], expected, backoff_ticks=150)` 可为 A3
本地 GM `INT32` 信号的未成功轮询设置编译期退避间隔。单位为设备计时器 tick，
不是微秒。第一次读取立即执行，每次读取仍使信号缓存行失效，保留不消费信号的
`>=` 等待语义；不能替代业务数据的发布与获取。默认值 0 保留原 TWAIT lowering。
正值需要 PTOAS 支持 `pto.comm.twait` 的 `backoffTicks` 属性以及正常递增的硬件
计时器，不支持计时器始终为零的模拟器。

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
