"""标注域数据层 —— 2026-11-05 自 ulabel py 镜像层回迁（Plan 2 D7）。

模块族（回迁时保持 ulabel py 侧的模块划分与命名）：
- core 镜像四件之三：``label`` / ``meta`` / ``geometry`` —— rs/ulabel-core 的
  Python 消费侧类型（真源在 Rust，契约见 ulabel docs/CONTRACT.md）；IO（load/save）
  无 jxl 消费方，未回迁。
- jxl 现役格式：``meta_jxl``（jxl meta 体系）/ ``a2d`` / ``prop`` / ``dataset`` /
  ``formats``（darknet 等外部训练格式）。
- 像素域裁片：``crop``（rs ulabel-utils/crop.rs 对齐镜像）。
- ``service``：ulabel-service 确认 API 的 REST 客户端（HTTP 边界消费方）。

类型域纪律：本包全部为【标注域】类型（多边形 + 属性表）；检测域类型
（rect 矩形、无属性表，如 ``jxl.det.d2d.D2dObject``）禁止进入本包——
检测域住 ``jxl.det``，转换只发生在显式转换器（``jxl.det.d2d_converters``）。
"""
