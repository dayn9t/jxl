"""PeopleNet Transformer (RT-DETR) ONNX 手术: MDA plugin 节点 → GridSample 等价子图。

背景: resnet50_peoplenet_transformer_op17.onnx 含 12 个 nvidia::MultiscaleDeformableAttnPlugin_TRT
节点 (6 encoder self-attn + 6 decoder cross-attn), ONNX Runtime 无该 TRT plugin 无法加载
(全系统不存在 MDA plugin, 见 ml 仓 docs/peoplenet/roadmap.md Phase 3)。

本脚本把每个 MDA 节点替换为标准 ONNX 算子子图 (GridSample + 加权求和), 语义与 mmcv
ms_deform_attn CUDA kernel 一致: 逐层双线性采样 (pixel = loc * W - 0.5, 越界补零,
等价 grid_sample align_corners=0) + attention_weights 加权求和。子图形状自描述
(heads/levels/points 运行时从 Shape(sampling_locations) 读取), 仅 spatial_shapes /
level_start_index 沿 producer 链解析为常量烘焙。

验证闭环 (避免同源对照):
    verify-unit — 子图 vs 手写双线性插值参考 (独立于 GridSample 的算术展开);
    verify-e2e  — 术后全模型端到端推理 + 画框 (布局约定错误的兜底检验)。

用法:
    uv run python -m jxl.bin.rtdetr_mda_surgery surgery <in.onnx> <out.onnx>
    uv run python -m jxl.bin.rtdetr_mda_surgery verify-unit
    uv run python -m jxl.bin.rtdetr_mda_surgery verify-e2e <in.onnx> <image> <out.jpg>
"""

from __future__ import annotations

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import typer
from loguru import logger
from onnx import TensorProto, helper, numpy_helper

app = typer.Typer(help="RT-DETR MDA plugin 手术: nvidia 域节点 → GridSample 等价子图")

MDA_OP = "MultiscaleDeformableAttnPlugin_TRT"
MDA_DOMAIN = "nvidia"
INPUT_H, INPUT_W = 544, 960
MEAN_RGB = np.array([123.675, 116.280, 103.53], dtype=np.float32)
SCALE = np.float32(0.017507)


# ---------------------------------------------------------------- 常量池


class ConstPool:
    """共享 initializer 池: 按 (dtype, bytes, shape) 去重, 12 个子图共用同名常量。"""

    def __init__(self) -> None:
        self._tensors: dict[tuple, TensorProto] = {}

    def get(self, arr: np.ndarray) -> str:
        arr = np.ascontiguousarray(arr)
        key = (str(arr.dtype), arr.tobytes(), arr.shape)
        if key not in self._tensors:
            name = f"gs_c{len(self._tensors)}"
            self._tensors[key] = numpy_helper.from_array(arr, name)
        return self._tensors[key].name

    def initializers(self) -> list[TensorProto]:
        return list(self._tensors.values())


# ---------------------------------------------------------------- 常量溯源


def trace_const(
    producers: dict[str, onnx.NodeProto],
    initializers: dict[str, np.ndarray],
    name: str,
    depth: int = 0,
) -> np.ndarray | None:
    """沿 producer 链解析编译期常量 (initializer/Constant/Cast/Reshape); 不可解析返回 None。"""
    if depth > 8:
        return None
    if name in initializers:
        return initializers[name]
    node = producers.get(name)
    if node is None:
        return None
    if node.op_type == "Constant":
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
        return None
    if node.op_type == "Cast":
        return trace_const(producers, initializers, node.input[0], depth + 1)
    if node.op_type == "Reshape":
        v = trace_const(producers, initializers, node.input[0], depth + 1)
        s = trace_const(producers, initializers, node.input[1], depth + 1)
        if v is not None and s is not None:
            return v.reshape(s)
    return None


# ---------------------------------------------------------------- 子图构建


def build_mda_subgraph(
    scope: str,
    value: str,
    sampling_locations: str,
    attention_weights: str,
    output: str,
    spatial_shapes: np.ndarray,
    level_starts: np.ndarray,
    pool: ConstPool,
) -> list[onnx.NodeProto]:
    """生成与 MDA plugin 等价的子图节点。

    输入约定 (mmdeploy msa2trt 导出, 已经 shape inference + 运行时错误双重实证):
        value [bs, Len_v, heads, head_dim] (已头拆分, 4-D),
        sampling_locations [bs, Q, heads, L, P, 2] (归一化 w,h),
        attention_weights [bs, Q, heads, L, P];
    输出 [bs, Q, heads*head_dim] (head-major 展平, mmdeploy TRT MSA 插件约定)。
    spatial_shapes [L,2] 与 level_starts [L] 烘焙为常量。
    """
    nodes: list[onnx.NodeProto] = []

    def add(op: str, inputs: list[str], **kw) -> str:
        out = f"{scope}/gs{len(nodes)}_{op}"
        nodes.append(helper.make_node(op, inputs, [out], name=out, **kw))
        return out

    def i64(a):
        return pool.get(np.array(a, dtype=np.int64))

    def f32(a):
        return pool.get(np.array(a, dtype=np.float32))

    shp_locs = add("Shape", [sampling_locations])
    shp_val = add("Shape", [value])
    heads = add("Gather", [shp_val, i64([2])])             # heads = value 第 2 维
    dim_d = add("Gather", [shp_val, i64([3])])             # head_dim = value 第 3 维
    dim_q = add("Gather", [shp_locs, i64([1])])
    dim_p = add("Gather", [shp_locs, i64([4])])

    acc: str | None = None
    for lvl, (hh, ww) in enumerate(spatial_shapes):
        hw = int(hh) * int(ww)
        start = int(level_starts[lvl])

        # value → [bs*heads, head_dim, H, W]
        v_l = add("Slice", [value, i64([start]), i64([start + hw]), i64([1])])
        t1 = add("Concat", [i64([-1]), i64([hh]), i64([ww]), heads, dim_d], axis=0)
        v1 = add("Reshape", [v_l, t1])                     # [bs, H, W, heads, d]
        v2 = add("Transpose", [v1], perm=[0, 3, 4, 1, 2])  # [bs, heads, d, H, W]
        t2 = add("Concat", [i64([-1]), dim_d, i64([hh]), i64([ww])], axis=0)
        v3 = add("Reshape", [v2, t2])                      # [bs*heads, d, H, W]

        # locations → [bs*heads, Q, P, 2] 归一化到 [-1,1]
        l_l = add("Slice", [sampling_locations, i64([lvl]), i64([lvl + 1]), i64([3])])
        l1 = add("Squeeze", [l_l, i64([3])])               # [bs, Q, heads, P, 2]
        l2 = add("Transpose", [l1], perm=[0, 2, 1, 3, 4])  # [bs, heads, Q, P, 2]
        t3 = add("Concat", [i64([-1]), dim_q, dim_p, i64([2])], axis=0)
        l3 = add("Reshape", [l2, t3])                      # [bs*heads, Q, P, 2]
        g1 = add("Mul", [l3, f32(2.0)])
        grid = add("Sub", [g1, f32(1.0)])

        samp = add(
            "GridSample", [v3, grid],
            mode="bilinear", padding_mode="zeros", align_corners=0,
        )                                                  # [bs*heads, d, Q, P]

        # weights → [bs*heads, 1, Q, P]
        w_l = add("Slice", [attention_weights, i64([lvl]), i64([lvl + 1]), i64([3])])
        w1 = add("Squeeze", [w_l, i64([3])])               # [bs, Q, heads, P]
        w2 = add("Transpose", [w1], perm=[0, 2, 1, 3])     # [bs, heads, Q, P]
        t4 = add("Concat", [i64([-1]), i64([1]), dim_q, dim_p], axis=0)
        w3 = add("Reshape", [w2, t4])                      # [bs*heads, 1, Q, P]

        prod = add("Mul", [samp, w3])
        contrib = add("ReduceSum", [prod, i64([3])], keepdims=0)  # [bs*heads, d, Q]
        acc = contrib if acc is None else add("Add", [acc, contrib])

    assert acc is not None
    o1 = add("Transpose", [acc], perm=[0, 2, 1])           # [bs*heads, Q, d]
    t5 = add("Concat", [i64([-1]), heads, dim_q, dim_d], axis=0)
    o2 = add("Reshape", [o1, t5])                          # [bs, heads, Q, d]
    o3 = add("Transpose", [o2], perm=[0, 2, 1, 3])         # [bs, Q, heads, d]
    dim_b = add("Gather", [shp_val, i64([0])])
    t6 = add("Concat", [dim_b, dim_q, i64([-1])], axis=0)
    nodes.append(helper.make_node("Reshape", [o3, t6], [output], name=f"{scope}/gs_out"))
    return nodes


# ---------------------------------------------------------------- 手术


def sanitize_scope(node: onnx.NodeProto) -> str:
    """从 MDA 节点输入路径取倒数 3 段做 scope (encoder/layers.0/self_attn), 保证 12 个 MDA 唯一。"""
    segs = [s.replace(".", "_") for s in node.input[0].split("/")]
    scope = "_".join(segs[-4:-1]) if len(segs) >= 4 else "_".join(segs[:-1]) or "mda"
    return f"gs/{scope}"


@app.command()
def surgery(in_path: str, out_path: str) -> None:
    """替换全部 MDA plugin 节点并保存术后模型。"""
    model = onnx.load(in_path)
    assert not any(i.data_location == TensorProto.EXTERNAL for i in model.graph.initializer), "外部数据不支持"
    producers = {out: n for n in model.graph.node for out in n.output}
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    pool = ConstPool()
    new_nodes: list[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type == MDA_OP and node.domain == MDA_DOMAIN:
            shapes = trace_const(producers, initializers, node.input[1])
            starts = trace_const(producers, initializers, node.input[2])
            if shapes is None or starts is None:
                logger.error(f"无法解析常量: {node.input[1]} / {node.input[2]} — fail-fast 不猜")
                raise typer.Exit(1)
            new_nodes.extend(
                build_mda_subgraph(
                    sanitize_scope(node), node.input[0], node.input[3], node.input[4],
                    node.output[0], shapes, starts, pool,
                )
            )
            replaced += 1
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(pool.initializers())
    onnx.checker.check_model(model)
    logger.info(f"替换 {replaced} 个 MDA 节点, 新增子图共 {len(model.graph.node)} 节点, 常量 {len(pool.initializers())} 个")
    onnx.save(model, out_path)
    logger.info(f"已保存: {out_path}")


# ---------------------------------------------------------------- 单元验证


def msda_reference(value: np.ndarray, locs: np.ndarray, weights: np.ndarray,
                   shapes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """手写双线性 MDA 参考 (独立于 GridSample): pixel = loc*W-0.5, 越界补零。value 4-D。"""
    bs, _lv, h, d = value.shape
    q, _lvls, p = locs.shape[1], locs.shape[3], locs.shape[4]
    v = value
    out = np.zeros((bs, q, h, d), dtype=np.float32)
    for b in range(bs):
        for qi in range(q):
            for hi in range(h):
                acc = np.zeros(d, dtype=np.float32)
                for li, (hh, ww) in enumerate(shapes):
                    s = starts[li]
                    for pi in range(p):
                        wgt = weights[b, qi, hi, li, pi]
                        xp = locs[b, qi, hi, li, pi, 0] * ww - 0.5
                        yp = locs[b, qi, hi, li, pi, 1] * hh - 0.5
                        x0, y0 = int(np.floor(xp)), int(np.floor(yp))
                        fx, fy = xp - x0, yp - y0
                        for dy in (0, 1):
                            for dx in (0, 1):
                                xx, yy = x0 + dx, y0 + dy
                                if 0 <= xx < ww and 0 <= yy < hh:
                                    wq = (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                                    acc += wgt * wq * v[b, s + yy * ww + xx, hi]
                out[b, qi, hi] = acc
    return out.reshape(bs, q, h * d)


@app.command()
def verify_unit(seed: int = 7) -> None:
    """合成小尺寸输入: 子图 (onnxruntime) vs 手写参考 数值对拍。"""
    rng = np.random.default_rng(seed)
    shapes = np.array([(5, 7), (3, 4)])
    starts = np.array([0, 35])
    bs, q, h, p, d = 2, 6, 2, 3, 4
    value = rng.standard_normal((bs, 47, h, d)).astype(np.float32)
    # 部分位置故意越界 [-0.2, 1.2] 检验补零语义
    locs = rng.uniform(-0.2, 1.2, (bs, q, h, 2, p, 2)).astype(np.float32)
    weights = rng.uniform(0, 1, (bs, q, h, 2, p)).astype(np.float32)

    pool = ConstPool()
    sub = build_mda_subgraph("test", "value", "locs", "weights", "out", shapes, starts, pool)
    graph = helper.make_graph(
        sub, "msda_unit",
        inputs=[
            helper.make_tensor_value_info("value", TensorProto.FLOAT, ["bs", "lv", "h", "d"]),
            helper.make_tensor_value_info("locs", TensorProto.FLOAT, ["bs", "q", "h", "l", "p", 2]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, ["bs", "q", "h", "l", "p"]),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, ["bs", "q", "e"])],
        initializer=pool.initializers(),
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)

    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    got = sess.run(["out"], {"value": value, "locs": locs, "weights": weights})[0]
    ref = msda_reference(value, locs, weights, shapes, starts)
    diff = np.abs(got - ref).max()
    logger.info(f"子图 vs 手写参考: shape={got.shape}, max|Δ|={diff:.2e}")
    assert diff < 2e-5, f"数值不一致: {diff}"
    logger.success("verify-unit 通过")


# ---------------------------------------------------------------- 端到端验证


def preprocess_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Rust peoplenet::preprocess Transformer 分支的 Python 镜像 (RGB/960×544/ImageNet)。"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) - MEAN_RGB) * SCALE
    return arr.transpose(2, 0, 1)[None]


@app.command()
def verify_e2e(model_path: str, image_path: str, out_jpg: str, conf: float = 0.5) -> None:
    """术后全模型推理: CUDA EP 优先, topk 解码 + 画框。"""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(model_path, providers=providers)
    logger.info(f"EP 顺序: {sess.get_providers()}")
    names = [i.name for i in sess.get_inputs()]
    assert names == ["inputs"], f"输入名不符: {names}"

    img = cv2.imread(image_path)
    assert img is not None, f"读取失败: {image_path}"
    logits, boxes = sess.run(["pred_logits", "pred_boxes"], {"inputs": preprocess_rgb(img)})
    scores = 1.0 / (1.0 + np.exp(-logits[0]))              # sigmoid
    best_cls = scores.argmax(axis=1)
    best_score = scores[np.arange(len(best_cls)), best_cls]
    order = np.argsort(-best_score)
    h, w = img.shape[:2]
    drawn = 0
    for idx in order[:20]:
        if best_score[idx] < conf:
            break
        cx, cy, bw, bh = boxes[0, idx] * np.array([w, h, w, h])
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        cv2.rectangle(img, (x0, y0), (x0 + int(bw), y0 + int(bh)), (0, 255, 0), 2)
        cv2.putText(img, f"c{best_cls[idx]} {best_score[idx]:.2f}", (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        logger.info(f"  检出: cls={best_cls[idx]} score={best_score[idx]:.3f} box=({x0},{y0},{int(bw)}x{int(bh)})")
        drawn += 1
    cv2.imwrite(out_jpg, img)
    logger.info(f"logits {logits.shape} / boxes {boxes.shape}; 画框 {drawn} 个 → {out_jpg}")


if __name__ == "__main__":
    app()
