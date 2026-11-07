"""Packed-expert shapes for tools/requant_experts.py and tools/pack_experts.py.

An expert blob is [gate | up | down], each block [weights | bf16 scales |
bf16 biases], affine groups of `group` along each row. Gate / up are
[inter, hidden], down is [hidden, inter]; all three blocks have the same
size. Weights are 8 nibbles per u32 (4-bit) or 8 codes per 3 bytes
(packed 3-bit). Expert e sits at e * stride in its layer file; stride
is the blob size unless packed_experts/layout.json declares padding
(the Qwen3.8-Flash-Next converter pads to 16 KiB).

Shapes come, in order of precedence, from --hidden / --inter / --group,
from <root>/packed_experts/layout.json (its gate_proj.weight /
gate_proj.scales components and group_size), else Qwen3.6-35B-A3B's
(hidden 2048, moe_intermediate 512, group 64; Qwen3-Next-80B-A3B has
the same).
"""
import json
import os

A3B = (2048, 512, 64)


def add_args(ap):
    ap.add_argument("--hidden", type=int, help="hidden size (default: layout.json, else 2048)")
    ap.add_argument("--inter", type=int, help="moe_intermediate size (default: layout.json, else 512)")
    ap.add_argument("--group", type=int, help="quantization group size (default: layout.json, else 64)")


def read_layout(root):
    p = os.path.join(root, "packed_experts", "layout.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


class Shape:
    def __init__(self, hidden, inter, group, stride=None):
        assert hidden % group == 0 and inter % group == 0, "hidden and inter must be multiples of group"
        assert (inter * hidden) % 8 == 0
        self.hidden, self.inter, self.group = hidden, inter, group
        self.w4 = inter * hidden // 2              # one matrix's 4-bit weights
        self.w3 = inter * hidden * 3 // 8          # the same, packed 3-bit
        self.sb = inter * (hidden // group) * 2    # its bf16 scales (biases the same)
        self.block4, self.block3 = self.w4 + 2 * self.sb, self.w3 + 2 * self.sb
        self.expert4, self.expert3 = 3 * self.block4, 3 * self.block3
        # Bytes between experts in the source files (nibble layout).
        self.stride = stride or self.expert4
        assert self.stride >= self.expert4

    def __str__(self):
        pad = f", stride {self.stride}" if self.stride != self.expert4 else ""
        return (f"hidden {self.hidden}, inter {self.inter}, group {self.group}: "
                f"4-bit expert {self.expert4} B, packed 3-bit {self.expert3} B{pad}")


def shape_for(root, a):
    """The shape of the 4-bit-layout experts under <root> (args first)."""
    lay = read_layout(root)
    hidden, inter, group = A3B
    stride = None
    if lay is not None:
        comps = {c["name"]: c for c in lay.get("components", [])}
        if "gate_proj.weight" in comps:
            inter = comps["gate_proj.weight"]["shape"][0]
            hidden = comps["gate_proj.weight"]["size"] * 2 // inter
        group = lay.get("group_size", group)
        if lay.get("bits", 4) != 4:
            raise SystemExit(f"{root}: layout.json says {lay['bits']}-bit; the source must be in the 4-bit layout")
        stride = lay.get("expert_stride")
    hidden = a.hidden or hidden
    inter = a.inter or inter
    group = a.group or group
    s = Shape(hidden, inter, group, stride)
    if lay is not None and lay.get("expert_size", s.expert4) != s.expert4:
        raise SystemExit(f"{root}: layout.json expert_size {lay['expert_size']} != {s.expert4} for {s}")
    return s, lay


def round_up(n, align):
    return (n + align - 1) // align * align if align else n


def write_layout(root, lay, expert_size, expert_stride, bits, shape):
    """packed_experts/layout.json for the output: the source's, with the
    new size / stride / bits and component offsets."""
    out = dict(lay or {})
    out.update({"expert_size": expert_size, "expert_stride": expert_stride, "bits": bits,
                "group_size": shape.group})
    w = shape.w4 if bits == 4 else shape.w3
    comps, off = [], 0
    for proj, (rows, cols) in (("gate_proj", (shape.inter, shape.hidden)), ("up_proj", (shape.inter, shape.hidden)),
                               ("down_proj", (shape.hidden, shape.inter))):
        wshape = [rows, cols // 8] if bits == 4 else [rows, cols * 3 // 8]
        comps.append({"name": f"{proj}.weight", "offset": off, "size": w,
                      "dtype": "U32" if bits == 4 else "U8", "shape": wshape})
        off += w
        for part in ("scales", "biases"):
            comps.append({"name": f"{proj}.{part}", "offset": off, "size": shape.sb, "dtype": "BF16",
                          "shape": [rows, cols // shape.group]})
            off += shape.sb
    assert off == expert_size
    out["components"] = comps
    with open(os.path.join(root, "packed_experts", "layout.json"), "w") as f:
        json.dump(out, f, indent=1)
