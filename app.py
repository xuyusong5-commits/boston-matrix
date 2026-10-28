"""自由坐标的四象限矩阵。

横轴、纵轴由使用者命名。十字线是有效行的平均值，品项落到四个格子里。
"""

from __future__ import annotations

import hashlib
import html
import io
import math
from dataclasses import dataclass
from string import Template

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PALETTE = {
    "paper": "#F5F2EB",
    "sidebar": "#EFEAE2",
    "card": "#FFFCF8",
    "ink": "#1C2822",
    "ink_soft": "#3E4943",
    "muted": "#6F685F",
    "line": "#E3DACE",
    "cross": "#1E2A24",
    "primary": "#24382E",
    "green": "#5F8F70",
    "blue": "#6D90B0",
    "gold": "#C1964E",
    "rose": "#B07B86",
    "header": "#F3EEE6",
    "banner": "#F8F2E2",
    "banner_line": "#E7D7B2",
    "banner_ink": "#5C4E32",
    "grid": "#E8E1D6",
}

SANS = (
    '"PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Noto Sans SC", '
    '"Microsoft YaHei", "Source Han Sans SC", "WenQuanYi Micro Hei", sans-serif'
)
SERIF = (
    '"Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", "STSong", '
    '"PMingLiU", "SimSun", "PingFang SC", "Noto Sans CJK SC", serif'
)
PLOTLY_FONT = (
    "PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, Noto Sans SC, "
    "Microsoft YaHei, Source Han Sans SC, sans-serif"
)

SAMPLE_X = "点击率"
SAMPLE_Y = "毛利率"
SAMPLE_ROWS = [
    ("手工柠檬茶", 8.6, 66),
    ("招牌牛肉面", 7.4, 43),
    ("季节沙拉", 4.8, 58),
    ("可乐", 9.0, 22),
    ("新品燕麦碗", 2.8, 54),
    ("炸鸡小食", 6.8, 27),
    ("白米饭", 3.6, 11),
    ("桂花甜品", 2.2, 49),
    ("工作日套餐", 7.9, 44),
    ("儿童餐", 4.4, 19),
]

CHART_HEIGHT = 660
CHART_MARGIN = dict(l=108, r=48, t=32, b=56)
PLOT_W = 920
PLOT_H = CHART_HEIGHT - CHART_MARGIN["t"] - CHART_MARGIN["b"]
MAX_UPLOAD_ROWS = 300
GENERIC_HEADERS = {"x", "y", "value", "数值", "横轴", "纵轴", "x轴", "y轴"}
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
    "displaylogo": False,
}


@dataclass(frozen=True)
class QuadrantSpec:
    key: str
    high_x: bool
    high_y: bool
    action: str
    dot: str
    wash: str
    fill: str


QUADRANT_SPECS: dict[str, QuadrantSpec] = {
    "lh": QuadrantSpec("lh", False, True, "先看短板", PALETTE["blue"], "#E7EFF6", "rgba(109, 144, 176, 0.18)"),
    "hh": QuadrantSpec("hh", True, True, "优先加码", PALETTE["green"], "#E7F2EA", "rgba(95, 143, 112, 0.17)"),
    "ll": QuadrantSpec("ll", False, False, "考虑收缩", PALETTE["rose"], "#F8EEEF", "rgba(176, 123, 134, 0.16)"),
    "hl": QuadrantSpec("hl", True, False, "先守住", PALETTE["gold"], "#F8F2E2", "rgba(193, 150, 78, 0.20)"),
}
# 与散点图同一阅读方向：左低右高，下低上高。
QUADRANT_ORDER = ("lh", "hh", "ll", "hl")
CORNERS = {
    "lh": dict(x=0.03, y=0.975, xanchor="left", yanchor="top"),
    "hh": dict(x=0.97, y=0.975, xanchor="right", yanchor="top"),
    "ll": dict(x=0.03, y=0.025, xanchor="left", yanchor="bottom"),
    "hl": dict(x=0.97, y=0.025, xanchor="right", yanchor="bottom"),
}


@dataclass(frozen=True)
class Item:
    name: str
    x: float
    y: float
    key: str


@dataclass
class Matrix:
    items: list[Item]
    groups: dict[str, list[Item]]
    skipped: int
    x_mean: float | None
    y_mean: float | None
    x_name: str
    y_name: str
    x_limits: tuple[float, float] | None
    y_limits: tuple[float, float] | None
    flat_x: bool
    flat_y: bool


@dataclass
class ParsedTable:
    frame: pd.DataFrame
    x_label: str | None
    y_label: str | None
    warning: str | None = None


@dataclass
class PlacedLabel:
    name: str
    dx: float
    dy: float
    w: float
    h: float
    nudged: bool
    anchor_px: float
    anchor_py: float


@dataclass
class LabelLayout:
    labels: list[PlacedLabel]
    plot_w: float
    plot_h: float
    point_r: float
    gap: float


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame([{"name": name, "x": x, "y": y} for name, x, y in SAMPLE_ROWS])


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": pd.Series(dtype="object"),
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
        }
    )


def axis_name(value: str | None, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def cell_name(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and abs(number - round(number)) < 1e-9 and abs(number) < 1e15:
            return str(int(round(number)))
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def coerce_scalar(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value) is True:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = (
            str(value)
            .strip()
            .replace(",", "")
            .replace("，", "")
            .replace("%", "")
            .replace("％", "")
            .replace("¥", "")
            .replace("￥", "")
            .replace(" ", "")
        )
        if text.lower() in {"", "nan", "none", "null"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if not math.isfinite(number):
        return None
    return number


def fmt_num(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    if magnitude >= 1000:
        body = f"{magnitude:,.0f}" if abs(magnitude - round(magnitude)) < 1e-9 else f"{magnitude:,.1f}"
        return sign + body
    if abs(magnitude - round(magnitude)) < 1e-9:
        return sign + str(int(round(magnitude)))
    if magnitude >= 10:
        text = f"{magnitude:.1f}"
    elif magnitude >= 1:
        text = f"{magnitude:.2f}"
    else:
        text = f"{magnitude:.3f}"
    return sign + text.rstrip("0").rstrip(".")


def axis_kind(values: list[float]) -> str:
    """0–1 的小数按比例换算成百分数；落在 0–100 的，直接当成百分率。"""
    if not values:
        return "number"
    if all(0 <= value <= 1 for value in values) and any(0 < value < 1 for value in values):
        return "proportion"
    if all(0 <= value <= 100 for value in values):
        return "percent"
    return "number"


def format_axis_value(value: float, kind: str) -> str:
    if kind == "proportion":
        return fmt_num(value * 100) + "%"
    if kind == "percent":
        return fmt_num(value) + "%"
    return fmt_num(value)


def axis_phrase(name: str, value: float, kind: str) -> str:
    label = name.strip()
    if kind in {"percent", "proportion"}:
        for suffix in ("（%）", "(%)", "％", "%"):
            if label.endswith(suffix):
                label = label[: -len(suffix)].strip()
                break
    return f"{label}{format_axis_value(value, kind)}"


def point_caption_lines(item: Item, matrix: Matrix, x_kind: str, y_kind: str) -> tuple[str, str]:
    name = chart_name(item.name, 14)
    detail = f"{axis_phrase(matrix.x_name, item.x, x_kind)}，{axis_phrase(matrix.y_name, item.y, y_kind)}"
    return name, detail


def chart_name(name: str, limit: int = 10) -> str:
    text = str(name).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def short_axis(name: str, limit: int = 10) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 1] + "…"


def _work_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return empty_frame()
    columns = {}
    for column in ("name", "x", "y"):
        if column in df.columns:
            columns[column] = df[column]
        else:
            columns[column] = pd.Series([pd.NA] * len(df), index=df.index)
    return pd.DataFrame(columns).reset_index(drop=True)


def _read_csv(raw: bytes) -> pd.DataFrame:
    decoded = None
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        try:
            decoded = raw.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError("编码读不出来。请把文件另存为 UTF-8 CSV。") from exc
    try:
        return pd.read_csv(io.StringIO(decoded))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("文件是空的。") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("这个 CSV 读不出来，请检查逗号和引号。") from exc


def axis_from_header(column: object) -> str | None:
    text = str(column).strip()
    lowered = text.lower()
    if not text or lowered.startswith("unnamed") or lowered in GENERIC_HEADERS:
        return None
    for suffix in ("（横轴）", "（纵轴）"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    if not text or text.lower() in GENERIC_HEADERS:
        return None
    return text


def parse_csv_bytes(raw: bytes) -> ParsedTable:
    if raw is None or not raw.strip():
        raise ValueError("文件是空的。")
    frame = _read_csv(raw).dropna(how="all")
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.shape[1] < 3:
        raise ValueError("至少需要三列：名称，以及两个数字坐标。")
    if frame.empty:
        raise ValueError("表里还没有数据行。")

    numeric_index: list[int] = []
    for index in range(1, frame.shape[1]):
        coerced = frame.iloc[:, index].map(coerce_scalar)
        score = float(pd.Series(coerced).notna().mean()) if len(frame) else 0.0
        if score >= 0.5:
            numeric_index.append(index)
    if len(numeric_index) < 2:
        raise ValueError("没有找到两列数字。请确认横轴和纵轴是数字。")

    x_index, y_index = numeric_index[0], numeric_index[1]
    parsed = pd.DataFrame(
        {
            "name": frame.iloc[:, 0].map(cell_name),
            "x": frame.iloc[:, x_index].map(coerce_scalar),
            "y": frame.iloc[:, y_index].map(coerce_scalar),
        }
    )
    blank = (parsed["name"] == "") & parsed["x"].isna() & parsed["y"].isna()
    parsed = parsed.loc[~blank].reset_index(drop=True)
    if parsed.empty:
        raise ValueError("这些行是空的。")

    warning = None
    if len(parsed) > MAX_UPLOAD_ROWS:
        parsed = parsed.iloc[:MAX_UPLOAD_ROWS].reset_index(drop=True)
        warning = f"只读入了前 {MAX_UPLOAD_ROWS} 行，避免图被挤满。"
    return ParsedTable(
        frame=parsed,
        x_label=axis_from_header(frame.columns[x_index]),
        y_label=axis_from_header(frame.columns[y_index]),
        warning=warning,
    )


def table_to_csv_bytes(df: pd.DataFrame, x_name: str, y_name: str) -> bytes:
    work = _work_frame(df)
    x_header = axis_name(x_name, "横轴")
    y_header = axis_name(y_name, "纵轴")
    if x_header == "品项":
        x_header = "品项（横轴）"
    if y_header in {"品项", x_header}:
        y_header = f"{axis_name(y_name, '纵轴')}（纵轴）"
    exported = pd.DataFrame({"品项": work["name"], x_header: work["x"], y_header: work["y"]})
    return exported.to_csv(index=False).encode("utf-8-sig")


def classify(x: float, y: float, x_mean: float, y_mean: float) -> str:
    """高于均值算高。落在均值上的算偏低，避免压线的点同时进入两格。"""
    high_x = x > x_mean
    high_y = y > y_mean
    if high_x and high_y:
        return "hh"
    if high_y:
        return "lh"
    if high_x:
        return "hl"
    return "ll"


def padded_limits(values: list[float], ratio: float = 0.22) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        span = max(abs(lo), 1.0) * 0.5
        if span <= 1e-12:
            span = 1.0
    pad = span * ratio
    return lo - pad, hi + pad


def quadrant_title(spec: QuadrantSpec, x_name: str, y_name: str) -> str:
    x_side = "高" if spec.high_x else "低"
    y_side = "高" if spec.high_y else "低"
    return f"{x_side}{x_name} · {y_side}{y_name}"


def quadrant_hint(key: str, x_name: str, y_name: str) -> str:
    hints = {
        "hh": f"{x_name}和{y_name}都高于均值。陈列、投放、备货或产能，优先给这里。",
        "lh": f"{y_name}高于均值，{x_name}还没跟上。先看短板在哪，再决定要不要加资源。",
        "hl": f"{x_name}高于均值，{y_name}偏低。一边已经立住了，先别放掉，再把弱的那边补上。",
        "ll": f"{x_name}和{y_name}都低于均值。少占货架、工时和产线，除非它另有用途。",
    }
    return hints[key]


def _strength(item: Item, x_mean: float, y_mean: float, x_span: float, y_span: float) -> float:
    sx = (item.x - x_mean) / x_span
    sy = (item.y - y_mean) / y_span
    if item.key == "hh":
        return sx + sy
    if item.key == "lh":
        return -sx + sy
    if item.key == "hl":
        return sx - sy
    return -sx - sy


def build_matrix(df: pd.DataFrame | None, x_name: str, y_name: str) -> Matrix:
    x_name = axis_name(x_name, "横轴")
    y_name = axis_name(y_name, "纵轴")
    groups: dict[str, list[Item]] = {key: [] for key in QUADRANT_ORDER}
    work = _work_frame(df)
    items: list[Item] = []
    skipped = 0
    for raw_name, raw_x, raw_y in zip(work["name"], work["x"], work["y"]):
        name = cell_name(raw_name)
        x_value = coerce_scalar(raw_x)
        y_value = coerce_scalar(raw_y)
        if name == "" and x_value is None and y_value is None:
            continue
        if name == "" or x_value is None or y_value is None:
            skipped += 1
            continue
        items.append(Item(name, x_value, y_value, ""))
    if not items:
        return Matrix(items, groups, skipped, None, None, x_name, y_name, None, None, False, False)

    xs = [item.x for item in items]
    ys = [item.y for item in items]
    x_mean = float(sum(xs) / len(xs))
    y_mean = float(sum(ys) / len(ys))
    keyed = [Item(item.name, item.x, item.y, classify(item.x, item.y, x_mean, y_mean)) for item in items]
    x_span = max(max(xs) - min(xs), 1e-9)
    y_span = max(max(ys) - min(ys), 1e-9)
    for item in keyed:
        groups[item.key].append(item)
    for key, group in groups.items():
        groups[key] = sorted(group, key=lambda item: (-_strength(item, x_mean, y_mean, x_span, y_span), item.name))
    return Matrix(
        items=keyed,
        groups=groups,
        skipped=skipped,
        x_mean=x_mean,
        y_mean=y_mean,
        x_name=x_name,
        y_name=y_name,
        x_limits=padded_limits(xs),
        y_limits=padded_limits(ys),
        flat_x=max(xs) - min(xs) <= 1e-9,
        flat_y=max(ys) - min(ys) <= 1e-9,
    )


def layout_point_labels(matrix: Matrix) -> LabelLayout:
    """把「品名，横轴值，纵轴值」放在圆点旁。偏移用像素，避免被轴距拉远。"""
    if matrix.x_limits is None or matrix.y_limits is None or matrix.x_mean is None or matrix.y_mean is None:
        return LabelLayout([], float(PLOT_W), float(PLOT_H), 14.0, 6.0)
    xmin, xmax = matrix.x_limits
    ymin, ymax = matrix.y_limits
    x_span = max(xmax - xmin, 1e-9)
    y_span = max(ymax - ymin, 1e-9)
    plot_w = float(PLOT_W)
    plot_h = float(PLOT_H)
    font_px = 12
    point_r = 14.0
    gap = 6.0
    x_kind = axis_kind([item.x for item in matrix.items])
    y_kind = axis_kind([item.y for item in matrix.items])
    specs: list[tuple[str, float, float, float, float]] = []
    for item in matrix.items:
        name_line, detail_line = point_caption_lines(item, matrix, x_kind, y_kind)
        text = f"{name_line}\n{detail_line}"
        width = max(len(name_line), len(detail_line), 1) * font_px + 14
        height = font_px * 2 + 14
        anchor_x = (item.x - xmin) / x_span * plot_w
        anchor_y = (item.y - ymin) / y_span * plot_h
        specs.append((text, anchor_x, anchor_y, width, height))
    mean_x = (matrix.x_mean - xmin) / x_span * plot_w
    mean_y = (matrix.y_mean - ymin) / y_span * plot_h

    def nearest(index: int) -> float:
        best = 1e9
        anchor_x, anchor_y = specs[index][1], specs[index][2]
        for other, spec in enumerate(specs):
            if other == index:
                continue
            dx = anchor_x - spec[1]
            dy = anchor_y - spec[2]
            best = min(best, dx * dx + dy * dy)
        return best

    def clamp(center_x: float, center_y: float, width: float, height: float) -> tuple[float, float]:
        half_w = min(width / 2, plot_w / 2 - 1)
        half_h = min(height / 2, plot_h / 2 - 1)
        return (
            min(max(center_x, half_w), plot_w - half_w),
            min(max(center_y, half_h), plot_h - half_h),
        )

    placed: list[PlacedLabel] = []

    def score(center_x: float, center_y: float, width: float, height: float) -> int:
        total = 0
        for other in placed:
            other_x = other.anchor_px + other.dx
            other_y = other.anchor_py + other.dy
            if abs(center_x - other_x) < (width + other.w) / 2 + gap and abs(center_y - other_y) < (
                height + other.h
            ) / 2 + gap:
                total += 5
        for spec in specs:
            if abs(center_x - spec[1]) < width / 2 + point_r and abs(center_y - spec[2]) < height / 2 + point_r:
                total += 6
        if (center_x - width / 2) < mean_x < (center_x + width / 2) or (center_y - height / 2) < mean_y < (
            center_y + height / 2
        ):
            total += 1
        return total

    order = sorted(range(len(specs)), key=lambda index: (-nearest(index), index))
    chosen: dict[int, PlacedLabel] = {}
    for index in order:
        text, anchor_x, anchor_y, width, height = specs[index]
        rise = point_r + height / 2 + 6
        side = point_r + width / 2 + 8
        inward = -1 if anchor_x >= plot_w / 2 else 1
        toward = rise if anchor_y >= mean_y else -rise
        away = -toward
        candidates = [
            (0.0, toward),
            (inward * side * 0.92, toward * 0.72),
            (-inward * side * 0.92, toward * 0.72),
            (inward * side, 0.0),
            (0.0, toward * 1.7),
            (inward * side * 1.2, toward),
            (-inward * side, toward * 0.35),
            (inward * side * 0.75, away * 0.55),
            (0.0, away),
            (-inward * side, away * 0.75),
            (inward * side * 1.45, toward * 0.2),
            (-inward * side * 1.45, 0.0),
        ]
        best: tuple[float, float, int, int] | None = None
        best_rank: tuple[int, int] | None = None
        for preference, (shift_x, shift_y) in enumerate(candidates):
            center_x, center_y = clamp(anchor_x + shift_x, anchor_y + shift_y, width, height)
            offset_x = center_x - anchor_x
            offset_y = center_y - anchor_y
            penalty = score(center_x, center_y, width, height)
            rank = (penalty, preference)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = (offset_x, offset_y, penalty, preference)
            if penalty == 0:
                break
        assert best is not None
        offset_x, offset_y, _penalty, preference = best
        nudged = preference != 0 and not (abs(offset_x) < 8 and abs(offset_y - toward) < 8)
        label = PlacedLabel(text, offset_x, offset_y, width, height, nudged, anchor_x, anchor_y)
        placed.append(label)
        chosen[index] = label
    _separate_labels(placed, plot_w, plot_h, gap, point_r)
    return LabelLayout([chosen[index] for index in range(len(specs))], plot_w, plot_h, point_r, gap)


def _point_clearance(label: PlacedLabel, dx: float, dy: float, point_r: float) -> float:
    """大于 0 表示说明没有盖住自己的圆点。"""
    clear_x = abs(dx) - (label.w / 2 + point_r)
    clear_y = abs(dy) - (label.h / 2 + point_r)
    return max(clear_x, clear_y)


def _nudge_offset(label: PlacedLabel, axis: str, sign: float, amount: float, point_r: float) -> bool:
    """往 sign 方向推。如果会把说明压回自己的圆点，就不动。"""
    updated_x = label.dx + (sign * amount if axis == "x" else 0)
    updated_y = label.dy + (sign * amount if axis == "y" else 0)
    before = _point_clearance(label, label.dx, label.dy, point_r)
    after = _point_clearance(label, updated_x, updated_y, point_r)
    if after < before - 0.1 and after < 4:
        return False
    label.dx = updated_x
    label.dy = updated_y
    return True


def _separate_labels(labels: list[PlacedLabel], plot_w: float, plot_h: float, gap: float, point_r: float) -> None:
    """把仍然叠在一起的说明轻轻推开，并保持在图内、避开圆点。"""
    if len(labels) < 2:
        return

    def clamp_label(label: PlacedLabel) -> None:
        center_x = min(max(label.anchor_px + label.dx, label.w / 2), plot_w - label.w / 2)
        center_y = min(max(label.anchor_py + label.dy, label.h / 2), plot_h - label.h / 2)
        label.dx = center_x - label.anchor_px
        label.dy = center_y - label.anchor_py

    for _ in range(48):
        moved = False
        for index, label in enumerate(labels):
            center_x = label.anchor_px + label.dx
            center_y = label.anchor_py + label.dy
            for other in labels[index + 1 :]:
                other_x = other.anchor_px + other.dx
                other_y = other.anchor_py + other.dy
                overlap_x = (label.w + other.w) / 2 + gap - abs(center_x - other_x)
                overlap_y = (label.h + other.h) / 2 + gap - abs(center_y - other_y)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                horizontal = overlap_x <= overlap_y
                if horizontal:
                    sign = 1 if other_x >= center_x else -1
                    amount = overlap_x + 2
                    axis = "x"
                else:
                    sign = 1 if other_y >= center_y else -1
                    amount = overlap_y + 2
                    axis = "y"
                other_moved = _nudge_offset(other, axis, sign, amount, point_r)
                label_moved = _nudge_offset(label, axis, -sign, amount, point_r)
                if not other_moved and not label_moved:
                    other_axis = "y" if axis == "x" else "x"
                    other_sign = 1 if (other.dy if other_axis == "y" else other.dx) >= 0 else -1
                    label_sign = 1 if (label.dy if other_axis == "y" else label.dx) >= 0 else -1
                    if other_sign == label_sign:
                        other_sign *= -1
                    other_moved = _nudge_offset(other, other_axis, other_sign, 18, point_r)
                    label_moved = _nudge_offset(label, other_axis, label_sign, 18, point_r)
                moved = moved or other_moved or label_moved
        for label in labels:
            clamp_label(label)
            if abs(label.dx) > 10 or abs(label.dy) > 10:
                label.nudged = True
        if not moved:
            break


def label_issues(layout: LabelLayout) -> list[str]:
    issues: list[str] = []
    labels = layout.labels
    for index, label in enumerate(labels):
        center_x = label.anchor_px + label.dx
        center_y = label.anchor_py + label.dy
        if center_x - label.w / 2 < -0.5 or center_x + label.w / 2 > layout.plot_w + 0.5:
            issues.append(f"{label.name} 超出左右边界")
        if center_y - label.h / 2 < -0.5 or center_y + label.h / 2 > layout.plot_h + 0.5:
            issues.append(f"{label.name} 超出上下边界")
        for other in labels[index + 1 :]:
            other_x = other.anchor_px + other.dx
            other_y = other.anchor_py + other.dy
            separated_x = abs(center_x - other_x) >= (label.w + other.w) / 2 + layout.gap - 0.1
            separated_y = abs(center_y - other_y) >= (label.h + other.h) / 2 + layout.gap - 0.1
            if not separated_x and not separated_y:
                issues.append(f"{label.name} 与 {other.name} 重叠")
        for other in labels:
            hits_x = abs(center_x - other.anchor_px) < label.w / 2 + layout.point_r - 0.1
            hits_y = abs(center_y - other.anchor_py) < label.h / 2 + layout.point_r - 0.1
            if hits_x and hits_y:
                issues.append(f"{label.name} 盖住了 {other.name} 的圆点")
    return issues


def _corner_text(spec: QuadrantSpec, x_name: str, y_name: str) -> str:
    # 分成两行，窄屏上左右角的说明才不会撞在一起。
    x_part = f"{'高' if spec.high_x else '低'}{short_axis(x_name, 8)}"
    y_part = f"{'高' if spec.high_y else '低'}{short_axis(y_name, 8)}"
    return f"{html.escape(x_part)}<br>{html.escape(y_part)}"


def _axis_kwargs(title: str | None, low: float, high: float) -> dict:
    axis = dict(
        range=[low, high],
        showgrid=True,
        gridcolor=PALETTE["grid"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(size=12, color=PALETTE["muted"], family=PLOTLY_FONT),
        fixedrange=True,
        nticks=6,
        automargin=True,
    )
    if title:
        axis["title"] = dict(
            text=html.escape(title),
            font=dict(size=13, color=PALETTE["ink_soft"], family=PLOTLY_FONT),
            standoff=10,
        )
    else:
        # 纵轴标题改成横排注释，避免中文被旋转 90 度。
        axis["title"] = dict(text="")
    return axis


def _horizontal_y_title(name: str) -> tuple[dict, int]:
    chunks = [name[i : i + 6] for i in range(0, max(len(name), 1), 6)]
    gutter = 52
    margin_left = gutter + max(len(chunk) for chunk in chunks) * 14 + 22
    annotation = dict(
        text="<br>".join(html.escape(chunk) for chunk in chunks),
        xref="paper",
        yref="paper",
        x=0,
        y=0.5,
        xanchor="right",
        yanchor="middle",
        xshift=-gutter,
        showarrow=False,
        align="right",
        font=dict(family=PLOTLY_FONT, size=13, color=PALETTE["ink_soft"]),
    )
    return annotation, margin_left


def _base_layout(**extra: object) -> dict:
    layout = dict(
        template="simple_white",
        height=CHART_HEIGHT,
        margin=CHART_MARGIN,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOTLY_FONT, color=PALETTE["ink_soft"]),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor=PALETTE["card"],
            bordercolor=PALETTE["line"],
            font=dict(family=PLOTLY_FONT, size=13, color=PALETTE["ink"]),
        ),
        dragmode=False,
        showlegend=False,
    )
    layout.update(extra)
    return layout


def build_figure(matrix: Matrix) -> go.Figure:
    if not matrix.items or matrix.x_limits is None or matrix.y_limits is None or matrix.x_mean is None:
        figure = go.Figure()
        figure.update_layout(
            **_base_layout(
                xaxis=dict(visible=False, range=[0, 1], fixedrange=True),
                yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
                annotations=[
                    dict(
                        text="先添几条品项，或在左侧载入示例<br>圆点和十字线会出现在这里",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.52,
                        showarrow=False,
                        align="center",
                        font=dict(family=PLOTLY_FONT, size=15, color=PALETTE["muted"]),
                    )
                ],
            )
        )
        return figure

    xmin, xmax = matrix.x_limits
    ymin, ymax = matrix.y_limits
    x_mean = float(matrix.x_mean)
    y_mean = float(matrix.y_mean)
    shapes = []
    for key in QUADRANT_ORDER:
        spec = QUADRANT_SPECS[key]
        shapes.append(
            dict(
                type="rect",
                x0=x_mean if spec.high_x else xmin,
                x1=xmax if spec.high_x else x_mean,
                y0=y_mean if spec.high_y else ymin,
                y1=ymax if spec.high_y else y_mean,
                fillcolor=spec.fill,
                line=dict(width=0),
                layer="below",
                opacity=1,
            )
        )
    for shape in (
        dict(type="line", x0=x_mean, x1=x_mean, y0=ymin, y1=ymax),
        dict(type="line", x0=xmin, x1=xmax, y0=y_mean, y1=y_mean),
    ):
        shape["line"] = dict(color=PALETTE["cross"], width=3, dash="solid")
        # simple_white 的形状默认只有 0.3 透明度，十字线会发虚。
        shape["opacity"] = 1
        shape["layer"] = "above"
        shapes.append(shape)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[x_mean],
            y=[y_mean],
            mode="markers",
            marker=dict(size=8, color=PALETTE["card"], line=dict(color=PALETTE["cross"], width=1.6)),
            hovertemplate=(
                "平均值<br>"
                + html.escape(axis_phrase(matrix.x_name, x_mean, axis_kind([item.x for item in matrix.items])))
                + "<br>"
                + html.escape(axis_phrase(matrix.y_name, y_mean, axis_kind([item.y for item in matrix.items])))
                + "<extra></extra>"
            ),
            showlegend=False,
            name="mean",
        )
    )
    x_kind = axis_kind([item.x for item in matrix.items])
    y_kind = axis_kind([item.y for item in matrix.items])
    customdata = [
        [
            item.name,
            axis_phrase(matrix.x_name, item.x, x_kind),
            axis_phrase(matrix.y_name, item.y, y_kind),
            quadrant_title(QUADRANT_SPECS[item.key], matrix.x_name, matrix.y_name),
        ]
        for item in matrix.items
    ]
    figure.add_trace(
        go.Scatter(
            x=[item.x for item in matrix.items],
            y=[item.y for item in matrix.items],
            mode="markers",
            marker=dict(
                size=17 if len(matrix.items) <= 18 else 13,
                color=[QUADRANT_SPECS[item.key].dot for item in matrix.items],
                line=dict(width=2.4, color=PALETTE["card"]),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "%{customdata[1]}<br>%{customdata[2]}<br>"
                + "<span style='color:#6F685F'>%{customdata[3]}</span><extra></extra>"
            ),
            showlegend=False,
            name="points",
            cliponaxis=False,
        )
    )

    annotations: list[dict] = []
    for key, anchor in CORNERS.items():
        spec = QUADRANT_SPECS[key]
        annotations.append(
            dict(
                text=_corner_text(spec, matrix.x_name, matrix.y_name),
                xref="paper",
                yref="paper",
                x=anchor["x"],
                y=anchor["y"],
                xanchor=anchor["xanchor"],
                yanchor=anchor["yanchor"],
                showarrow=False,
                align="left" if anchor["xanchor"] == "left" else "right",
                font=dict(family=PLOTLY_FONT, size=11, color="rgba(36, 52, 44, 0.45)"),
            )
        )
    y_title, margin_left = _horizontal_y_title(matrix.y_name)
    annotations.append(y_title)
    placed = layout_point_labels(matrix)
    for item, label in zip(matrix.items, placed.labels):
        annotations.append(
            dict(
                x=item.x,
                y=item.y,
                ax=label.dx,
                ay=-label.dy,
                xref="x",
                yref="y",
                axref="pixel",
                ayref="pixel",
                xanchor="center",
                yanchor="middle",
                text="<br>".join(html.escape(line) for line in label.name.split("\n")),
                showarrow=True,
                arrowhead=0,
                arrowwidth=1.1,
                arrowcolor="rgba(30, 42, 36, 0.45)",
                standoff=14,
                startstandoff=3,
                font=dict(family=PLOTLY_FONT, size=12, color=PALETTE["ink"]),
                align="center",
                bgcolor="rgba(255,252,248,0.94)",
                borderpad=4,
            )
        )
    figure.update_layout(
        **_base_layout(
            shapes=shapes,
            annotations=annotations,
            margin=dict(CHART_MARGIN, l=max(CHART_MARGIN["l"], margin_left)),
            xaxis=_axis_kwargs(matrix.x_name, xmin, xmax),
            yaxis=_axis_kwargs(None, ymin, ymax),
        )
    )
    return figure


CSS = Template(
    """
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {
  font-family: $sans !important;
}
.stApp { background: $paper; color: $ink; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; height: 0; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1080px;
  padding-top: 1.35rem;
  padding-bottom: 2.6rem;
}
[data-testid="stVerticalBlock"] { gap: 0.65rem; }
section[data-testid="stSidebar"] {
  background: $sidebar;
  border-right: 1px solid $line;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.4rem; }
section[data-testid="stSidebar"] input {
  background: $card !important;
  border: 1px solid $line !important;
  border-radius: 12px !important;
  color: $ink !important;
}
section[data-testid="stSidebar"] input:focus {
  border-color: $green !important;
  box-shadow: 0 0 0 3px rgba(95, 143, 112, 0.18) !important;
}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: $primary !important;
  color: $paper !important;
  border: 1px solid $primary !important;
  border-radius: 999px !important;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  color: $ink !important;
  border: 1px solid $line !important;
  border-radius: 999px !important;
}
[data-testid="stFileUploader"] section {
  background: $card;
  border: 1px dashed $line;
  border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
  font-size: 0 !important;
  line-height: 0 !important;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p::after {
  content: "选择文件";
  font-size: 0.88rem;
  line-height: 1.2;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
  font-size: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span::after {
  content: "CSV，最大 20MB";
  font-size: 0.75rem;
  color: $muted;
}
[data-testid="stDataEditor"] {
  max-width: 820px;
  border-radius: 16px;
  overflow: hidden;
}
.brand { display: flex; gap: 10px; align-items: center; margin: 0 0 0.35rem; }
.mark {
  width: 32px; height: 32px; flex: none; border-radius: 8px;
  background-image:
    linear-gradient($blue, $blue),
    linear-gradient($green, $green),
    linear-gradient($rose, $rose),
    linear-gradient($gold, $gold);
  background-size: calc(50% - 1.5px) calc(50% - 1.5px);
  background-position: 0 0, 100% 0, 0 100%, 100% 100%;
  background-repeat: no-repeat;
}
.brand strong { display: block; font-size: 0.98rem; letter-spacing: 0.04em; color: $ink; }
.brand em { display: block; margin-top: 1px; font-style: normal; color: $muted; font-size: 0.75rem; }
.side-kicker {
  margin: 0.95rem 0 0.1rem;
  font-size: 0.75rem;
  letter-spacing: 0.14em;
  color: $muted;
}
.side-note { margin: 0.15rem 0 0.2rem; color: $muted; font-size: 0.8rem; line-height: 1.6; }
.hero { max-width: 40rem; margin: 0.15rem 0 0.15rem; }
.eyebrow {
  display: flex; align-items: center; gap: 8px;
  margin: 0 0 8px; color: $green; font-size: 0.78rem;
  letter-spacing: 0.16em; font-weight: 550;
}
.eyebrow i { width: 8px; height: 8px; border-radius: 2px; background: $green; display: inline-block; }
h1.title {
  font-family: $serif !important;
  font-size: 2.15rem !important;
  line-height: 1.25 !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em !important;
  color: $ink !important;
  margin: 0 0 0.45rem !important;
}
.lede { margin: 0; color: $ink_soft; font-size: 1.02rem; line-height: 1.75; }
.stats { display: flex; flex-wrap: wrap; gap: 10px; margin: 0.15rem 0 0.1rem; }
.stat {
  flex: 1 1 160px;
  background: $card;
  border: 1px solid $line;
  border-radius: 16px;
  padding: 12px 16px 11px;
  box-shadow: 0 1px 2px rgba(48, 38, 22, 0.03);
}
.stat span { display: block; color: $muted; font-size: 0.78rem; margin-bottom: 3px; word-break: break-word; }
.stat strong {
  font-size: 1.42rem; font-weight: 600; letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums; color: $ink;
}
.banner {
  background: $banner; border: 1px solid $banner_line; color: $banner_ink;
  border-radius: 12px; padding: 10px 14px; font-size: 0.88rem; line-height: 1.6;
  margin-top: 8px;
}
.section-head { margin: 0.35rem 0 0.05rem; }
.section-head h2, h2.section-title {
  font-family: $serif !important;
  font-size: 1.42rem !important;
  font-weight: 600 !important;
  line-height: 1.3 !important;
  color: $ink !important;
  margin: 0 0 4px !important;
}
.section-head p { margin: 0; color: $muted; font-size: 0.92rem; line-height: 1.65; }
.data-head { margin-top: 0.35rem; }
[data-testid="stPlotlyChart"] {
  background: $card;
  border: 1px solid $line;
  border-radius: 18px;
  padding: 2px 2px 0 0;
  box-shadow: 0 1px 2px rgba(48, 38, 22, 0.04), 0 16px 36px rgba(48, 38, 22, 0.045);
}
.quadrants { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
.qcard { border-radius: 18px; padding: 16px 16px 10px; border: 1px solid rgba(28, 40, 34, 0.05); }
.q-lh { background: #E7EFF6; }
.q-hh { background: #E7F2EA; }
.q-ll { background: #F8EEEF; }
.q-hl { background: #F8F2E2; }
.q-lh .qaction, .q-lh .qbar { color: $blue; background: $blue; }
.q-hh .qaction, .q-hh .qbar { color: $green; background: $green; }
.q-ll .qaction, .q-ll .qbar { color: $rose; background: $rose; }
.q-hl .qaction, .q-hl .qbar { color: $gold; background: $gold; }
.qaction { background: transparent !important; }
.qtop { display: flex; align-items: center; gap: 8px; }
.qbar { width: 8px; height: 8px; border-radius: 99px; display: inline-block; flex: none; }
.qaction { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; }
.qcount {
  margin-left: auto; min-width: 1.6rem; text-align: center;
  background: rgba(255, 252, 248, 0.76); border-radius: 999px;
  padding: 2px 8px; font-size: 0.78rem; color: $ink;
}
.qtitle { margin: 8px 0 6px; font-size: 1.05rem; font-weight: 600; line-height: 1.45; color: $ink; }
.qhint { margin: 0 0 10px; color: $ink_soft; font-size: 0.86rem; line-height: 1.65; }
.phead { text-align: right; color: $muted; font-size: 0.72rem; margin-top: 2px; word-break: break-word; }
.plist { max-height: 240px; overflow: auto; }
.prow {
  display: flex; justify-content: space-between; gap: 16px; align-items: baseline;
  padding: 8px 0; border-top: 1px solid rgba(28, 40, 34, 0.07); font-size: 0.92rem;
}
.pname { color: $ink; word-break: break-word; }
.pmeta { color: $muted; font-variant-numeric: tabular-nums; white-space: nowrap; font-size: 0.84rem; }
.qempty { color: $muted; font-size: 0.86rem; margin: 8px 0 12px; }
.foot { color: $muted; font-size: 0.8rem; margin: 0.2rem 0 0.4rem; }
@media (max-width: 720px) {
  .quadrants { grid-template-columns: 1fr; }
  h1.title { font-size: 1.7rem !important; }
  .stat strong { font-size: 1.2rem; }
  [data-testid="stMainBlockContainer"], .block-container { padding-left: 1rem; padding-right: 1rem; }
}
"""
)


def build_css() -> str:
    values = dict(PALETTE)
    values["sans"] = SANS
    values["serif"] = SERIF
    return CSS.substitute(values)


def hero_html() -> str:
    return (
        '<header class="hero">'
        '<p class="eyebrow"><i></i>门店 · 餐厅 · 工厂</p>'
        '<h1 class="title">一张四象限，看清该加码什么</h1>'
        '<p class="lede">横轴、纵轴自己起名。十字线取平均值，每个品项落到四个格子里。'
        "点击率、毛利率、复购、良品率，都能用。</p>"
        "</header>"
    )


def sidebar_brand_html() -> str:
    return (
        '<div class="brand"><span class="mark" aria-hidden="true"></span>'
        "<div><strong>四象限</strong><em>自由坐标，数据留在本页</em></div></div>"
    )


def stats_html(matrix: Matrix) -> str:
    count = str(len(matrix.items))
    x_kind = axis_kind([item.x for item in matrix.items])
    y_kind = axis_kind([item.y for item in matrix.items])
    x_mean = format_axis_value(matrix.x_mean, x_kind) if matrix.x_mean is not None else "—"
    y_mean = format_axis_value(matrix.y_mean, y_kind) if matrix.y_mean is not None else "—"
    cards = [
        ("有效品项", count),
        (f"{matrix.x_name} · 均值", x_mean),
        (f"{matrix.y_name} · 均值", y_mean),
    ]
    body = "".join(
        f'<div class="stat"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    return f'<div class="stats">{body}</div>'


def notice_html(matrix: Matrix, banner: str | None) -> str:
    notes: list[str] = []
    if banner:
        notes.append(banner)
    if matrix.skipped:
        notes.append(f"有 {matrix.skipped} 行没放进图里：缺名称，或坐标不是数字。空行会跳过。")
    if matrix.items and matrix.flat_x:
        notes.append(f"「{matrix.x_name}」这一列的数都一样，竖线标出的就是这个值。")
    if matrix.items and matrix.flat_y:
        notes.append(f"「{matrix.y_name}」这一列的数都一样，横线标出的就是这个值。")
    if len(matrix.items) > 24:
        notes.append("品项比较多，名字可能会挤。鼠标放在圆点上可以看全称。")
    return "".join(f'<div class="banner">{html.escape(note)}</div>' for note in notes)


def chart_head_html(matrix: Matrix) -> str:
    return (
        '<div class="section-head"><h2>分布</h2>'
        f"<p>实线是「{html.escape(matrix.x_name)}」和「{html.escape(matrix.y_name)}」的平均值。</p></div>"
    )


def cards_html(matrix: Matrix) -> str:
    cards: list[str] = []
    for key in QUADRANT_ORDER:
        spec = QUADRANT_SPECS[key]
        items = matrix.groups[key]
        title = html.escape(quadrant_title(spec, matrix.x_name, matrix.y_name))
        hint = html.escape(quadrant_hint(spec.key, matrix.x_name, matrix.y_name))
        if items:
            x_kind = axis_kind([item.x for item in matrix.items])
            y_kind = axis_kind([item.y for item in matrix.items])
            rows = "".join(
                "<div class='prow'>"
                f"<span class='pname'>{html.escape(item.name)}</span>"
                f"<span class='pmeta'>{html.escape(format_axis_value(item.x, x_kind))} · {html.escape(format_axis_value(item.y, y_kind))}</span>"
                "</div>"
                for item in items
            )
            body = (
                f'<div class="phead">{html.escape(matrix.x_name)} · {html.escape(matrix.y_name)}</div>'
                f'<div class="plist">{rows}</div>'
            )
        else:
            body = '<p class="qempty">这一格还空着</p>'
        cards.append(
            f'<article class="qcard q-{spec.key}">'
            '<div class="qtop"><span class="qbar"></span>'
            f'<span class="qaction">{html.escape(spec.action)}</span>'
            f'<span class="qcount">{len(items)}</span></div>'
            f'<h3 class="qtitle">{title}</h3>'
            f'<p class="qhint">{hint}</p>{body}</article>'
        )
    return (
        '<section class="quad-wrap"><div class="section-head"><h2>四个格子</h2>'
        "<p>跟图同一个方向：左边低、右边高，下边低、上边高。压在均值上的，算偏低。</p></div>"
        f'<div class="quadrants">{"".join(cards)}</div></section>'
    )


def _init_state() -> None:
    if "x_label" not in st.session_state:
        st.session_state.x_label = SAMPLE_X
    if "y_label" not in st.session_state:
        st.session_state.y_label = SAMPLE_Y
    if "table" not in st.session_state:
        st.session_state.table = sample_frame()
    if "editor_version" not in st.session_state:
        st.session_state.editor_version = 0
    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0
    st.session_state.setdefault("upload_token", None)
    st.session_state.setdefault("banner", None)
    st.session_state.setdefault("pending_csv", None)
    st.session_state.setdefault("pending_token", None)


def _load_sample() -> None:
    st.session_state.x_label = SAMPLE_X
    st.session_state.y_label = SAMPLE_Y
    st.session_state.table = sample_frame()
    st.session_state.editor_version += 1
    st.session_state.uploader_version += 1
    st.session_state.upload_token = None
    st.session_state.banner = None
    st.session_state.pending_csv = None
    st.session_state.pending_token = None


def _clear_table() -> None:
    st.session_state.table = empty_frame()
    st.session_state.editor_version += 1
    st.session_state.banner = None


def _apply_pending_upload() -> None:
    raw = st.session_state.get("pending_csv")
    if raw is None:
        return
    token = st.session_state.get("pending_token")
    st.session_state.pending_csv = None
    st.session_state.pending_token = None
    try:
        parsed = parse_csv_bytes(raw)
    except ValueError as exc:
        st.session_state.banner = str(exc)
        st.session_state.upload_token = token
        return
    st.session_state.table = parsed.frame
    if parsed.x_label:
        st.session_state.x_label = parsed.x_label
    if parsed.y_label:
        st.session_state.y_label = parsed.y_label
    st.session_state.editor_version += 1
    st.session_state.banner = parsed.warning
    st.session_state.upload_token = token


def _show_chart(slot: st.delta_generator.DeltaGenerator, figure: go.Figure) -> None:
    slot.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
        selection_mode=(),
        alt="四象限散点图，虚线是平均值",
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.html(sidebar_brand_html())
        st.html('<p class="side-kicker">坐标名称</p>')
        st.text_input("横轴名称", key="x_label", placeholder="例如：点击率")
        st.text_input("纵轴名称", key="y_label", placeholder="例如：毛利率")
        st.html('<p class="side-note">会写进图、均值和四个格子。</p>')
        st.html('<p class="side-kicker">数据</p>')
        st.button("载入示例", type="primary", on_click=_load_sample, width="stretch")
        st.button("清空表格", on_click=_clear_table, width="stretch")
        st.html('<p class="side-note">示例是一家社区餐厅的菜品，会覆盖当前表格和坐标名。</p>')
        st.html('<p class="side-kicker">上传 CSV</p>')
        upload = st.file_uploader(
            "上传 CSV",
            type=["csv"],
            label_visibility="collapsed",
            key=f"upload_{st.session_state.uploader_version}",
        )
        st.html('<p class="side-note">第一列是名称，后面两列是数字。表头会变成坐标名，UTF-8 和 GBK 都可以。</p>')
        if upload is not None:
            raw = upload.getvalue()
            token = hashlib.sha256(raw).hexdigest()
            if token != st.session_state.get("upload_token"):
                st.session_state.pending_csv = raw
                st.session_state.pending_token = token
                st.rerun()


def _render_main() -> None:
    x_name = axis_name(st.session_state.x_label, "横轴")
    y_name = axis_name(st.session_state.y_label, "纵轴")
    st.html(hero_html())
    stats_slot = st.empty()
    chart_head_slot = st.empty()
    chart_slot = st.empty()
    cards_slot = st.empty()

    st.html(
        '<div class="section-head data-head"><h2>品项表</h2>'
        "<p>直接改字、改数，或增删行。图和上面的格子会跟着变。</p></div>"
    )
    edited = st.data_editor(
        st.session_state.table,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_order=["name", "x", "y"],
        column_config={
            "name": st.column_config.TextColumn("品项", width="large", help="菜品、商品或产线"),
            "x": st.column_config.NumberColumn(x_name, width="medium", format="%.2f", help="越大越靠右"),
            "y": st.column_config.NumberColumn(y_name, width="medium", format="%.2f", help="越大越靠上"),
        },
        key=f"editor_{st.session_state.editor_version}",
    )
    if not isinstance(edited, pd.DataFrame):
        edited = _work_frame(pd.DataFrame(edited))
    elif not isinstance(edited.index, pd.RangeIndex) or list(edited.columns) != ["name", "x", "y"]:
        edited = _work_frame(edited)
    st.session_state.table = edited
    st.download_button(
        "下载 CSV",
        data=table_to_csv_bytes(edited, x_name, y_name),
        file_name="四象限.csv",
        mime="text/csv",
    )

    matrix = build_matrix(edited, x_name, y_name)
    stats_slot.html(stats_html(matrix) + notice_html(matrix, st.session_state.get("banner")))
    chart_head_slot.html(chart_head_html(matrix))
    _show_chart(chart_slot, build_figure(matrix))
    cards_slot.html(cards_html(matrix))
    st.html('<p class="foot">数据只在这次打开的页面里计算，关掉就没有了。</p>')


def main() -> None:
    st.set_page_config(
        page_title="四象限矩阵",
        page_icon="◇",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "坐标自由命名的四象限矩阵。数据只留在这次打开的页面里。"},
    )
    _init_state()
    _apply_pending_upload()
    st.html(f"<style>{build_css()}</style>")
    _render_sidebar()
    _render_main()


if __name__ == "__main__":
    main()
