"""四象限矩阵的坐标、分格、读表和页面接线。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

import app


class MatrixLogicTest(unittest.TestCase):
    def test_sample_covers_four_quadrants_around_the_mean(self) -> None:
        matrix = app.build_matrix(app.sample_frame(), app.SAMPLE_X, app.SAMPLE_Y)
        self.assertAlmostEqual(matrix.x_mean, 5.75)
        self.assertAlmostEqual(matrix.y_mean, 39.3)
        self.assertEqual(
            {item.name for item in matrix.groups["hh"]},
            {"手工柠檬茶", "招牌牛肉面", "工作日套餐"},
        )
        self.assertEqual(
            {item.name for item in matrix.groups["lh"]},
            {"季节沙拉", "新品燕麦碗", "桂花甜品"},
        )
        self.assertEqual({item.name for item in matrix.groups["hl"]}, {"可乐", "炸鸡小食"})
        self.assertEqual({item.name for item in matrix.groups["ll"]}, {"白米饭", "儿童餐"})
        self.assertEqual(app.quadrant_title(app.QUADRANT_SPECS["hh"], "点击率", "毛利率"), "高点击率 · 高毛利率")

    def test_axis_names_are_not_hardcoded_into_the_chart(self) -> None:
        matrix = app.build_matrix(app.sample_frame(), "复购", "坪效")
        figure = app.build_figure(matrix)
        self.assertIn("复购", figure.layout.xaxis.title.text)
        self.assertEqual(figure.layout.yaxis.title.text, "")
        texts = " ".join(annotation.text or "" for annotation in figure.layout.annotations)
        self.assertIn("高复购", texts)
        self.assertIn("高坪效", texts)
        self.assertIn("手工柠檬茶", texts)
        self.assertIn("复购8.6%", texts)
        self.assertIn("坪效66%", texts)

    def test_crosshair_is_solid_and_point_labels_include_both_axes(self) -> None:
        matrix = app.build_matrix(app.sample_frame(), "点击率", "毛利率")
        figure = app.build_figure(matrix)
        lines = [shape for shape in figure.layout.shapes if shape.type == "line"]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(line.line.dash, "solid")
            self.assertGreaterEqual(line.line.width, 2.4)
            self.assertEqual(line.opacity, 1)
        texts = " ".join(annotation.text or "" for annotation in figure.layout.annotations)
        self.assertIn("手工柠檬茶<br>点击率8.6%，毛利率66%", texts)
        self.assertIn("可乐<br>点击率9%，毛利率22%", texts)

    def test_percent_formatting_follows_the_value_scale(self) -> None:
        self.assertEqual(app.axis_kind([0.13, 0.55]), "proportion")
        self.assertEqual(app.axis_kind([8.6, 66]), "percent")
        self.assertEqual(app.axis_kind([200, 800]), "number")
        self.assertEqual(app.format_axis_value(0.13, "proportion"), "13%")
        self.assertEqual(app.format_axis_value(8.6, "percent"), "8.6%")
        self.assertEqual(app.format_axis_value(1200, "number"), "1,200")
        counts = app.build_matrix(
            pd.DataFrame({"name": ["甲", "乙"], "x": [200, 800], "y": [3, 9]}),
            "月销量",
            "评分",
        )
        figure = app.build_figure(counts)
        texts = " ".join(annotation.text or "" for annotation in figure.layout.annotations)
        self.assertIn("月销量200", texts)
        self.assertNotIn("月销量200%", texts)
        self.assertIn("评分3%", texts)

    def test_crosshair_follows_the_mean_when_it_is_not_the_axis_midpoint(self) -> None:
        frame = pd.DataFrame(
            {
                "name": ["凉菜", "例汤", "白米饭", "小食", "时蔬", "套餐", "可乐", "礼盒"],
                "x": [10, 15, 20, 25, 30, 40, 50, 90],
                "y": [98, 96, 94, 92, 90, 88, 30, 10],
            }
        )
        matrix = app.build_matrix(frame, "点击率", "毛利率")
        self.assertAlmostEqual(matrix.x_mean, 35)
        self.assertAlmostEqual(matrix.y_mean, 74.75)
        xmin, xmax = matrix.x_limits
        ymin, ymax = matrix.y_limits
        self.assertLess(matrix.x_mean, (xmin + xmax) / 2)
        self.assertGreater(matrix.y_mean, (ymin + ymax) / 2)
        self.assertLess((matrix.x_mean - xmin) / (xmax - xmin), 0.45)
        self.assertGreater((matrix.y_mean - ymin) / (ymax - ymin), 0.55)
        figure = app.build_figure(matrix)
        vertical = next(shape for shape in figure.layout.shapes if shape.type == "line" and shape.x0 == shape.x1)
        horizontal = next(shape for shape in figure.layout.shapes if shape.type == "line" and shape.y0 == shape.y1)
        self.assertEqual(vertical.xref, "x")
        self.assertEqual(horizontal.yref, "y")
        self.assertAlmostEqual(float(vertical.x0), matrix.x_mean)
        self.assertAlmostEqual(float(horizontal.y0), matrix.y_mean)
        self.assertNotAlmostEqual(float(vertical.x0), (xmin + xmax) / 2, places=3)
        self.assertNotAlmostEqual(float(horizontal.y0), (ymin + ymax) / 2, places=3)

    def test_ties_on_the_mean_fall_to_the_low_side(self) -> None:
        frame = pd.DataFrame({"name": ["甲", "乙"], "x": [1, 1], "y": [1, 3]})
        matrix = app.build_matrix(frame, "横轴", "纵轴")
        self.assertEqual(matrix.x_mean, 1)
        self.assertEqual(matrix.y_mean, 2)
        placed = {item.name: item.key for item in matrix.items}
        self.assertEqual(placed, {"甲": "ll", "乙": "lh"})

    def test_blank_rows_are_ignored_and_partial_rows_are_counted(self) -> None:
        frame = pd.DataFrame(
            {
                "name": ["甲", "", "乙", "丙", ""],
                "x": [1, None, None, 3, 4],
                "y": [4, None, 6, 7, 8],
            }
        )
        matrix = app.build_matrix(frame, "A", "B")
        self.assertEqual([item.name for item in matrix.items], ["甲", "丙"])
        self.assertEqual(matrix.skipped, 2)

    def test_negative_values_and_single_point_still_draw(self) -> None:
        frame = pd.DataFrame({"name": ["甲", "乙", "丙", "丁"], "x": [-5, -1, 2, 4], "y": [-10, 3, -2, 8]})
        matrix = app.build_matrix(frame, "成本", "良品率")
        figure = app.build_figure(matrix)
        self.assertEqual(sum(1 for shape in figure.layout.shapes if shape.type == "rect"), 4)
        self.assertEqual(sum(1 for shape in figure.layout.shapes if shape.type == "line"), 2)
        single = app.build_matrix(pd.DataFrame({"name": ["只此一件"], "x": [2], "y": [3]}), "A", "B")
        self.assertEqual(single.items[0].key, "ll")
        self.assertTrue(single.flat_x and single.flat_y)
        empty = app.build_figure(app.build_matrix(app.empty_frame(), "A", "B"))
        self.assertIn("载入示例", empty.layout.annotations[0].text)

    def test_sample_labels_do_not_cover_each_other_or_the_dots(self) -> None:
        matrix = app.build_matrix(app.sample_frame(), app.SAMPLE_X, app.SAMPLE_Y)
        issues = app.label_issues(app.layout_point_labels(matrix))
        self.assertEqual(issues, [])

    def test_names_are_escaped_in_the_cards(self) -> None:
        frame = pd.DataFrame({"name": ["<script>", "正常"], "x": [1, 3], "y": [2, 4]})
        matrix = app.build_matrix(frame, "点击率", "毛利率")
        rendered = app.cards_html(matrix)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_csv_roundtrip_and_encodings(self) -> None:
        raw = app.table_to_csv_bytes(app.sample_frame(), "点击率", "毛利率")
        parsed = app.parse_csv_bytes(raw)
        self.assertEqual(parsed.x_label, "点击率")
        self.assertEqual(parsed.y_label, "毛利率")
        self.assertEqual(len(parsed.frame), 10)

        same_axis = app.table_to_csv_bytes(app.sample_frame(), "指标", "指标")
        parsed_same = app.parse_csv_bytes(same_axis)
        self.assertEqual(parsed_same.x_label, "指标")
        self.assertEqual(parsed_same.y_label, "指标")

        gbk = "品项,备注,点击率,毛利率\n茶,热卖,8.6,66%\n饭,主食,\"1,200\",40\n".encode("gbk")
        parsed_gbk = app.parse_csv_bytes(gbk)
        self.assertEqual(parsed_gbk.x_label, "点击率")
        self.assertEqual(parsed_gbk.y_label, "毛利率")
        self.assertEqual(parsed_gbk.frame.loc[0, "name"], "茶")
        self.assertAlmostEqual(parsed_gbk.frame.loc[0, "y"], 66)
        self.assertAlmostEqual(parsed_gbk.frame.loc[1, "x"], 1200)

        generic = "名称,x,y\nA,1,2\n".encode("utf-8-sig")
        parsed_generic = app.parse_csv_bytes(generic)
        self.assertIsNone(parsed_generic.x_label)
        self.assertIsNone(parsed_generic.y_label)
        self.assertEqual(parsed_generic.frame.loc[0, "name"], "A")

    def test_csv_rejects_short_and_empty_files(self) -> None:
        with self.assertRaises(ValueError):
            app.parse_csv_bytes("只有,两列\n1,2\n".encode())
        with self.assertRaises(ValueError):
            app.parse_csv_bytes(b"")
        with self.assertRaises(ValueError):
            app.parse_csv_bytes("品项,备注,说明\nA,B,C\n".encode())

    def test_stylesheet_uses_the_same_palette(self) -> None:
        css = app.build_css()
        self.assertIn(app.PALETTE["paper"], css)
        self.assertIn(app.PALETTE["green"], css)
        self.assertIn("Noto Sans CJK SC", css)


class PageWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        from streamlit.testing.v1 import AppTest

        self.at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
        self.assertFalse(self.at.exception, self.at.exception)

    def _blob(self) -> str:
        return "\n".join(element.value for element in self.at.markdown) + "\n".join(
            getattr(element, "value", "") or getattr(element, "body", "") or "" for element in self.at.html
        )

    def _click(self, label: str):
        for button in self.at.sidebar.button:
            if button.label == label:
                return button.click().run()
        self.fail(label)

    def test_renaming_an_axis_updates_the_quadrant_copy(self) -> None:
        self.assertEqual(self.at.sidebar.text_input[0].value, "点击率")
        self.at.sidebar.text_input[0].set_value("复购率").run()
        self.assertFalse(self.at.exception, self.at.exception)
        blob = self._blob()
        self.assertIn("高复购率 · 高毛利率", blob)
        self.assertIn("实线是「复购率」和「毛利率」的平均值", blob)

    def test_clear_and_sample_and_added_row_update_the_lists(self) -> None:
        self._click("清空表格")
        self.assertFalse(self.at.exception, self.at.exception)
        blob = self._blob()
        self.assertIn("这一格还空着", blob)
        self.assertNotIn("手工柠檬茶", blob)

        self._click("载入示例")
        blob = self._blob()
        self.assertIn("手工柠檬茶", blob)
        self.assertIn("高点击率 · 高毛利率", blob)
        self.assertIn("5.75", blob)
        self.assertIn("39.3", blob)

        self.at.session_state["table"] = pd.concat(
            [self.at.session_state["table"], pd.DataFrame([{"name": "测试新品", "x": 99, "y": 99}])],
            ignore_index=True,
        )
        self.at.session_state["editor_version"] = int(self.at.session_state["editor_version"]) + 1
        self.at.run()
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertIn("测试新品", self._blob())


if __name__ == "__main__":
    unittest.main()
