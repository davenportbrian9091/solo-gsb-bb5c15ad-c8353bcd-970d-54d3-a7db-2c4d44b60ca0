"""app.assets 的单元测试（仅依赖标准库，可直接 `python -m unittest discover`）。"""

from __future__ import annotations

import copy
import random
import unittest
from itertools import permutations

from app.assets import MAX_PRELOAD_ASSETS, plan_preloads
from app.compiler import START, CompileError


def cmd(cid, channel="main", anchor=START, offset=0, duration=1000):
    return {
        "id": cid,
        "channel": channel,
        "anchor": anchor,
        "offset_ms": offset,
        "duration_ms": duration,
    }


def asset(aid, command_id, ready=0, load=10, priority=0):
    return {
        "id": aid,
        "command_id": command_id,
        "ready_ms": ready,
        "load_ms": load,
        "priority": priority,
    }


def chain_commands(n, gap=0, duration=100):
    """生成起点依次为 0, d+gap, 2(d+gap), ... 的口令链，返回 (commands, starts)。"""
    commands = [
        {
            "id": "c0",
            "channel": "main",
            "anchor": START,
            "offset_ms": 0,
            "duration_ms": duration,
        }
    ]
    for i in range(1, n):
        commands.append(
            {
                "id": f"c{i}",
                "channel": "main",
                "anchor": f"c{i - 1}",
                "offset_ms": gap,
                "duration_ms": duration,
            }
        )
    step = duration + gap
    return commands, [i * step for i in range(n)]


class BasicScheduleTests(unittest.TestCase):
    def test_serial_no_wait(self):
        # c0 [0,100) 起点 0；c1 锚 c0 [100,200)；素材分别关联 c1 / c0 之外的口令
        # 用起点 100 与 200 的命令：再加 c2 [200,300)
        commands, starts = chain_commands(3)
        self.assertEqual(starts, [0, 100, 200])
        result = plan_preloads(
            commands,
            [asset("x", "c1", 0, 60), asset("y", "c2", 0, 60)],
        )
        self.assertEqual(
            result["loads"],
            [
                {"id": "x", "start_ms": 0, "end_ms": 60},
                {"id": "y", "start_ms": 60, "end_ms": 120},
            ],
        )
        self.assertEqual(result["skipped_ids"], [])

    def test_wait_for_ready_inserts_idle(self):
        commands, _ = chain_commands(3)
        # y 先装（ready 0, 50ms，关联 c1 截止 100），x ready 100 只能 100 开始
        # （x 关联 c2 截止 200：100-101 按时；若 x 先则 100-101，y 101-151 > 100）
        result = plan_preloads(
            commands,
            [asset("x", "c2", 100, 1), asset("y", "c1", 0, 50)],
        )
        self.assertEqual(
            result["loads"],
            [
                {"id": "y", "start_ms": 0, "end_ms": 50},
                {"id": "x", "start_ms": 100, "end_ms": 101},
            ],
        )
        self.assertEqual(result["skipped_ids"], [])

    def test_finish_equal_to_deadline_is_on_time(self):
        commands, _ = chain_commands(2)
        # 素材关联 c1（起点 100）：0-100 恰好完成，允许（含等号）
        result = plan_preloads(
            commands, [asset("x", "c1", 0, 100)]
        )
        self.assertEqual(
            result["loads"], [{"id": "x", "start_ms": 0, "end_ms": 100}]
        )

    def test_start_equal_to_ready_allowed(self):
        commands, _ = chain_commands(2)
        result = plan_preloads(
            commands, [asset("x", "c1", 30, 70)]
        )
        self.assertEqual(
            result["loads"], [{"id": "x", "start_ms": 30, "end_ms": 100}]
        )

    def test_asset_after_deadline_skipped(self):
        commands, _ = chain_commands(2)
        # 关联 c0（起点 0）：任何正耗时素材都无法按时完成
        result = plan_preloads(commands, [asset("x", "c0", 0, 1)])
        self.assertEqual(result["loads"], [])
        self.assertEqual(result["skipped_ids"], ["x"])

    def test_too_long_for_any_order_skipped(self):
        commands, _ = chain_commands(3)
        result = plan_preloads(
            commands,
            [
                asset("x", "c1", 0, 50),
                asset("y", "c2", 0, 200),  # 即使 0 开始也 200 才好，超过 200? 恰 200 可以
            ],
        )
        # y 单独 0-200 恰好 <= 200；但与 x 合计必超 200，只能二选一：条数 1
        ids = {entry["id"] for entry in result["loads"]}
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(result["skipped_ids"]), 1)

    def test_empty_assets(self):
        commands, _ = chain_commands(2)
        result = plan_preloads(commands, [])
        self.assertEqual(result, {"loads": [], "skipped_ids": []})

    def test_empty_commands_and_assets(self):
        result = plan_preloads([], [])
        self.assertEqual(result, {"loads": [], "skipped_ids": []})


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        # c0 [0,100) c1 [100,200) c2 [200,300)
        self.commands, _ = chain_commands(3)

    def test_count_dominates_priority(self):
        # 口令起点：zcmd 100、xcmd 200、ycmd 240（时长各 10，互不影响）
        commands = [
            cmd("base", anchor=START, offset=0, duration=10),
            cmd("zcmd", anchor="base", offset=90, duration=10),   # 起点 100
            cmd("xcmd", anchor="zcmd", offset=90, duration=10),   # 起点 200
            cmd("ycmd", anchor="xcmd", offset=30, duration=10),   # 起点 240
        ]
        # x: load 200 deadline 200、pri 100，只能独占 [0,200)
        # y: load 50 deadline 240、pri 0；z: load 50 deadline 100、pri 0
        # {z, y}：y 0-50、z 50-100（z 恰在截止 100 完成）共 2 条
        # x 与 z 无法同装（总 250 且 z 必须 100 前完成）；
        # x 与 y 也无法同装（x 0-200 后 y 200-250 > 240；y 先则 x 250 才完工）
        result = plan_preloads(
            commands,
            [
                asset("x", "xcmd", 0, 200, priority=100),
                asset("y", "ycmd", 0, 50, priority=0),
                asset("z", "zcmd", 0, 50, priority=0),
            ],
        )
        self.assertEqual(
            [entry["id"] for entry in result["loads"]], ["y", "z"]
        )
        self.assertEqual(result["skipped_ids"], ["x"])

    def test_priority_breaks_count_tie(self):
        commands, _ = chain_commands(3)
        # 二选一：x load 100 d100（c1）pri 1；y load 200 d200（c2）pri 9
        # 两条合计任何顺序都装不下（100+200=300），条数并列 1，取 pri 大的 y
        result = plan_preloads(
            commands,
            [
                asset("x", "c1", 0, 100, priority=1),
                asset("y", "c2", 0, 200, priority=9),
            ],
        )
        self.assertEqual(
            result["loads"], [{"id": "y", "start_ms": 0, "end_ms": 200}]
        )
        self.assertEqual(result["skipped_ids"], ["x"])

    def test_sorted_id_list_lex_breaks_tie(self):
        commands, _ = chain_commands(3)
        # 两个二选一套路：x/y 互斥二选一（同 pri），选 id 较小者 x
        # x d100 load100；y d200 load200：合计装不下，单条都可行
        result = plan_preloads(
            commands,
            [
                asset("x", "c1", 0, 100, priority=5),
                asset("y", "c2", 0, 200, priority=5),
            ],
        )
        self.assertEqual([e["id"] for e in result["loads"]], ["x"])
        self.assertEqual(result["skipped_ids"], ["y"])

    def test_lex_compares_sorted_lists_not_sequences(self):
        # 集合 {a, c} 与 {b, c} 条数 / pri 相同：升序列表 [a,c] < [b,c]
        commands, _ = chain_commands(4)
        # 构造两组互斥对：a d100 load100、b d200 load200（二者互斥，单条可行）；
        # c d300 load1 pri0 总能入选。
        result = plan_preloads(
            commands,
            [
                asset("a", "c1", 0, 100, priority=5),
                asset("b", "c2", 0, 200, priority=5),
                asset("c", "c3", 0, 1, priority=0),
            ],
        )
        self.assertEqual(
            sorted(e["id"] for e in result["loads"]), ["a", "c"]
        )
        self.assertEqual(result["skipped_ids"], ["b"])

    def test_earliest_completion_within_set(self):
        # x d100 ready0 load20（c1）；y d200 ready50 load20（c2）
        # 顺序 x,y：x 0-20，y max(20,50)=50-70，完工 70
        # 顺序 y,x：y 50-70，x 70-90，完工 90
        result = plan_preloads(
            self.commands,
            [asset("x", "c1", 0, 20), asset("y", "c2", 50, 20)],
        )
        self.assertEqual(
            result["loads"],
            [
                {"id": "x", "start_ms": 0, "end_ms": 20},
                {"id": "y", "start_ms": 50, "end_ms": 70},
            ],
        )

    def test_order_lex_breaks_completion_tie(self):
        # 两种顺序都完工 80：取加载顺序 [x, y]
        result = plan_preloads(
            self.commands,
            [asset("x", "c1", 0, 40), asset("y", "c2", 0, 40)],
        )
        self.assertEqual(
            result["loads"],
            [
                {"id": "x", "start_ms": 0, "end_ms": 40},
                {"id": "y", "start_ms": 40, "end_ms": 80},
            ],
        )

    def test_lex_min_order_is_not_always_smallest_first(self):
        """贪心重建必须靠可行性 / 完工 oracle 拒绝字典序更小但更晚完工的首元素。"""
        # c1 d100、c2 d200、c3 d300
        # a: ready0   load60 d300（c3）
        # b: ready100 load1  d100（c1）
        # 集合 {a,b}：
        #   a 先：a 0-60，b max(60,100)=100-101 > 100 -> 不可行
        #   b 先：b max(0,100)=100-101 > 100 -> 也不可行（b 自身必须 100 前完成）
        # 调整：b ready90 load10 d100；a ready0 load60 d300
        #   a 先：0-60，b max(60,90)=90-100 完工 100
        #   b 先：max(0,90)=90-100，a 100-160 完工 160
        # 最早完工 100 -> 顺序 [a, b]（虽然 b 的 id 更大，但 a 必须先做）
        commands, _ = chain_commands(4)
        result = plan_preloads(
            commands,
            [asset("a", "c3", 0, 60), asset("b", "c1", 90, 10)],
        )
        self.assertEqual(
            result["loads"],
            [
                {"id": "a", "start_ms": 0, "end_ms": 60},
                {"id": "b", "start_ms": 90, "end_ms": 100},
            ],
        )

    def test_greedy_oracle_rejects_slower_lex_prefix(self):
        """首元素字典序更小但导致更晚完工时，oracle 应跳过它。"""
        # d100 / d200：
        # a: ready 60 load10 d200（c2）
        # b: ready 0  load60 d100（c1）
        # 集合 {a,b}：
        #   a 先：max(0,60)=60-70，b 70-130 > 100 不可行
        #   b 先：0-60，a max(60,60)=60-70 完工 70 -> 唯一可行顺序 [b, a]
        commands, _ = chain_commands(3)
        result = plan_preloads(
            commands,
            [asset("a", "c2", 60, 10), asset("b", "c1", 0, 60)],
        )
        self.assertEqual(
            [(e["id"], e["start_ms"], e["end_ms"]) for e in result["loads"]],
            [("b", 0, 60), ("a", 60, 70)],
        )


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.commands, _ = chain_commands(2)

    def test_compile_errors_propagate_first(self):
        with self.assertRaises(CompileError) as ctx:
            plan_preloads([cmd("a", "main", "ghost")], [asset("x", "a")])
        self.assertEqual(ctx.exception.code, "MISSING_ANCHOR")

        with self.assertRaises(CompileError) as ctx:
            plan_preloads(
                [cmd("a", "main", START, -10, 5)], [asset("x", "a")]
            )
        self.assertEqual(ctx.exception.code, "NEGATIVE_START")

    def test_empty_asset_id(self):
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(self.commands, [asset("", "c0")])
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_duplicate_asset_id(self):
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(
                self.commands,
                [asset("x", "c0", load=1), asset("x", "c1", load=1)],
            )
        self.assertEqual(ctx.exception.code, "DUPLICATE_ID")
        self.assertEqual(ctx.exception.details["duplicates"], ["x"])

    def test_too_many_assets(self):
        commands, _ = chain_commands(13)
        assets = [asset(f"a{i}", f"c{i}") for i in range(13)]
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(commands, assets)
        self.assertEqual(ctx.exception.code, "TOO_MANY_PRELOAD_ASSETS")
        self.assertEqual(ctx.exception.details["count"], 13)
        self.assertEqual(ctx.exception.details["limit"], MAX_PRELOAD_ASSETS)

    def test_non_positive_load_ms(self):
        for bad in (0, -5):
            with self.assertRaises(CompileError) as ctx:
                plan_preloads(
                    self.commands, [asset("x", "c1", load=bad)]
                )
            self.assertEqual(ctx.exception.code, "INVALID_DURATION")

    def test_negative_ready_ms(self):
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(self.commands, [asset("x", "c1", ready=-1)])
        self.assertEqual(ctx.exception.code, "INVALID_READY_MS")
        self.assertEqual(ctx.exception.details["ids"], ["x"])

    def test_negative_priority(self):
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(self.commands, [asset("x", "c1", priority=-1)])
        self.assertEqual(ctx.exception.code, "INVALID_PRIORITY")

    def test_unknown_command_id(self):
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(self.commands, [asset("x", "ghost")])
        self.assertEqual(ctx.exception.code, "ASSET_COMMAND_NOT_FOUND")
        self.assertEqual(ctx.exception.details["ids"], ["ghost"])

    def test_duplicate_command_link(self):
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(
                self.commands,
                [asset("x", "c1", load=1), asset("y", "c1", load=1)],
            )
        self.assertEqual(ctx.exception.code, "DUPLICATE_ASSET_COMMAND")
        self.assertEqual(ctx.exception.details["duplicates"], ["c1"])

    def test_unknown_command_reported_before_duplicate(self):
        # 同一请求里既引用未知口令、又重复关联：先报未知口令
        with self.assertRaises(CompileError) as ctx:
            plan_preloads(
                self.commands,
                [asset("x", "ghost", load=1), asset("y", "ghost", load=1)],
            )
        self.assertEqual(ctx.exception.code, "ASSET_COMMAND_NOT_FOUND")

    def test_inputs_not_mutated(self):
        commands, _ = chain_commands(2)
        assets = [asset("x", "c1", 10, 20, 3)]
        commands_snapshot = copy.deepcopy(commands)
        assets_snapshot = copy.deepcopy(assets)
        plan_preloads(commands, assets)
        self.assertEqual(commands, commands_snapshot)
        self.assertEqual(assets, assets_snapshot)


class BruteForceFuzzTests(unittest.TestCase):
    """随机输入下与全排列暴力枚举结果逐案对照（仅测试代码枚举，核心逻辑用 DP）。"""

    def test_matches_brute_force(self):
        random.seed(20260925)
        for trial in range(500):
            n = random.randint(0, 6)
            # 随机口令链：起点不再是整齐的 100 网格，制造差异化截止时刻
            duration = random.randint(20, 120)
            gap = random.randint(0, 120)
            commands, starts = chain_commands(n + 1, gap=gap, duration=duration)
            # 每条素材随机关联一条口令（可能重复关联 → 期望 422，直接跳过）
            assets = []
            used_commands = set()
            duplicate_link = False
            for i in range(n):
                cid = f"c{random.randrange(n + 1)}"
                if cid in used_commands:
                    duplicate_link = True
                used_commands.add(cid)
                assets.append(
                    asset(
                        f"a{i}",
                        cid,
                        ready=random.randint(0, starts[-1] + 100),
                        load=random.randint(1, duration + gap + 80),
                        priority=random.randint(0, 5),
                    )
                )
            if duplicate_link:
                with self.assertRaises(CompileError) as ctx:
                    plan_preloads(commands, assets)
                self.assertEqual(ctx.exception.code, "DUPLICATE_ASSET_COMMAND")
                continue

            result = plan_preloads(commands, copy.deepcopy(assets))
            expected = self._brute_force(assets, starts)
            self.assertEqual(
                [(e["id"], e["start_ms"], e["end_ms"]) for e in result["loads"]],
                expected,
                trial,
            )
            selected = [e["id"] for e in result["loads"]]
            self.assertEqual(
                result["skipped_ids"],
                sorted(a["id"] for a in assets if a["id"] not in selected),
                trial,
            )
            self._assert_legality(result, assets, starts)

    @staticmethod
    def _brute_force(assets, starts):
        """枚举每个入选子集与其全部排列，按四级目标返回最优加载序列。"""
        best_key = None
        best_seq = None

        for chosen_mask in range(1 << len(assets)):
            chosen = [i for i in range(len(assets)) if chosen_mask >> i & 1]
            count = len(chosen)
            pri_sum = sum(assets[i]["priority"] for i in chosen)
            sorted_ids = tuple(sorted(assets[i]["id"] for i in chosen))
            set_key = (count, pri_sum)

            # 集合级裁决：条数、pri、升序 id 列表
            if best_key is not None:
                if set_key < best_key[0]:
                    continue
                if set_key == best_key[0] and sorted_ids >= best_key[1]:
                    continue

            # 枚举排列找最早完工；并列取加载 id 顺序字典序最小
            order_best_finish = None
            order_best_seq = None
            order_best_plan = None
            for perm in permutations(chosen):
                cursor = 0
                seq = []
                feasible = True
                for i in perm:
                    a = assets[i]
                    start = max(cursor, a["ready_ms"])
                    finish = start + a["load_ms"]
                    deadline = starts[int(a["command_id"][1:])]
                    if finish > deadline:
                        feasible = False
                        break
                    seq.append((a["id"], start, finish))
                    cursor = finish
                if not feasible:
                    continue
                id_seq = tuple(s[0] for s in seq)
                if (
                    order_best_finish is None
                    or cursor < order_best_finish
                    or (cursor == order_best_finish and id_seq < order_best_seq)
                ):
                    order_best_finish = cursor
                    order_best_seq = id_seq
                    order_best_plan = seq

            if order_best_finish is None:
                continue
            best_key = (set_key, sorted_ids)
            best_seq = order_best_plan

        return best_seq if best_seq is not None else []

    def _assert_legality(self, result, assets, starts=None):
        by_id = {a["id"]: a for a in assets}
        cursor = 0
        seen = set()
        for entry in result["loads"]:
            aid = entry["id"]
            self.assertNotIn(aid, seen)
            seen.add(aid)
            a = by_id[aid]
            self.assertGreaterEqual(entry["start_ms"], a["ready_ms"])
            self.assertGreaterEqual(entry["start_ms"], cursor)
            self.assertEqual(
                entry["end_ms"] - entry["start_ms"], a["load_ms"]
            )
            if starts is not None:
                deadline = starts[int(a["command_id"][1:])]
                self.assertLessEqual(entry["end_ms"], deadline)
            cursor = entry["end_ms"]
        self.assertEqual(
            result["skipped_ids"],
            sorted(set(by_id) - seen),
        )


if __name__ == "__main__":
    unittest.main()
