"""素材预载规划核心逻辑（纯标准库实现，不依赖 Web 框架）。

规则概述
--------
* 请求沿用 ``POST /compile`` 的口令数组，复用同一个编排器解析每条口令的起点
  毫秒；每条素材关联一条现有口令，该口令的**起点**即素材加载的截止时刻。
* 素材字段：``id``（唯一、非空）、``command_id``（关联口令，必须存在且各素材
  不可重复关联）、``ready_ms``（最早可开始加载时刻，非负）、``load_ms``
  （加载耗时，正整数）、``priority``（非负整数优先级）。素材至多 12 条。
* 单台加载器从 ``0`` 时刻起**串行、不可中断**地工作；一条素材的实际开始时刻为
  「上一条加载完成时刻」与 ``ready_ms`` 取大，结束时刻必须不晚于关联口令起点，
  否则该素材不能入选（素材可跳过）。
* 目标依次为：① 按时加载条数最多；② 入选素材 ``priority`` 总和最大；
  ③ 入选素材按 id 升序展开的列表字典序最小。同一入选集合内再取**最后一条
  加载完成时刻最早**的加载顺序；完工时刻并列时取加载 id 顺序字典序最小。

实现要点
--------
素材至多 12 条，做子集动态规划：``finish[mask]`` 记录恰好串行加载完集合
``mask`` 时的最早完工时刻（末位枚举，完工时刻对机器初始时刻单调，故每集合
只需一个最早值即可判定可行性与最早完工）。外层在所有可达子集中按条数、
优先级总和、升序 id 列表裁决出入选集合；随后对该集合贪心重建「最早完工并列中
字典序最小」的顺序：每一步尝试最小的下一个 id，用在剩余子集上、以当前加载器
时刻为起点重跑的最早完工 DP 验证它是否仍能达到全局最早完工。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .compiler import CompileError, compile_commands

# 一次预载规划最多允许的素材条数
MAX_PRELOAD_ASSETS = 12


class PreloadError(CompileError):
    """素材预载规划失败的业务异常；复用 CompileError 的 422 转换通道。"""


def plan_preloads(
    commands: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    """求串行加载器上按时加载条数最多的素材预载方案。

    不会修改传入的 ``commands`` / ``assets``。

    :raises CompileError: 口令编排非法（重复 id、依赖环、负起点等，与 /compile 相同）。
    :raises PreloadError: 素材非法（空 id、重复 id、超过 12 条、load_ms 非正、
        ready_ms 为负、priority 为负、关联口令不存在或被重复关联）时抛出，
        ``code`` 可用于程序化区分错误类型。
    """
    # 1) 复用编排器解析并校验现有口令（深拷贝，setdefault 等不会回写调用方数据；
    #    预载只需各口令起点作为截止时刻，跳过 O(n²) 的冲突对枚举）
    compiled = compile_commands(deepcopy(commands), with_conflicts=False)
    starts = {entry["id"]: entry["start_ms"] for entry in compiled["timeline"]}

    # 2) 素材校验，得到按 id 升序的
    #    (id, command_id, ready_ms, load_ms, priority) 列表与各素材截止时刻
    items, deadlines = _validate_assets(assets, starts)

    # 3) 子集动态规划：条数最多 → 优先级总和最大 → 升序 id 列表字典序最小；
    #    同一集合内完工最早、并列加载顺序字典序最小
    return _schedule(items, deadlines)


def _validate_assets(
    assets: list[dict[str, Any]],
    starts: dict[str, int],
) -> tuple[list[tuple[str, str, int, int, int]], list[int]]:
    """校验素材列表，返回按 id 升序的素材元组列表与对应截止时刻列表。"""
    # 1) 空 id
    empty_ids = [asset["id"] for asset in assets if asset["id"] == ""]
    if empty_ids:
        raise PreloadError(
            "INVALID_ID", "素材 id 不能为空", {"ids": empty_ids}
        )

    # 2) 重复 id
    seen: set[str] = set()
    duplicates: list[str] = []
    for asset in assets:
        aid = asset["id"]
        if aid in seen and aid not in duplicates:
            duplicates.append(aid)
        seen.add(aid)
    if duplicates:
        raise PreloadError(
            "DUPLICATE_ID",
            f"存在重复的素材 id: {', '.join(sorted(duplicates))}",
            {"duplicates": sorted(duplicates)},
        )

    # 3) 条数上限
    if len(assets) > MAX_PRELOAD_ASSETS:
        raise PreloadError(
            "TOO_MANY_PRELOAD_ASSETS",
            f"最多 {MAX_PRELOAD_ASSETS} 条素材参与一次预载规划，"
            f"当前 {len(assets)} 条",
            {"count": len(assets), "limit": MAX_PRELOAD_ASSETS},
        )

    # 4) 加载耗时必须为正整数
    bad_load = sorted(
        asset["id"] for asset in assets if asset["load_ms"] <= 0
    )
    if bad_load:
        raise PreloadError(
            "INVALID_DURATION",
            "素材 load_ms 必须为正整数",
            {"ids": bad_load},
        )

    # 5) ready_ms 必须为非负整数
    bad_ready = sorted(
        asset["id"] for asset in assets if asset["ready_ms"] < 0
    )
    if bad_ready:
        raise PreloadError(
            "INVALID_READY_MS",
            "素材 ready_ms 必须为非负整数",
            {"ids": bad_ready},
        )

    # 6) 优先级必须为非负整数
    bad_priority = sorted(
        asset["id"] for asset in assets if asset["priority"] < 0
    )
    if bad_priority:
        raise PreloadError(
            "INVALID_PRIORITY",
            "素材 priority 必须为非负整数",
            {"ids": bad_priority},
        )

    # 7) 关联口令必须存在（先报不存在，再报重复关联：与目标类接口的校验顺序一致）
    unknown = sorted(
        {asset["command_id"] for asset in assets if asset["command_id"] not in starts}
    )
    if unknown:
        raise PreloadError(
            "ASSET_COMMAND_NOT_FOUND",
            f"素材关联的口令 id 不存在: {', '.join(unknown)}",
            {"ids": unknown},
        )

    # 8) 同一口令至多被一条素材关联
    command_seen: set[str] = set()
    duplicate_commands: list[str] = []
    for asset in assets:
        command_id = asset["command_id"]
        if command_id in command_seen and command_id not in duplicate_commands:
            duplicate_commands.append(command_id)
        command_seen.add(command_id)
    if duplicate_commands:
        raise PreloadError(
            "DUPLICATE_ASSET_COMMAND",
            "一条口令至多可被一条素材关联，以下口令被重复关联: "
            f"{', '.join(sorted(duplicate_commands))}",
            {"duplicates": sorted(duplicate_commands)},
        )

    items = sorted(
        (
            (
                asset["id"],
                asset["command_id"],
                asset["ready_ms"],
                asset["load_ms"],
                asset["priority"],
            )
            for asset in assets
        ),
        key=lambda item: item[0],
    )
    deadlines = [starts[command_id] for _, command_id, _, _, _ in items]
    return items, deadlines


def _schedule(
    items: list[tuple[str, str, int, int, int]],
    deadlines: list[int],
) -> dict[str, Any]:
    """子集动态规划求最优入选集合，并贪心重建最早完工 / 字典序最小顺序。"""
    n = len(items)
    if n == 0:
        return {"loads": [], "skipped_ids": []}

    ready = [item[2] for item in items]
    loads = [item[3] for item in items]
    priorities = [item[4] for item in items]

    size = 1 << n

    # 各子集的元素个数、优先级总和与「子集内 id 升序」元组（下标升序即 id 升序）
    popcount = [0] * size
    sub_priority = [0] * size
    id_tuple: list[tuple[str, ...]] = [()] * size
    for mask in range(1, size):
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ low
        popcount[mask] = popcount[rest] + 1
        sub_priority[mask] = sub_priority[rest] + priorities[i]
        id_tuple[mask] = id_tuple[rest] + (items[i][0],)

    # earliest[mask]：加载器从 0 时刻起串行加载 mask 的最早完工时刻；
    # INF 表示该子集不存在任何全部按时完成的加载顺序。
    # 最早完工对后缀追加单调（更晚就绪的加载器不可能更早完工），故每集合只需
    # 保留最早值即可同时支撑可行性判定与最早完工计算。
    INF = 1 << 62
    earliest = _earliest_finish((1 << n) - 1, ready, loads, deadlines, INF)

    # 外层裁决：条数最多 → 优先级总和最大 → 入选 id 升序列表字典序最小
    best_mask = -1
    best_count = -1
    best_priority = -1
    for mask in range(size):
        if earliest[mask] == INF:
            continue
        count = popcount[mask]
        priority = sub_priority[mask]
        if (
            count > best_count
            or (count == best_count and priority > best_priority)
            or (
                count == best_count
                and priority == best_priority
                and (best_mask == -1 or id_tuple[mask] < id_tuple[best_mask])
            )
        ):
            best_mask = mask
            best_count = count
            best_priority = priority

    # 对入选集合贪心重建最早完工并列中字典序最小的加载顺序。
    # 已选前缀后机器在 cursor 时刻空出，试选 i 后机器在 after 时刻空出：
    # 只要剩余集合能从 after 起串行加载、逐条不晚于各自截止时刻且整体不晚于
    # target 完工，i 即可作为下一个（最早完工集合下该整体完工恰为 target）。
    # 该可行性直接用「带初始机器时刻」的最早完工子集 DP 精确判定（n ≤ 12）。
    target = earliest[best_mask]
    order: list[int] = []
    remaining = best_mask
    cursor = 0
    while remaining:
        bits = remaining
        chosen = -1
        while bits:
            low = bits & -bits
            i = low.bit_length() - 1
            bits ^= low
            after = max(cursor, ready[i]) + loads[i]
            if after > deadlines[i]:
                continue
            rest = remaining ^ low
            # 空剩余集时完工即 after；否则用带初始时刻的最早完工 DP 精确判定
            rest_finish = (
                after
                if rest == 0
                else _earliest_from(
                    rest, after, ready, loads, deadlines, INF
                )
            )
            if rest_finish <= target:
                chosen = i
                break
        # best_mask 本身可达，贪心每步至少存在一个能保持目标完工时刻的选择
        assert chosen != -1
        order.append(chosen)
        remaining ^= 1 << chosen
        cursor = max(cursor, ready[chosen]) + loads[chosen]

    # 按选定顺序重放，给出每条素材的实际起止时刻（含等待 ready 的空闲）
    selected: set[int] = set()
    load_entries: list[dict[str, Any]] = []
    cursor = 0
    for i in order:
        selected.add(i)
        start = max(cursor, ready[i])
        finish = start + loads[i]
        load_entries.append(
            {"id": items[i][0], "start_ms": start, "end_ms": finish}
        )
        cursor = finish

    skipped_ids = [items[i][0] for i in range(n) if i not in selected]
    return {"loads": load_entries, "skipped_ids": skipped_ids}


def _earliest_finish(
    universe: int,
    ready: list[int],
    loads: list[int],
    deadlines: list[int],
    inf: int,
) -> list[int]:
    """对 ``universe`` 的全部子集计算从加载器时刻 0 串行加载的最早完工时刻。

    返回的数组对 ``universe`` 外的位下标无定义；不可行子集记为 ``inf``。
    """
    return _earliest_table(universe, 0, ready, loads, deadlines, inf)


def _earliest_from(
    universe: int,
    machine_at: int,
    ready: list[int],
    loads: list[int],
    deadlines: list[int],
    inf: int,
) -> int:
    """加载器 ``machine_at`` 时刻才空出时，串行加载完 ``universe`` 的最早完工。

    每条素材仍只能在各自 ``ready_ms`` 后开始且不晚于自身截止时刻完工；
    不存在任何可行顺序时返回 ``inf``。
    """
    return _earliest_table(
        universe, machine_at, ready, loads, deadlines, inf
    )[universe]


def _earliest_table(
    universe: int,
    machine_at: int,
    ready: list[int],
    loads: list[int],
    deadlines: list[int],
    inf: int,
) -> list[int]:
    """子集最早完工 DP：``finish[mask]`` 为加载完 mask 的最早完工时刻。

    加载器初始在 ``machine_at`` 时刻空出；不可行子集记为 ``inf``。
    末位枚举：最后加载 i 时，其开始 = max(前驱完工, ready_i)，
    完工须 <= i 自身截止时刻。
    """
    size = universe + 1
    finish = [inf] * size
    finish[0] = machine_at
    for mask in range(1, size):
        if mask & ~universe:
            continue
        bits = mask
        best = inf
        while bits:
            low = bits & -bits
            i = low.bit_length() - 1
            bits ^= low
            prev = finish[mask ^ low]
            if prev == inf:
                continue
            # 不可中断串行：开始时刻 = max(上一条完工, ready_ms)
            end = max(prev, ready[i]) + loads[i]
            if end <= deadlines[i] and end < best:
                best = end
        finish[mask] = best
    return finish


