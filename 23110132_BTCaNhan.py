import tkinter as tk
from tkinter import ttk  # dùng ttk để lấy Combobox
from queue import Queue  # hàng đợi
from collections import deque
import random  # thư viện random
import time  # thư viện tính thời gian chạy thuật toán
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Toplevel
import heapq
import math

"""-------------------------------------
            Cấu hình bàn cờ
    ------------------------------------"""
N = 8
CELL = 40
PAD = 4
sang = "#f5deb3"
toi = "#8b5a2b"
trang = "#111"
den = "#fff"

vi_tri_xe = []

# dic lưu thời gian chạy
times = {"BFS": None, "DFS": None, "DLS": None, "IDS": None, "UCS": None, "GS": None, "A*": None, "HC": None,
         "SA": None, "BS": None,
         "GENE": None, "AND-OR": None, "BELIEF": None, "PARTIAL": None, "BTK": None, "FWD": None, "AC3": None}
expand_nodes = {"BFS": None, "DFS": None, "DLS": None, "IDS": None, "UCS": None, "GS": None, "A*": None, "HC": None,
                "SA": None, "BS": None,
                "GENE": None, "AND-OR": None, "BELIEF": None, "PARTIAL": None, "BTK": None, "FWD": None, "AC3": None}

"""-------------------------------------
                Hàm BFS
    ------------------------------------"""


def bfs(n=8, goal=None):
    steps = []
    start = tuple()
    goal = tuple(goal) if goal else tuple(range(n))
    q = Queue()
    q.put(start)
    parent = {start: None}

    expanded_count = 0  # biến đếm số node mở rộng

    while not q.empty():
        state = q.get()
        steps.append(state)  # lưu lại để hiển thị
        expanded_count += 1  # mỗi lần lấy ra từ queue thì +1

        row = len(state)
        if row == n and state == goal:
            break
        used = set(state)
        for col in range(n):
            if col not in used:
                child = state + (col,)
                if child not in parent:
                    parent[child] = state
                    q.put(child)
    expand_nodes["BFS"] = expanded_count
    return steps


"""-------------------------------------
                Hàm DFS
    ------------------------------------"""


def dfs(n=8, goal=None):
    start = tuple()
    goal = tuple(goal)
    stack = [start]
    yield ("visit", start)
    expanded_count = 0  # biến đếm số node mở rộng
    while stack:
        state = stack.pop()
        expanded_count += 1  # mỗi lần lấy ra từ queue thì +1
        if state == goal:
            expand_nodes["DFS"] = expanded_count
            return state, expanded_count
        used = set(state)
        for col in reversed(range(n)):
            if col not in used:
                child = state + (col,)
                stack.append(child)
                yield ("visit", child)
    yield ("fail", ())


"""-------------------------------------
                Hàm DLS
    ------------------------------------"""
expand_dls = 0  # biến toàn cục đếm số nút mở rộng


def dls(goal, limit, on_visit=None):
    def gen():
        global expand_dls
        expand_dls = 0  # reset trước mỗi lần chạy

        state = []  # p[r] = c cho r=0..len(state)-1
        used_cols = set()  # các cột đã dùng

        yield ("init", state.copy())

        # chạy đệ quy
        res = yield from recursive(state, used_cols, row=0,
                                   limit=limit, goal=goal, on_visit=on_visit)

        # nếu không thành công trong giới hạn -> phát failure
        if res is not True:
            yield ("failure", [])

        # phát thêm số nút mở rộng
        yield ("expanded_count", expand_dls)

    return gen()


def recursive(state, used_cols, row, limit, goal, on_visit):
    """
    Hàm đệ quy *phát bước*.
    Trả về:
      - True  : SUCCESS
      - None  : CUTOFF (bị chặn bởi limit ở đâu đó bên dưới)
      - False : FAILURE (hết nhánh mà không thấy nghiệm)
    """
    global expand_dls
    expand_dls += 1  # mỗi lần gọi recursive = mở rộng thêm 1 nút

    # gọi callback nếu hợp lệ
    if callable(on_visit):
        on_visit(state)

    # điều kiện dừng: đủ 8 hàng
    if row == N:
        if (goal is None) or (state == goal):
            yield ("done", state.copy())
            expand_nodes["DLS"] = expand_dls
            return True  # SUCCESS
        return False  # FAILURE

    # bị chặn bởi depth limit
    if limit == 0:
        yield ("cutoff", state.copy())
        return None  # CUTOFF

    cutoff_occurred = False

    for c in range(N):
        if c in used_cols:
            continue

        # PLACE
        state.append(c)
        used_cols.add(c)
        yield ("place", state.copy())

        res = yield from recursive(state, used_cols, row + 1, limit - 1, goal, on_visit)

        if res is True:  # SUCCESS truyền ngược lên
            return True
        elif res is None:  # có CUTOFF ở dưới
            cutoff_occurred = True

        # BACKTRACK
        state.pop()
        used_cols.remove(c)
        yield ("backtrack", state.copy())

    # Không còn con để mở rộng
    if cutoff_occurred:
        return None  # CUTOFF
    return False  # FAILURE


"""-------------------------------------
                Hàm IDS
    ------------------------------------"""


def ids(n=8, goal=None):
    start = tuple()
    goal = tuple(goal)
    yield ("visit", start)
    max_limit = n
    expanded_count = 0

    for limit in range(max_limit + 1):
        stack = [start]
        visited = set()
        while stack:
            state = stack.pop()
            depth = len(state)
            expanded_count += 1
            if state == goal:
                yield ("done", state)
                expand_nodes["IDS"] = expanded_count
                return
            if depth >= limit:
                continue
            if state in visited:
                continue
            visited.add(state)
            used = set(state)
            for col in range(n - 1, -1, -1):
                if col not in used:
                    child = state + (col,)
                    stack.append(child)
                    yield ("visit", child)
        yield ("fail", ())


"""-------------------------------------
                Hàm UCS
    ------------------------------------"""


def generate_next_states(state, n):
    """Sinh ra các trạng thái con bằng cách đặt quân tiếp theo vào các cột trống"""
    next_states = []
    row = state.index(-1) if -1 in state else n  # tìm hàng đầu tiên chưa đặt
    if row == n:
        return next_states

    used_cols = set([c for c in state if c != -1])
    for col in range(n):
        if col not in used_cols:
            new_state = list(state)
            new_state[row] = col
            next_states.append(tuple(new_state))
    return next_states


def cost_function(state, goal):
    cost = 0
    # 1. Đếm quân đặt sai vị trí
    for i in range(len(state)):
        if state[i] != goal[i]:
            cost += 1

    # 2. Thêm chi phí xung đột (xe ăn nhau)
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] == state[j]:  # cùng cột
                cost += 2
            if i == j:  # cùng hàng (nếu state encode hàng khác)
                cost += 2
    return cost


def ucs(n, goal):
    start = tuple([-1] * n)  # chưa đặt xe nào
    frontier = [(0, start)]  # (cost, state)
    visited = set()
    expand = 0

    while frontier:
        cost, state = heapq.heappop(frontier)
        if state in visited:
            continue
        visited.add(state)
        expand += 1

        yield ("expand", state, cost, expand)

        if state == tuple(goal):
            expand_nodes["UCS"] = expand
            yield ("done", state, cost, expand)
            return

        # Sinh trạng thái con
        for next_state in generate_next_states(state, n):
            new_cost = cost + cost_function(next_state, goal)
            heapq.heappush(frontier, (new_cost, next_state))


"""-------------------------------------
                Hàm GS
    ------------------------------------"""


def hes(row: int, col: int, goal_state):
    """
    Heuristic cơ bản cho 8 quân xe:
    khoảng cách cột hiện tại tới cột đích của hàng 'row'.
    goal_state: list/tuple p[r] = c
    """
    return abs(col - goal_state[row])


def _snapshot_frontier(frontier, k=None):
    """Trả snapshot PQ đã sắp xếp theo cost để in/log."""
    snap = [(c, list(s)) for (c, s) in frontier]
    snap.sort(key=lambda x: x[0])  # sort lại để xem log
    return snap if k is None else snap[:k]


def gs(n=8, goal_state=None, heuristic_fn=hes, snapshot_top=None):
    """
    Phát:
      ("visit",    state, cost, frontier_snapshot)
      ("frontier", None,  None, frontier_snapshot)
      ("done",     state, cost, frontier_snapshot)

    heuristic_fn: hàm kiểu (row, col, goal_state) -> số
    snapshot_top: nếu đặt số nguyên k, chỉ log k phần tử đầu của PQ.
    """
    if goal_state is None:
        goal_state = tuple(range(n))

    start = tuple()
    frontier = []
    heapq.heappush(frontier, (0, start))  # priority queue khởi tạo
    seen = set()
    expand = 0

    yield ("frontier", None, None, _snapshot_frontier(frontier, snapshot_top))

    while frontier:
        cost, state = heapq.heappop(frontier)  # lấy node có cost nhỏ nhất
        if state in seen:
            continue
        seen.add(state)
        expand += 1

        yield ("visit", list(state), cost, _snapshot_frontier(frontier, snapshot_top))

        row = len(state)
        if row == N:
            yield ("done", list(state), cost, _snapshot_frontier(frontier, snapshot_top))
            expand_nodes["GS"] = expand
            return

        used = set(state)
        for col in range(n):
            if col in used:
                continue
            step_cost = heuristic_fn(row, col, goal_state)  # dùng hàm heuristic riêng
            next_state = state + (col,)
            heapq.heappush(frontier, (cost + step_cost, next_state))

        yield ("frontier", None, None, _snapshot_frontier(frontier, snapshot_top))


"""-------------------------------------
                Hàm A*
    ------------------------------------"""


def hes_a(state, goal_state):
    """
    Heuristic: tổng khoảng cách cột hiện tại so với cột goal.
    """
    cost = 0
    for r, c in enumerate(state):
        cost += abs(c - goal_state[r])
    return cost


def A(n=8, goal_state=None):
    expand = 0  # reset trước khi chạy

    if goal_state is None:
        goal_state = tuple(range(n))
    else:
        goal_state = tuple(goal_state)

    start = tuple()
    frontier = [(0, start, 0)]  # (f, state, g)
    seen = set()

    while frontier:
        f, state, g = heapq.heappop(frontier)
        if state in seen:
            continue
        seen.add(state)

        expand += 1  # mỗi lần lấy node ra mở rộng → tăng bộ đếm

        # Nếu đạt goal
        if len(state) == n and state == goal_state:
            yield ("done", state, g)
            expand_nodes["A*"] = expand
            return

        row = len(state)
        used = set(state)

        # Sinh successor
        for col in range(n):
            if col not in used:
                new_state = state + (col,)
                g_new = g + 1
                h_new = hes_a(new_state, goal_state)
                f_new = g_new + h_new
                heapq.heappush(frontier, (f_new, new_state, g_new))

        yield ("expand", state, g)


"""-------------------------------------
                Hàm Hill-Climbing
    ------------------------------------"""


def heuristic(state, goal):
    """
    Trả tổng |col - goal_col| theo hàng.
    Hỗ trợ hai dạng state:
      - [c0, c1, ...]     (danh sách cột)
      - [(r0,c0),(r1,c1), ...] (cặp hàng-cột)
    """
    if not state:
        return 0
    # nếu là danh sách các (r,c)
    if isinstance(state[0], (tuple, list)) and len(state[0]) == 2:
        return sum(abs(c - goal[r]) for r, c in state)
    # nếu là danh sách các cột
    return sum(abs(c - goal[r]) for r, c in enumerate(state))


def hill_climbing(start, goal):
    """
    Hill-Climbing với state = list cột.
    Trả về list các bước: (state, h)
    """
    # ép về list cột
    current = list(start)
    goal = list(goal)
    h_curr = heuristic(current, goal)
    steps = [(current[:], h_curr)]
    expand = 0

    while True:
        best_h = h_curr
        best_state = None
        expand += 1
        # thử di chuyển từng hàng sang cột khác
        n = len(current)
        for r in range(n):
            c_now = current[r]
            for c in range(n):
                if c == c_now:
                    continue
                new_state = current[:]
                new_state[r] = c
                h_new = heuristic(new_state, goal)
                if h_new < best_h:
                    best_h = h_new
                    best_state = new_state

        # không còn cải thiện
        if best_state is None:
            break

        # nhận cải thiện tốt nhất
        current = best_state
        h_curr = best_h
        steps.append((current[:], h_curr))

        if current == goal:
            expand_nodes["HC"] = expand
            break

    return steps


"""-------------------------------------
            Hàm Simulated_Annealing
    ------------------------------------"""


def sa(start, goal, T0=100, alpha=0.95, max_iter=1000, stall_limit=100, T_min=1e-3):
    current = start[:]
    h_curr = heuristic(current, goal)
    T = T0
    stall = 0
    expand = 0
    for it in range(max_iter):
        expand += 1
        # Nếu đạt đích thì dừng ngay
        if current == goal:
            print(expand)
            break

        # Nếu nhiệt độ quá thấp thì dừng
        if T < T_min:
            break

        # chọn quân và cột mới
        r = random.randint(0, len(start) - 1)
        c_new = random.randint(0, len(start) - 1)
        new_state = current[:]
        new_state[r] = c_new

        h_new = heuristic(new_state, goal)
        delta = h_new - h_curr

        if delta < 0:
            prob = 1.0
        else:
            prob = math.exp(-delta / T)

        # Chấp nhận move
        if delta < 0 or random.random() < prob:
            current = new_state
            h_curr = h_new
            stall = 0  # reset khi có cải thiện
        else:
            stall += 1

        # yield kết quả để hiển thị
        yield (h_curr, current[:], prob, T)

        # Nếu lặp quá lâu không cải thiện thì dừng
        if stall >= stall_limit:
            break

        # giảm nhiệt độ
        T *= alpha
    expand_nodes["SA"] = expand


"""-------------------------------------
            Hàm Beam Search
    ------------------------------------"""


def bs(n=8, goal_state=None, k=2, heuristic_fn=None, snapshot_top=None):
    """
    Beam Search (BS)
    Yield log ở dạng:
      ("frontier", None, None, frontier_snapshot)
      ("visit",    state, cost, frontier_snapshot)
      ("done",     state, cost, frontier_snapshot)
    """
    if goal_state is None:
        goal_state = tuple(range(n))

    start = tuple()
    frontier = [(0, start)]
    expand = 0

    # Log frontier ban đầu
    yield ("frontier", None, None, _snapshot_frontier(frontier, snapshot_top))

    while frontier:
        expand += 1
        next_frontier = []

        # Lấy k phần tử tốt nhất từ frontier
        for cost, state in sorted(frontier)[:k]:
            yield ("visit", list(state), cost, _snapshot_frontier(frontier, snapshot_top))

            row = len(state)
            if row == n:
                yield ("done", list(state), cost, _snapshot_frontier(frontier, snapshot_top))
                expand_nodes["BS"] = expand
                return

            used = set(state)
            for col in range(n):
                if col not in used:
                    step_cost = heuristic_fn(row, col, goal_state) if heuristic_fn else 0
                    next_state = state + (col,)
                    next_frontier.append((cost + step_cost, next_state))

        # Cập nhật frontier với k phần tử tốt nhất từ next_frontier
        frontier = sorted(next_frontier)[:k]
        yield ("frontier", None, None, _snapshot_frontier(frontier, snapshot_top))


"""-------------------------------------
            Hàm Genetic
    ------------------------------------"""
import random


# --- Hàm tính độ thích nghi (fitness) ---
def fitness(state, goal):
    goal = tuple(goal)
    # Đếm số vị trí trong state trùng với goal
    return sum(1 for i in range(len(state)) if state[i] == goal[i])


# --- Hàm chính của giải thuật di truyền ---
def gene(n, goal, ca_the, ti_le_dot_bien, max_the_he):
    goal = tuple(goal)
    expand = 0

    # 1. Khởi tạo quần thể ngẫu nhiên (population)
    population = [tuple(random.sample(range(n), n)) for _ in range(ca_the)]
    # 2. Tiến hóa qua các thế hệ
    for gen in range(max_the_he):

        # --- chọn cá thể tốt nhất trong thế hệ hiện tại ---
        best = max(population, key=lambda s: fitness(s, goal))
        expand += len(population)
        fit = fitness(best, goal)
        yield ("step", best, fit)  # trả ra bước này cho giao diện/log

        # Nếu đã tìm thấy cá thể đúng goal thì kết thúc
        if best == goal:
            yield ("done", best, fit)
            expand_nodes["GENE"] = expand
            return

        # --- Chọn lọc (Selection): lấy nửa trên cá thể tốt ---
        scored = sorted(population, key=lambda s: fitness(s, goal), reverse=True)
        parents = scored[:ca_the // 2]
        # --- Lai ghép (Crossover) ---
        children = []
        while len(children) < ca_the:
            p1, p2 = random.sample(parents, 2)  # chọn ngẫu nhiên 2 cha mẹ
            cut = random.randint(1, n - 1)  # điểm cắt
            child = list(p1[:cut])  # phần đầu lấy từ p1
            used = set(child)
            # phần còn lại lấy từ p2 nhưng bỏ trùng
            for x in p2:
                if x not in used:
                    child.append(x)
                    used.add(x)
            children.append(tuple(child))

        # --- Đột biến (Mutation) ---
        so_ca_the_dot_bien = max(1, int(ti_le_dot_bien * len(children)))
        for i in random.sample(range(len(children)), so_ca_the_dot_bien):
            s = list(children[i])
            a, b = random.sample(range(n), 2)  # chọn 2 gen ngẫu nhiên
            s[a], s[b] = s[b], s[a]  # hoán đổi
            children[i] = tuple(s)

        # --- Cập nhật quần thể ---
        population = children
    expand_nodes["GENE"] = expand


"""-------------------------------------
            Hàm AND–OR Search
------------------------------------"""

expanded_and_or = 0


def goal_test(state, goal):
    return tuple(state) == tuple(goal)


def actions(state, row):
    N = len(state)
    used = {c for c in state[:row] if c != -1}
    return [c for c in range(N) if c not in used]


def result(state, row, col):
    s = list(state)
    s[row] = col
    return tuple(s)


def _snap(frontier, k=None):
    if k is None: return []
    out = []
    for i, (cost, (a, s2)) in enumerate(frontier[:k]):
        out.append((cost, list(s2)))
    return out


def and_or_graph_search_with_logs(initial_state, goal, snapshot_top=20):
    global expanded_and_or
    expanded_and_or = 0  # reset mỗi lần chạy
    start = tuple(initial_state)
    status = yield from _or_search(start, goal, path=[], snapshot_top=snapshot_top)
    return status  # "success" | "failure"


def _or_search(state, goal, path, snapshot_top):
    global expanded_and_or
    expanded_and_or += 1
    if goal_test(state, goal):
        # ĐÃ ĐẠT MỤC TIÊU
        yield ("done", list(state), 0, [])
        expand_nodes["AND-OR"] = expanded_and_or
        return "success"

    if state in path:
        yield ("failure", list(state), 0, [])
        return "failure"

    if -1 not in state:  # hết chỗ đặt mà chưa đạt goal
        yield ("failure", list(state), 0, [])
        return "failure"

    row = state.index(-1)
    frontier = []
    for a in actions(state, row):
        s2 = result(state, row, a)
        heapq.heappush(frontier, (0, (a, s2)))

    yield ("frontier", list(state), 0, _snap(frontier, snapshot_top))

    any_ok = False
    while frontier:
        _, (a, s2) = heapq.heappop(frontier)
        yield ("visit", list(s2), 0, _snap(frontier, snapshot_top))

        st = yield from _and_search([s2], goal, path + [state], snapshot_top)
        if st == "success":
            any_ok = True
            yield ("success", list(s2), 0, _snap(frontier, snapshot_top))
            return "success"

    # Không có nhánh nào thành công
    yield ("failure", list(state), 0, _snap(frontier, snapshot_top))
    return "failure"


def _and_search(states, goal, path, snapshot_top):
    # Tất cả con đều phải success
    for s in states:
        st = yield from _or_search(s, goal, path, snapshot_top)
        if st == "failure":
            yield ("failure", list(s), 0, [])
            return "failure"
    return "success"


def AND_OR_Graph_Search(initial_state, goal):
    start = tuple(initial_state)
    return OR_Search(start, goal, [])


def OR_Search(state, goal, path):
    if goal_test(state, goal):
        return []
    if state in path:
        return None
    if -1 not in state:
        return None

    row = state.index(-1)
    for a in actions(state, row):
        s2 = result(state, row, a)
        plan = AND_Search([s2], goal, path + [state])
        if plan is not None:
            return [(row, a)] + plan
    return None


def AND_Search(states, goal, path):
    plan_acc = []
    for s in states:
        plan = OR_Search(s, goal, path)
        if plan is None:
            return None
        plan_acc.extend(plan)
    return plan_acc


"""-------------------------------------
            Hàm belief search
    ------------------------------------"""
from collections import deque
import time


# =========================================
# HÀM CƠ BẢN
# =========================================
def goal_test(state, goal):
    return state == tuple(goal)


def actions(state, row):
    """Trả về danh sách cột có thể đặt quân ở hàng row (tránh trùng cột)."""
    N = len(state)
    used_cols = {c for c in state[:row] if c != -1}
    return [c for c in range(N) if c not in used_cols]


def result(state, row, col):
    """Trả về trạng thái mới sau khi đặt 1 quân xe vào hàng row, cột col."""
    new_state = list(state)
    new_state[row] = col
    return tuple(new_state)


def goal_belief(belief, goal):
    """True nếu MỌI trạng thái (ma trận) trong belief đều bằng goal."""
    return all(s == goal for s in belief)


def belief_successors(belief):
    """Sinh các belief kế tiếp sau khi mở rộng 1 tập belief."""
    successors = []
    for b in belief:
        if -1 not in b:  # state đã đầy đủ rồi, bỏ qua
            continue
        row = b.index(-1)
        for a in actions(b, row):
            new_state = result(b, row, a)
            successors.append(new_state)
    return successors


def belief_search_with_logs(initial_belief, goal):
    """
    Thuật toán tìm kiếm trong không gian niềm tin (Belief Space Search).
    Có yield log từng bước để hiển thị trong GUI.
    """
    frontier = deque()
    # Mỗi phần tử trong frontier là (belief, kế hoạch)
    frontier.append((initial_belief, []))
    visited = set()
    step = 0
    expanded_count = 0

    while frontier:
        belief, plan = frontier.popleft()
        step += 1
        expanded_count += 1

        yield ("expand", belief, step)

        # Kiểm tra goal
        if goal_belief(belief, goal):
            yield ("goal", belief, step)
            expand_nodes["BELIEF"] = expanded_count
            return ("success", expanded_count)

        frozen_b = frozenset(belief)
        if frozen_b in visited:
            continue
        visited.add(frozen_b)

        # Duy trì từng belief riêng biệt để không hòa trộn nhánh
        for s in belief:
            if -1 not in s:
                continue
            row = s.index(-1)
            for a in actions(s, row):
                new_state = result(s, row, a)
                new_belief = {new_state}
                new_plan = plan + [new_belief]
                frontier.append((new_belief, new_plan))

    yield ("failure", None, step)
    return ("failure", expanded_count)


"""-------------------------------------
            Hàm partial observable search
    ------------------------------------"""


def update_belief(belief, observation):
    """Lọc các trạng thái không phù hợp với quan sát."""
    new_belief = set()
    for s in belief:
        ok = True
        for row, col in observation.items():
            if s[row] != -1 and s[row] != col:
                ok = False
                break
        if ok:
            new_belief.add(s)
    return new_belief


def expand_belief(belief):
    """Sinh các belief kế tiếp."""
    successors = []
    for s in belief:
        if -1 not in s:
            continue
        row = s.index(-1)
        for a in actions(s, row):
            new_state = result(s, row, a)
            successors.append(new_state)
    return set(successors)


def partial_observable_search_with_logs(initial_belief, goal, observations):
    """
    Tìm kiếm trong môi trường thấy một phần.
    Ghi log từng bước cho GUI.
    """
    frontier = deque()
    frontier.append((initial_belief, []))
    visited = set()
    step = 0
    expanded_count = 0

    while frontier:
        belief, plan = frontier.popleft()
        step += 1
        expanded_count += 1

        yield ("expand", belief, step)

        # Cập nhật belief theo quan sát (nếu có)
        if step - 1 < len(observations):
            obs = observations[step - 1]
            belief = update_belief(belief, obs)
            yield ("observe", obs, len(belief), step)

        # Kiểm tra goal
        if goal_belief(belief, goal):
            yield ("goal", belief, step)
            expand_nodes["PARTIAL"] = expanded_count
            return ("success", expanded_count)

        frozen_b = frozenset(belief)
        if frozen_b in visited:
            continue
        visited.add(frozen_b)

        successors = expand_belief(belief)
        if successors:
            new_plan = plan + [successors]
            frontier.append((successors, new_plan))

    yield ("failure", None, step)
    return ("failure", expanded_count)


"""-------------------------------------
            Hàm Backtracking
    ------------------------------------"""


def is_safe(position, row, col):
    for (r, c) in position:
        if r == row or c == col:
            return False
    return True


def btk(n=N):
    solved = False  # cờ kiểm tra đã có nghiệm chưa
    expand = 0

    def backtrack(i, position):
        nonlocal solved
        nonlocal expand
        if solved:  # nếu đã tìm thấy thì không làm gì nữa
            return
        if i == n:
            yield ("done", position[:])
            solved = True
            expand_nodes["BTK"] = expand
            return
        cells = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(cells)
        for (row, col) in cells:

            if solved:
                return
            expand += 1
            if is_safe(position, row, col):
                position.append((row, col))
                yield ("place", i, (row, col), position[:])
                yield from backtrack(i + 1, position)
                if not solved:  # chỉ gỡ nếu chưa có nghiệm
                    position.pop()
                    yield ("remove", i, (row, col), position[:])

    yield from backtrack(0, [])


def pairs_to_state(pairs, n):
    """Chuyển danh sách (row, col) -> state dạng [col0, col1, ...]"""
    state = [-1] * n
    for r, c in pairs:
        state[r] = c
    return state


"""-------------------------------------
            Hàm Forward Checking
    ------------------------------------"""


def fwd(n=N):
    domains = {i: [(r, c) for r in range(n) for c in range(n)] for i in range(n)}
    assignment = {}
    solved = False
    expand = 0

    def forward(i, domains, assignment):
        nonlocal solved
        nonlocal expand
        if solved:
            return
        if i == n:
            yield ("done", assignment.copy())
            expand_nodes["FWD"] = expand
            solved = True
            return

        values = domains[i][:]
        random.shuffle(values)

        for (row, col) in values:
            if solved:
                return
            expand += 1
            assignment[i] = (row, col)
            yield ("place", i, (row, col), assignment.copy())
            new_domains = {j: list(domains[j]) for j in range(n)}
            ok = True
            for j in range(i + 1, n):
                new_domains[j] = [(r, c) for (r, c) in new_domains[j] if r != row and c != col]
                if not new_domains[j]:
                    ok = False
                    break
            if ok:
                yield from forward(i + 1, new_domains, assignment)
            if not solved:  # chỉ backtrack nếu chưa có nghiệm
                assignment.pop(i, None)
                yield ("remove", i, (row, col), assignment.copy())

    yield from forward(0, domains, assignment)


"""-------------------------------------
            Hàm AC3
    ------------------------------------"""


def ac3_steps(n=8):
    """Generator: thuật toán AC-3 đặt n quân xe trên bàn cờ n x n"""

    domains = {i: [(r, c) for r in range(n) for c in range(n)] for i in range(n)}
    assignment = {}
    solved = False
    expand = 0

    def revise(domains, xi, xj):
        revised = False
        to_remove = []
        for (ri, ci) in domains[xi]:
            ok = False
            for (rj, cj) in domains[xj]:
                if ri != rj and ci != cj:
                    ok = True
                    break
            if not ok:
                to_remove.append((ri, ci))
        for v in to_remove:
            domains[xi].remove(v)
            revised = True
        return revised

    def ac3(domains):
        from collections import deque
        queue = deque([(xi, xj) for xi in domains for xj in domains if xi != xj])
        while queue:
            xi, xj = queue.popleft()
            if revise(domains, xi, xj):
                if not domains[xi]:
                    return False
                for xk in domains:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))
        return True

    def backtrack(i, domains, assignment):
        nonlocal solved, expand
        if solved: return
        if i == n:
            yield ("done", assignment.copy())
            expand_nodes["AC3"] = expand
            solved = True
            return

        values = domains[i][:]
        random.shuffle(values)

        for (row, col) in values:
            if solved: return
            expand += 1
            assignment[i] = (row, col)
            yield ("place", i, (row, col), assignment.copy())

            new_domains = {j: list(domains[j]) for j in domains}
            new_domains[i] = [(row, col)]
            for j in range(i + 1, n):
                new_domains[j] = [(r, c) for (r, c) in new_domains[j] if r != row and c != col]

            if ac3(new_domains):
                yield from backtrack(i + 1, new_domains, assignment)

            if not solved:
                assignment.pop(i, None)
                yield ("remove", i, (row, col), assignment.copy())

    yield from backtrack(0, domains, assignment)


"""-------------------------------------
            Vẽ bàn cờ
    ------------------------------------"""


def veBanCo(canvas):
    canvas.delete("all")
    W = PAD * 2 + N * CELL
    canvas.config(width=W, height=W)
    for r in range(N):
        for c in range(N):
            x0 = PAD + c * CELL
            y0 = PAD + r * CELL
            x1 = x0 + CELL
            y1 = y0 + CELL
            fill = sang if (r + c) % 2 == 0 else toi
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, width=0)


"""-------------------------------------
            Random trạng thái đích
    ------------------------------------"""


def tao_trang_thai_dich(n=8):
    """Sinh ngẫu nhiên vị trí 8 quân xe trên bàn cờ n x n"""
    trang_thai = list(range(n))  # [0,1,2,3,4,5,6,7]
    random.shuffle(trang_thai)  # xáo trộn thành hoán vị ngẫu nhiên
    return trang_thai


"""-------------------------------------
                Vẽ xe
    ------------------------------------"""


def veXe(canvas, xe=None):
    if xe:
        for r in range(len(xe)):
            c = xe[r]
            if c == -1:
                continue
            cx = PAD + c * CELL + CELL // 2
            cy = PAD + r * CELL + CELL // 2
            o_toi = (r + c) % 2 == 1
            color = den if o_toi else trang
            canvas.create_text(
                cx, cy, text="♖", fill=color,
                font=("Segoe UI Symbol", int(CELL * 0.75), "bold")
            )


"""-------------------------------------
                Tạo frame
    ------------------------------------"""


def taoFrame(parent, text, xe=None):
    f = tk.Frame(parent, bg="#f3ebca")
    f.place(relx=0.5, rely=0.5, anchor="center")
    lb = tk.Label(f, text=text, font=("Segoe UI", 12, "bold"), bg="#f3ebca")
    lb.pack(pady=(0, 6))
    cv = tk.Canvas(f, highlightthickness=0, bg="white")
    cv.pack(fill="both", expand=True)
    veBanCo(cv)
    veXe(cv, xe)
    return f, cv


"""-------------------------------------
                Main
    ------------------------------------"""


def main():
    root = tk.Tk()
    root.title("Bàn cờ 8x8")
    root.configure(bg="#f3ebca")
    root.state("zoomed")

    """-------------------------------------
                Cấu hình lưới
    ------------------------------------"""
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=0)  # Cột trái (nút chức năng)
    root.grid_columnconfigure(1, weight=1)  # Cột phải (bàn cờ)

    """-------------------------------------
            Cột phải chứa 3 bàn cờ ngang
    ------------------------------------"""
    right_panel = tk.Frame(root, bg="#f3ebca")
    right_panel.grid(row=0, column=1, padx=20, pady=20, sticky="n")

    """-------------------------------------
                Trạng thái ban đầu
    ------------------------------------"""
    frame_start, cv1 = taoFrame(right_panel, "Trạng thái ban đầu", vi_tri_xe)
    frame_start.pack(side="left", padx=20)

    """-------------------------------------
            Trạng thái mục tiêu
    ------------------------------------"""
    frame_goal, cv2 = taoFrame(right_panel, "Trạng thái đích", list(reversed(vi_tri_xe)))
    frame_goal.pack(side="left", padx=20)

    # --- Biến lưu trạng thái mục tiêu ---
    goal_state = []
    current_state = []

    """-------------------------------------
        Hàm làm mới trạng thái mục tiêu
    ------------------------------------"""

    def lam_moi_dich():
        nonlocal goal_state
        goal_state = tao_trang_thai_dich()  # random mới
        veBanCo(cv2)
        veXe(cv2, goal_state)

    """-------------------------------------
        Hàm làm mới 
    ------------------------------------"""

    def lam_moi():
        veBanCo(cv1)
        veXe(cv1, vi_tri_xe)

    """-------------------------------------
        Hàm làm mới - random
    ------------------------------------"""

    def lam_moi_random():
        nonlocal current_state
        current_state = tao_trang_thai_dich()  # random mới
        veBanCo(cv1)
        veXe(cv1, current_state)

    """-------------------------------------
            Tạo cột trái: chức năng
    ------------------------------------"""
    left_panel = tk.Frame(root, bg="#f3ebca")
    left_panel.grid(row=0, column=0, padx=30, pady=30, sticky="ns")

    """-------------------------------------
            Khung chọn thuật toán
    ------------------------------------"""
    frame_algo = tk.Frame(left_panel, bg="#f3ebca", bd=2, relief="groove")
    frame_algo.pack(padx=10, pady=10, fill="x")
    lbl = tk.Label(frame_algo, text="Chọn Thuật Toán", font=("Segoe UI", 14, "bold"), bg="#f3ebca", fg="purple")
    lbl.pack(anchor="w", padx=5, pady=5)
    algo_var = tk.StringVar()
    combo = ttk.Combobox(frame_algo, textvariable=algo_var, state="readonly", width=25, font=("Segoe UI", 14))
    combo['values'] = ("BFS", "DFS", "DLS", "IDS", "UCS", "GS", "A*", "Hill Climbing", "Simulated Annealing", "Beam",
                       "Genetic", "And-Or", "Belief Search", "Partial Observable", "Backtracking", "Forward Checking",
                       "AC3")
    combo.current(0)  # mặc định chọn BFS
    combo.pack(padx=10, pady=5)

    """-------------------------------------
            Khung chọn chức năng
    ------------------------------------"""
    frame_all = tk.Frame(left_panel, bg="#f3ebca", bd=2, relief="groove")
    frame_all.pack(padx=10, pady=10, fill="x")
    # --- Nhãn chức năng ---
    tk.Label(frame_all, text="Chức năng", font=("Segoe UI", 14, "bold"), bg="#f3ebca").pack(pady=15)
    # --- Các nút chức năng ---
    tk.Button(frame_all, text="Làm mới (Ban đầu)", width=25, height=1, font=("Segoe UI", 12, "bold"), bg="#ffffff",
              command=lam_moi).pack(pady=8)
    tk.Button(frame_all, text="Làm mới (Random)", width=25, height=1, font=("Segoe UI", 12, "bold"), bg="#ffffff",
              command=lam_moi_random).pack(pady=8)

    tk.Button(frame_all, text="Làm mới (Đích)", width=25, height=1, font=("Segoe UI", 12, "bold"), bg="#ffffff",
              command=lam_moi_dich).pack(pady=8)
    btn_run = tk.Button(frame_all, text="Chạy", width=25, height=1, font=("Segoe UI", 12, "bold"), bg="#d1f7d1")
    btn_run.pack(pady=8)

    btn_run_all = tk.Button(frame_all, text="Chạy hết", width=25, height=1, font=("Segoe UI", 12, "bold"), bg="#d1d1f7")
    btn_run_all.pack(pady=8)
    btn_show_chart = tk.Button(frame_all, text="Hiển thị biểu đồ", width=25, height=1, font=("Segoe UI", 12, "bold"),
                               bg="#d0a1ff")
    btn_show_chart.pack(pady=8)

    """-------------------------------------
            Trạng thái hiện tại
    ------------------------------------"""
    frame_current, cv3 = taoFrame(right_panel, "Trạng thái hiện tại", None)
    frame_current.pack(side="left", padx=20)

    """-------------------------------------
    Cột phải chia tỉ lệ 3 bàn cờ và ghi rõ các bước
    ------------------------------------"""
    right_panel = tk.Frame(root, bg="#f3ebca")
    right_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
    # Chia right_panel thành 2 hàng
    right_panel.grid_rowconfigure(0, weight=0)  # hàng 0: 70% (3 bàn cờ)
    right_panel.grid_rowconfigure(1, weight=10)  # hàng 1: 30% (log)
    right_panel.grid_columnconfigure(0, weight=1)

    # --- Hàng 0: 3 bàn cờ ---
    boards_frame = tk.Frame(right_panel, bg="#f3ebca")
    boards_frame.grid(row=0, column=0, sticky="nsew")
    frame_start, cv1 = taoFrame(boards_frame, "Trạng thái ban đầu", vi_tri_xe)
    frame_start.pack(side="left", padx=20, pady=10, expand=True)
    frame_goal, cv2 = taoFrame(boards_frame, "Trạng thái đích", list(reversed(vi_tri_xe)))
    frame_goal.pack(side="left", padx=20, pady=10, expand=True)
    frame_current, cv3 = taoFrame(boards_frame, "Trạng thái hiện tại", None)
    frame_current.pack(side="left", padx=20, pady=10, expand=True)

    # --- Hàng 1: log ---
    log_frame = tk.Frame(right_panel, bg="#f3ebca", bd=2, relief="groove")
    log_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
    tk.Label(log_frame, text="Các bước chạy thuật toán:", font=("Segoe UI", 12, "bold"), bg="#f3ebca", fg="black").pack(
        anchor="w", padx=10, pady=5)
    text_log = tk.Text(log_frame, wrap="word", font=("Consolas", 11))
    text_log.pack(side="left", fill="both", expand=True, padx=10, pady=5)
    scrollbar = tk.Scrollbar(log_frame, command=text_log.yview)
    scrollbar.pack(side="right", fill="y")
    text_log.config(yscrollcommand=scrollbar.set)

    def clear_log():
        text_log.delete("1.0", tk.END)  # Xóa toàn bộ nội dung log

    tk.Button(frame_all, text="Xoá", width=25, height=1, font=("Segoe UI", 12, "bold"), bg="#f7abf3",
              command=clear_log).pack(pady=8)

    """-------------------------------------
            Vẽ biểu đồ
    ------------------------------------"""

    def show_chart():
        # --- Tạo cửa sổ mới ---
        chart_win = Toplevel(root)
        chart_win.title("So sánh thời gian chạy thuật toán")
        chart_win.geometry("1200x800")  # cao hơn để chứa 2 biểu đồ
        chart_win.configure(bg="#f3ebca")

        # --- Frame chính ---
        main_frame = tk.Frame(chart_win, bg="#f3ebca")
        main_frame.pack(fill="both", expand=True)

        # --- Tạo canvas và scrollbar ---
        canvas_frame = tk.Canvas(main_frame, bg="#f3ebca", highlightthickness=0)
        canvas_frame.pack(side="left", fill="both", expand=True)

        v_scroll = tk.Scrollbar(main_frame, orient="vertical", command=canvas_frame.yview)
        v_scroll.pack(side="right", fill="y")

        h_scroll = tk.Scrollbar(chart_win, orient="horizontal", command=canvas_frame.xview)
        h_scroll.pack(side="bottom", fill="x")

        canvas_frame.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # --- Frame chứa nội dung ---
        content_frame = tk.Frame(canvas_frame, bg="#f3ebca")
        canvas_window = canvas_frame.create_window((0, 0), window=content_frame, anchor="nw")

        def on_configure(event):
            canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))

        content_frame.bind("<Configure>", on_configure)

        # --- Cuộn bằng chuột ---
        def on_mousewheel(event):
            canvas_frame.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def on_shift_mousewheel(event):
            canvas_frame.xview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_frame.bind_all("<MouseWheel>", on_mousewheel)
        canvas_frame.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)

        # ================= Biểu đồ 1: Thời gian =================
        fig1 = Figure(figsize=(11, 3), dpi=100)
        ax1 = fig1.add_subplot(111)

        algos = list(times.keys())
        vals = [times[a] if times[a] is not None else 0 for a in algos]

        ax1.bar(algos, vals, width=0.4, color="skyblue", edgecolor="black")
        ax1.set_ylabel("Thời gian (giây)")
        ax1.set_title("So sánh thời gian chạy thuật toán")
        ax1.set_xlim(-0.5, len(algos) - 0.5)

        for i, v in enumerate(vals):
            ax1.text(i, v + (max(vals) * 0.02 if max(vals) > 0 else 0.001),
                     f"{v:.4f}", ha="center", fontsize=10)

        fig1.tight_layout()
        chart_canvas1 = FigureCanvasTkAgg(fig1, master=content_frame)
        chart_canvas1.draw()
        chart_canvas1.get_tk_widget().pack(pady=20, padx=20)

        # ================= Biểu đồ 2: Số nút mở rộng =================
        fig2 = Figure(figsize=(11, 3), dpi=100)
        ax2 = fig2.add_subplot(111)

        # expanded_nodes: dict, ví dụ {"BFS": 120, "DFS": 98, ...}
        expanded_vals = []
        for a in algos:
            v = expand_nodes.get(a, 0)  # nếu thiếu key -> 0
            expanded_vals.append(0 if v is None else v)

        ax2.bar(algos, expanded_vals, width=0.4, color="lightgreen", edgecolor="black")
        ax2.set_ylabel("Số nút mở rộng")
        ax2.set_title("So sánh số nút mở rộng của các thuật toán")
        ax2.set_xlim(-0.5, len(algos) - 0.5)

        for i, v in enumerate(expanded_vals):
            ax2.text(i, v + (max(expanded_vals) * 0.02 if max(expanded_vals) > 0 else 0.001),
                     f"{v}", ha="center", fontsize=10)

        fig2.tight_layout()
        chart_canvas2 = FigureCanvasTkAgg(fig2, master=content_frame)
        chart_canvas2.draw()
        chart_canvas2.get_tk_widget().pack(pady=20, padx=20)

        # --- Nút đóng ---
        tk.Button(
            content_frame,
            text="Đóng",
            font=("Segoe UI", 12),
            command=chart_win.destroy,
            bg="#ffe1a1"
        ).pack(pady=20)

    btn_show_chart.config(command=show_chart)

    # Biến cờ để dừng after khi nhấn "Chạy hết"
    running = True

    """-------------------------------------
            Hàm chạy từng bước
    ------------------------------------"""

    def run_algorithm():
        nonlocal running
        running = True
        algo_name = algo_var.get()

        if algo_name == "BFS":
            """-------------------------------------
                    Chạy BFS
            ------------------------------------"""
            steps = []
            goal = goal_state
            result = bfs(8, goal)
            steps = result
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:  # nếu bị dừng thì thoát
                    return
                if i < len(steps):
                    veBanCo(cv3)
                    veXe(cv3, steps[i])
                    text_log.insert("end", f"Bước {i}: {steps[i]}\n")
                    text_log.see("end")
                    root.after(1, show_step, i + 1)

            show_step()

        elif algo_name == "DFS":
            """-------------------------------------
                    Chạy DFS
            ------------------------------------"""
            goal = goal_state
            result = dfs(8, goal)  # result là generator

            steps = list(result)  # Lưu toàn bộ các bước
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:  # nếu bị dừng thì thoát
                    return
                if i < len(steps):
                    kind, state = steps[i]
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Bước {i}: {kind} -> {state}\n")
                    text_log.see("end")
                    root.after(300, show_step, i + 1)
                else:
                    text_log.insert("end", "=== Kết thúc DFS ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "DLS":
            """-------------------------------------
                    Chạy DLS
            ------------------------------------"""
            depth_limit = 8  # giới hạn độ sâu
            goal = goal_state

            # dls() trả về generator gồm (kind, state)
            result = dls(goal, 8, 0)
            steps = list(result)

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} (limit={depth_limit}) ===\n")

            def show_step(i=0):
                if not running:
                    return
                if i < len(steps):
                    kind, state = steps[i]
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Bước {i}: {kind} -> {state}\n")
                    text_log.see("end")
                    root.after(300, show_step, i + 1)
                else:
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "IDS":
            """-------------------------------------
                    Chạy IDS
            ------------------------------------"""
            goal = goal_state
            result = ids(8, goal)  # result là generator

            steps = list(result)  # Lưu toàn bộ các bước
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:  # nếu bị dừng thì thoát
                    return
                if i < len(steps):
                    kind, state = steps[i]
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Bước {i}: {kind} -> {state}\n")
                    text_log.see("end")
                    root.after(10, show_step, i + 1)
                else:
                    text_log.insert("end", "=== Kết thúc DFS ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "UCS":
            """-------------------------------------
                        Chạy UCS (step-by-step)
            ------------------------------------"""
            goal = goal_state
            result = ucs(8, goal)  # result là generator
            gen = ucs(8, goal)

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step():
                if not running:
                    return
                try:
                    step = next(gen)  # lấy 1 bước từ UCS
                    if step[0] == "expand":
                        _, state, cost, expand = step
                        veBanCo(cv3)
                        veXe(cv3, state)
                        text_log.insert("end", f"Expand: {state}, cost={cost}, expand={expand}\n")
                    elif step[0] == "done":
                        _, state, cost, expand = step
                        veBanCo(cv3)
                        veXe(cv3, state)
                        text_log.insert("end", f"Hoàn thành: {state}, cost={cost}, expand={expand}\n")
                        text_log.insert("end", f"=== UCS kết thúc ===\n")
                        text_log.see("end")
                        return
                    text_log.see("end")
                    root.after(50, show_step)  # chạy tiếp sau 50ms
                except StopIteration:
                    text_log.insert("end", f"=== UCS kết thúc ===\n")
                    text_log.see("end")

            show_step()


        elif algo_name == "GS":
            """-------------------------------------
                        Chạy GS
            ------------------------------------"""
            goal = goal_state  # list/tuple p[r]=c

            result = gs(8, goal)  # hoặc: gs(N, goal, heuristic_fn=hes, snapshot_top=20)
            steps = list(result)

            if not steps:
                text_log.insert("end", "[GS] Không có bước nào.\n")
                text_log.see("end")
                return

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:
                    return
                if i < len(steps):
                    step = steps[i]
                    # Hỗ trợ cả (kind, state, cost) và (kind, state, cost, pq)
                    if isinstance(step, (list, tuple)) and len(step) >= 3:
                        kind, state, cost = step[0], step[1], step[2]
                        pq = step[3] if len(step) >= 4 else None
                    else:
                        # fallback (rất hiếm)
                        kind, state, cost, pq = "visit", step, None, None

                    # Vẽ khi có state (visit/done)
                    if kind in ("visit", "done") and state is not None:
                        veBanCo(cv3)
                        veXe(cv3, state)

                    # Log bước
                    if kind == "frontier":
                        # In PQ nếu có
                        if pq is None:
                            text_log.insert("end", f"Bước {i}: FRONTIER (không có snapshot)\n")
                        else:
                            text_log.insert("end", f"Bước {i}: FRONTIER size={len(pq)}\n")
                            for j, (cst, st) in enumerate(pq):
                                text_log.insert("end", f"   [{j}] cost={cst} state={st}\n")
                    else:
                        text_log.insert("end", f"Bước {i}: {kind} -> {state} | cost={cost}\n")

                    text_log.see("end")
                    # tốc độ animate
                    root.after(150, show_step, i + 1)
                else:
                    # Kết thúc + in thời gian
                    text_log.insert("end", f"Kết quả cuối: {steps[-1]}\n")
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "A*":
            """-------------------------------------
                        Chạy A*
            ------------------------------------"""
            goal = goal_state
            result = A(8, goal)  # result là generator

            steps = list(result)  # Lưu toàn bộ các bước
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:  # nếu bị dừng thì thoát
                    return
                if i < len(steps):
                    kind, state, cost = steps[i]  # lấy đủ 3 giá trị
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Bước {i}: {kind} -> {state}, g={cost}\n")
                    text_log.see("end")
                    root.after(10, show_step, i + 1)
                else:
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "Hill Climbing":
            """-------------------------------------
                        Chạy Hill Climbing
            ------------------------------------"""
            goal = goal_state
            start = current_state
            result = hill_climbing(start, goal)  # result là list steps

            steps = list(result)  # Lưu toàn bộ các bước
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:  # nếu bị dừng thì thoát
                    return
                if i < len(steps):
                    state, h_val = steps[i]  # lấy đúng 2 giá trị
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Bước {i}: h={h_val} -> {state}\n")
                    text_log.see("end")
                    root.after(300, show_step, i + 1)
                else:
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "Simulated Annealing":
            """-------------------------------------
                    Chạy Simulated Annealing
            ------------------------------------"""
            goal = goal_state
            start = current_state
            steps = list(sa(start, goal, T0=100, alpha=0.95, max_iter=1000))

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:
                    return
                if i < len(steps):
                    rec = steps[i]

                    # Linh hoạt 2/3/4 trường
                    if len(rec) == 4:
                        h, state, prob, T = rec
                    elif len(rec) == 3:
                        h, state, prob = rec
                        T = None
                    elif len(rec) == 2:
                        state, h = rec
                        prob = None
                        T = None
                    else:
                        # fallback
                        h = None;
                        state = rec;
                        prob = None;
                        T = None

                    veBanCo(cv3)
                    veXe(cv3, state)

                    if prob is None and T is None:
                        text_log.insert("end", f"Bước {i}: h={h} -> {state}\n")
                    elif T is None:
                        text_log.insert("end", f"Bước {i}: h={h}, p={prob:.4f} -> {state}\n")
                    else:
                        text_log.insert("end", f"Bước {i}: h={h}, p={prob:.4f}, T={T:.4g} -> {state}\n")

                    text_log.see("end")
                    root.after(200, show_step, i + 1)
                else:
                    text_log.insert("end", f"Kết quả cuối: {steps[-1][1]}\n")
                    text_log.insert("end", f"=== Simulated Annealing kết thúc ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "Beam":
            """-------------------------------------
                        Chạy BS
            ------------------------------------"""
            goal = goal_state  # list/tuple p[r]=c

            result = bs(8, goal_state=goal, heuristic_fn=hes, snapshot_top=20)
            steps = list(result)

            if not steps:
                text_log.insert("end", "[BS] Không có bước nào.\n")
                text_log.see("end")
                return

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:
                    return
                if i < len(steps):
                    step = steps[i]
                    # Hỗ trợ cả (kind, state, cost) và (kind, state, cost, pq)
                    if isinstance(step, (list, tuple)) and len(step) >= 3:
                        kind, state, cost = step[0], step[1], step[2]
                        pq = step[3] if len(step) >= 4 else None
                    else:
                        # fallback (rất hiếm)
                        kind, state, cost, pq = "visit", step, None, None

                    # Vẽ khi có state (visit/done)
                    if kind in ("visit", "done") and state is not None:
                        veBanCo(cv3)
                        veXe(cv3, state)

                    # Log bước
                    if kind == "frontier":
                        # In PQ nếu có
                        if pq is None:
                            text_log.insert("end", f"Bước {i}: FRONTIER (không có snapshot)\n")
                        else:
                            text_log.insert("end", f"Bước {i}: FRONTIER size={len(pq)}\n")
                            for j, (cst, st) in enumerate(pq):
                                text_log.insert("end", f"   [{j}] cost={cst} state={st}\n")
                    else:
                        text_log.insert("end", f"Bước {i}: {kind} -> {state} | cost={cost}\n")

                    text_log.see("end")
                    # tốc độ animate
                    root.after(150, show_step, i + 1)
                else:
                    # Kết thúc + in thời gian
                    text_log.insert("end", f"Kết quả cuối: {steps[-1]}\n")
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "Genetic":
            """-------------------------------------
                        Chạy Genetic 
            ------------------------------------"""
            goal = goal_state
            n = len(goal)

            # Gọi hàm gene → result là generator, convert thành list steps
            result = gene(n, goal, ca_the=10, ti_le_dot_bien=0.1, max_the_he=100)
            steps = list(result)  # mỗi step có dạng ("step"/"done", state, fitness)

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:  # nếu bị dừng thì thoát
                    return
                if i < len(steps):
                    _, state, fit = steps[i]  # unpack đúng 3 giá trị
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Gen {i}: fitness={fit} -> {state}\n")
                    text_log.see("end")
                    root.after(300, show_step, i + 1)  # tiếp tục bước sau sau 300ms
                else:
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "And-Or":
            """-------------------------------------
                        Chạy AND–OR Graph Search
            ------------------------------------"""
            goal = tuple(goal_state)
            initial = tuple([-1] * 8)

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            import time
            start_time = time.time()
            steps = []

            # Gọi generator thuật toán
            gen = and_or_graph_search_with_logs(initial_state=initial, goal=goal, snapshot_top=20)

            try:
                while True:
                    steps.append(next(gen))
            except StopIteration as e:
                # Có thể e.value chỉ là "success" hoặc None
                status = e.value if e.value is not None else "success"

            elapsed = time.time() - start_time
            times["AND-OR"] = elapsed

            # ==== HIỂN THỊ TỪNG BƯỚC ====
            def show_step(i=0):
                if not running:  # nếu người dùng nhấn Dừng
                    return

                if i < len(steps):
                    kind, state, cost, pq = steps[i]
                    veBanCo(cv3)

                    # chỉ vẽ khi là trạng thái hợp lệ
                    if state and kind.lower() in ("visit", "success", "done"):
                        veXe(cv3, state)

                    # ghi log
                    if kind.lower() == "frontier":
                        text_log.insert("end", f"Bước {i}: FRONTIER size={len(pq)}\n")
                        for j, (c, st) in enumerate(pq):
                            text_log.insert("end", f"   [{j}] cost={c} state={st}\n")
                    elif kind.lower() == "done":
                        text_log.insert("end", f"Bước {i}: ✅ DONE -> {state}\n")
                    elif kind.lower() == "failure":
                        text_log.insert("end", f"Bước {i}: ❌ FAILURE -> {state}\n")
                    elif kind.lower() == "success":
                        text_log.insert("end", f"Bước {i}: ✅ SUCCESS -> {state}\n")
                    elif kind.lower() == "visit":
                        text_log.insert("end", f"Bước {i}: VISIT -> {state}\n")
                    else:
                        text_log.insert("end", f"Bước {i}: {kind.upper()} -> {state}\n")

                    text_log.see("end")
                    root.after(300, show_step, i + 1)

                else:
                    # Hoàn tất
                    done_steps = [s for s in steps if s[0].lower() == "done"]
                    if done_steps:
                        _, final_state, _, _ = done_steps[-1]
                        veBanCo(cv3)
                        veXe(cv3, final_state)

                    text_log.insert("end", f"\n⏱️ Thời gian chạy: {elapsed:.4f} giây\n")
                    text_log.insert("end", f"=== Kết thúc {algo_name} ({status}) ===\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "Belief Search":
            """-------------------------------------
                        Chạy Belief Space Search (Run Animation)
            ------------------------------------"""
            # Bước 1: Khởi tạo
            goal = tuple(goal_state)
            N = len(goal_state)  # 🔧 Khai báo N để tránh lỗi
            initial = {tuple([-1] * N)}  # belief ban đầu: chưa biết vị trí nào

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            import time
            start_time = time.time()
            steps = []

            # Bước 2: Gọi generator của thuật toán
            gen = belief_search_with_logs(initial, goal)

            try:
                while True:
                    steps.append(next(gen))
            except StopIteration as e:
                status = e.value if e.value is not None else "success"

            elapsed = time.time() - start_time

            # Bước 3: Hiển thị từng bước (animation)
            def show_step(i=0):
                if not running:  # Nếu người dùng nhấn Dừng
                    return

                if i < len(steps):
                    entry = steps[i]

                    # --- Hỗ trợ cả 2 kiểu yield khác nhau ---
                    if len(entry) == 3:
                        kind, belief_state, step_id = entry
                    elif len(entry) == 2:
                        kind, belief_state = entry
                        step_id = i + 1
                    else:
                        kind = entry[0]
                        belief_state = entry[1] if len(entry) > 1 else set()
                        step_id = i + 1

                    # --- Vẽ bàn cờ ---
                    veBanCo(cv3)

                    # Hiển thị 1 trạng thái đại diện của belief
                    if belief_state:
                        example_state = next(iter(belief_state))
                        if isinstance(example_state, tuple):
                            veXe(cv3, example_state)

                    # --- Ghi log ---
                    if kind.lower() == "expand":
                        text_log.insert("end",
                                        f"Bước {step_id}: 🔍 Mở rộng {len(belief_state)} trạng thái trong belief\n")
                        for s in belief_state:
                            text_log.insert("end", f"   {s}\n")
                    elif kind.lower() == "goal":
                        text_log.insert("end", f"Bước {step_id}: ✅ Tất cả trạng thái trong belief đạt đích!\n")
                        for s in belief_state:
                            text_log.insert("end", f"   {s}\n")
                    elif kind.lower() == "failure":
                        text_log.insert("end", f"Bước {step_id}: ❌ Thất bại — không còn belief mới\n")
                    else:
                        text_log.insert("end", f"Bước {step_id}: {kind.upper()} -> {belief_state}\n")

                    text_log.see("end")
                    root.after(300, show_step, i + 1)

                else:
                    # --- Sau khi hoàn tất ---
                    text_log.insert("end", f"\n⏱️ Thời gian chạy: {elapsed:.4f} giây\n")
                    text_log.insert("end", f"=== Kết thúc {algo_name} ({status}) ===\n")
                    text_log.see("end")

            # Bước 4: Chạy animation
            show_step()


        elif algo_name == "Partial Observable":
            """-------------------------------------
                        Chạy Partial Observable Search
            ------------------------------------"""
            goal = tuple(goal_state)
            N = len(goal)

            # belief ban đầu: chưa biết gì cả
            initial_belief = {tuple([-1] * N)}

            # danh sách quan sát (có thể tùy chỉnh theo bài)
            observations = [
                {0: 0},
                {1: 1, 2: 2},
                {3: 3, 4: 4},
                {5: 5, 6: 6, 7: 7}
            ]

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            import time
            start_time = time.time()
            steps = []

            # gọi generator của thuật toán
            gen = partial_observable_search_with_logs(initial_belief, goal, observations)

            try:
                while True:
                    steps.append(next(gen))
            except StopIteration as e:
                result = e.value
                if isinstance(result, tuple):
                    status, expanded_count = result
                else:
                    status, expanded_count = result or "failure", len(steps)

            elapsed = time.time() - start_time

            def show_step(i=0):
                if not running:
                    return

                if i < len(steps):
                    entry = steps[i]
                    kind = entry[0]
                    payload = entry[1:]

                    veBanCo(cv3)

                    if kind == "expand":
                        # ("expand", belief, step)  hoặc  ("expand", belief)
                        if len(payload) >= 2:
                            belief, step_id = payload[0], payload[1]
                        else:
                            belief, step_id = payload[0], i + 1
                        text_log.insert("end", f"Bước {step_id}: 🔍 Mở rộng {len(belief)} trạng thái\n")
                        for s in belief:
                            text_log.insert("end", f"   {s}\n")

                    elif kind == "observe":
                        # Hỗ trợ 2 kiểu:
                        # 1) ("observe", obs, remain, step)
                        # 2) ("observe", (obs, remain, step))
                        if len(payload) == 3:
                            obs, remain, step_id = payload
                        elif len(payload) == 1 and isinstance(payload[0], tuple) and len(payload[0]) == 3:
                            obs, remain, step_id = payload[0]
                        else:
                            # fallback: cố gắng suy luận
                            obs = payload[0]
                            remain = payload[1] if len(payload) > 1 else None
                            step_id = payload[2] if len(payload) > 2 else (i + 1)

                        text_log.insert("end", f"Bước {step_id}: 👁 Quan sát {obs} → còn {remain} trạng thái\n")

                    elif kind == "goal":
                        # ("goal", belief_state, step)  hoặc  ("goal", belief_state)
                        if len(payload) >= 2:
                            belief_state, step_id = payload[0], payload[1]
                        else:
                            belief_state, step_id = payload[0], i + 1

                        text_log.insert("end", f"Bước {step_id}: ✅ ĐẠT ĐÍCH — belief chứa goal\n")
                        for s in belief_state:
                            text_log.insert("end", f"   {s}\n")
                        veXe(cv3, next(iter(belief_state)))

                    elif kind == "failure":
                        step_id = payload[0] if len(payload) >= 1 else (i + 1)
                        text_log.insert("end", f"Bước {step_id}: ❌ THẤT BẠI — không còn belief mới\n")

                    text_log.see("end")
                    root.after(300, show_step, i + 1)


                else:
                    text_log.insert("end", f"\n🔹 Số belief mở rộng: {expanded_count}\n")
                    text_log.insert("end", f"⏱️ Thời gian chạy: {elapsed:.4f} giây\n")
                    text_log.insert("end", f"=== Kết thúc {algo_name} ({status}) ===\n")
                    text_log.see("end")

            # chạy animation
            show_step()


        elif algo_name == "Backtracking":
            """-------------------------------------
                        Chạy Backtracking
            ------------------------------------"""
            step_iter = iter(btk(8))  # tạo iterator từ generator
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step():
                if not running:  # nếu bấm Dừng thì thoát
                    return
                try:
                    step = next(step_iter)  # lấy bước tiếp theo
                except StopIteration:
                    # hết các bước -> kết thúc
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")
                    return

                # unpack bước
                if isinstance(step, tuple) and len(step) == 4:
                    action, idx, (row, col), pos_pairs = step
                    state = pairs_to_state(pos_pairs, 8)  # chuyển (row,col) -> state dạng list
                    veBanCo(cv3)
                    veXe(cv3, state)

                    verb = "Đặt" if action == "place" else "Gỡ"
                    text_log.insert("end", f"{verb} quân {idx} tại ({row},{col}) -> {state}\n")

                text_log.see("end")
                root.after(300, show_step)  # delay 300ms rồi gọi lại

            show_step()

        elif algo_name == "Forward Checking":
            """-------------------------------------
                        Chạy Forward Checking
            ------------------------------------"""
            result = fwd(8)  # generator
            steps = list(result)  # lấy toàn bộ bước
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step(i=0):
                if not running:
                    return
                if i < len(steps):
                    step = steps[i]
                    if step[0] == "done":
                        _, assignment = step
                        state = pairs_to_state(list(assignment.values()), 8)
                        veBanCo(cv3)
                        veXe(cv3, state)
                        text_log.insert("end", f"Bước {i}: Hoàn thành -> {state}\n")
                    elif step[0] in ("place", "remove"):
                        action, idx, (row, col), assignment = step
                        state = pairs_to_state(list(assignment.values()), 8)
                        verb = "Đặt" if action == "place" else "Gỡ"
                        veBanCo(cv3)
                        veXe(cv3, state)
                        text_log.insert("end", f"Bước {i}: {verb} quân {idx} tại ({row},{col}) -> {state}\n")
                    text_log.see("end")
                    root.after(200, show_step, i + 1)
                else:
                    text_log.insert("end", f"Kết thúc {algo_name}\n")
                    text_log.see("end")

            show_step()

        elif algo_name == "AC3":
            """-------------------------------------
                        Chạy AC-3
            ------------------------------------"""
            steps = ac3_steps(8)  # KHÔNG ép list(), giữ generator
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            def show_step():
                if not running:
                    return
                try:
                    step = next(steps)  # lấy 1 bước
                    if step[0] == "done":
                        _, assignment = step
                        state = pairs_to_state(list(assignment.values()), 8)
                        veBanCo(cv3)
                        veXe(cv3, state)
                        text_log.insert("end", f"Hoàn thành -> {state}\n")
                    elif step[0] in ("place", "remove"):
                        action, idx, (row, col), assignment = step
                        state = pairs_to_state(list(assignment.values()), 8)
                        verb = "Đặt" if action == "place" else "Gỡ"
                        veBanCo(cv3)
                        veXe(cv3, state)
                        text_log.insert("end", f"{verb} quân {idx} tại ({row},{col}) -> {state}\n")
                    text_log.see("end")
                    root.after(200, show_step)  # gọi lại sau 200ms
                except StopIteration:
                    text_log.insert("end", f"=== Kết thúc {algo_name} ===\n")
                    text_log.see("end")

            show_step()

    btn_run.config(command=run_algorithm)

    """-------------------------------------
            Hàm chạy hết
    ------------------------------------"""

    def run_all():
        nonlocal running
        running = False  # dừng chạy từng bước
        algo_name = algo_var.get()

        if algo_name == "BFS":
            """-------------------------------------
                    Chạy BFS
            ------------------------------------"""
            start_time = time.time()

            steps = bfs(8, goal_state)  # BFS trả về list trạng thái
            end_time = time.time()
            elapsed = end_time - start_time
            times["BFS"] = elapsed
            veBanCo(cv3)
            veXe(cv3, steps[-1])  # vẽ trạng thái cuối

            text_log.insert("end", f"=== {algo_name} chạy hết ===\n")
            for i, s in enumerate(steps):
                text_log.insert("end", f"Bước {i}: {s}\n")
            text_log.insert("end", f"Kết quả cuối: {steps[-1]}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.see("end")

        elif algo_name == "DFS":
            """-------------------------------------
                    Chạy DFS
            ------------------------------------"""
            start_time = time.time()
            result = dfs(8, goal_state)  # dfs trả về generator
            steps = list(result)  # lấy toàn bộ các bước (visit, found, ...)

            end_time = time.time()
            elapsed = end_time - start_time
            times["DFS"] = elapsed
            # Vẽ trạng thái cuối cùng (state cuối cùng trong các bước)
            veBanCo(cv3)
            veXe(cv3, steps[-1][1])  # steps[-1] là (kind, state)

            # Ghi log ra text_log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, (kind, state) in enumerate(steps):
                text_log.insert("end", f"Bước {i}: {kind} -> {state}\n")

            text_log.insert("end", f"Kết quả cuối: {steps[-1][1]}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "DLS":
            """-------------------------------------
                    Chạy DLS
            ------------------------------------"""
            start_time = time.time()
            result = dls(goal_state, 8, 0)
            steps = list(result)  # lấy toàn bộ các bước (visit, found, ...)

            end_time = time.time()
            elapsed = end_time - start_time
            times["DLS"] = elapsed

            # Vẽ trạng thái cuối cùng (chỉ khi state là list)
            veBanCo(cv3)
            for kind, state in reversed(steps):
                if isinstance(state, list):  # chỉ chọn state là list
                    last_state = state
                    break

            if last_state is not None:
                veXe(cv3, last_state)
            # Ghi log ra text_log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, (kind, state) in enumerate(steps):
                text_log.insert("end", f"Bước {i}: {kind} -> {state}\n")

            text_log.insert("end", f"Kết quả cuối: {last_state}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")


        elif algo_name == "IDS":
            """-------------------------------------
                    Chạy IDS
            ------------------------------------"""
            start_time = time.time()
            result = ids(8, goal_state)  # dfs trả về generator
            steps = list(result)  # lấy toàn bộ các bước (visit, found, ...)

            end_time = time.time()
            elapsed = end_time - start_time
            times["IDS"] = elapsed
            # Vẽ trạng thái cuối cùng (state cuối cùng trong các bước)
            veBanCo(cv3)
            veXe(cv3, steps[-1][1])  # steps[-1] là (kind, state)

            # Ghi log ra text_log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, (kind, state) in enumerate(steps):
                text_log.insert("end", f"Bước {i}: {kind} -> {state}\n")

            text_log.insert("end", f"Kết quả cuối: {steps[-1][1]}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "UCS":
            """-------------------------------------
                        Chạy UCS 
            ------------------------------------"""
            goal = goal_state

            start_time = time.time()
            steps = list(ucs(8, goal))  # chạy hết UCS
            end_time = time.time()
            elapsed = end_time - start_time
            times["UCS"] = elapsed

            # Vẽ trạng thái cuối
            last_kind, last_state, last_cost, last_expand = steps[-1]
            veBanCo(cv3)
            veXe(cv3, last_state)

            # Log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, step in enumerate(steps):
                if step[0] == "expand":
                    _, state, cost, expand = step
                    text_log.insert("end", f"Bước {i}: Expand -> {state}, cost={cost}, expand={expand}\n")
                elif step[0] == "done":
                    _, state, cost, expand = step
                    text_log.insert("end", f"Hoàn thành -> {state}, cost={cost}, expand={expand}\n")

            text_log.insert("end", f"Số node mở rộng (expand): {expand_nodes.get('UCS', 0)}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")


        elif algo_name == "GS":
            """-------------------------------------
                        Chạy GS
            ------------------------------------"""
            goal = goal_state
            start_time = time.time()
            # truyền heuristic_fn=hes nếu bạn muốn rõ ràng (không bắt buộc vì đã default)
            steps = list(gs(8, goal_state=goal, heuristic_fn=hes, snapshot_top=20))
            elapsed = time.time() - start_time
            times["GS"] = elapsed

            if steps and steps[-1][0] == "done":
                _, last_state, last_cost, _ = steps[-1]
                veBanCo(cv3);
                veXe(cv3, last_state)

            text_log.insert("end", "=== Bắt đầu chạy GS ===\n")
            for i, (kind, state, cost, pq) in enumerate(steps):
                if kind == "frontier":
                    text_log.insert("end", f"Bước {i}: FRONTIER size={len(pq)}\n")
                    for j, (c, st) in enumerate(pq):
                        text_log.insert("end", f"   [{j}] cost={c} state={st}\n")
                else:
                    text_log.insert("end", f"Bước {i}: {kind} -> {state} | cost={cost}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", "=== GS kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "A*":
            """-------------------------------------
                        Chạy A*
            ------------------------------------"""
            start_time = time.time()
            result = A(8, goal_state)
            steps = list(result)  # lấy toàn bộ các bước (expand, done, ...)

            end_time = time.time()
            elapsed = end_time - start_time
            times["A*"] = elapsed

            # Vẽ trạng thái cuối cùng (state cuối cùng trong các bước)
            veBanCo(cv3)
            last_kind, last_state, last_cost = steps[-1]
            if isinstance(last_state, (list, tuple)):
                veXe(cv3, last_state)

            # Ghi log ra text_log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, (kind, state, cost) in enumerate(steps):
                text_log.insert("end", f"Bước {i}: {kind} -> {state}, g={cost}\n")

            text_log.insert("end", f"Kết quả cuối: {last_state}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "Hill Climbing":
            """-------------------------------------
                        Chạy Hill Climbing
            ------------------------------------"""
            goal = goal_state
            start = current_state

            start_time = time.time()
            result = hill_climbing(start, goal)  # result là list các bước (state, h)
            steps = list(result)  # Lưu toàn bộ các bước
            end_time = time.time()

            elapsed = end_time - start_time
            times["HC"] = elapsed

            # Vẽ trạng thái cuối cùng
            veBanCo(cv3)
            veXe(cv3, steps[-1][0])  # steps[-1] = (state, h)

            # In log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, (state, h) in enumerate(steps):
                text_log.insert("end", f"Bước {i}: h={h} -> {state}\n")

            text_log.insert("end", f"Kết quả cuối: {steps[-1][0]}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "Simulated Annealing":
            """-------------------------------------
                        Chạy Simulated Annealing
            ------------------------------------"""
            goal = goal_state
            start = current_state

            start_time = time.time()
            steps = list(sa(start, goal, T0=100, alpha=0.95, max_iter=1000))
            end_time = time.time()
            elapsed = end_time - start_time
            times["SA"] = elapsed

            # ---- helper: chuẩn hoá 1 step về (state, h, p) ----
            def _sa_unpack(rec):
                # chấp nhận (state,h), (state,h,p), (h,state), (h,state,p), thậm chí (h,state,p,T)
                a = rec[0]
                if isinstance(a, (list, tuple)):  # a là state
                    state = a
                    h = rec[1] if len(rec) > 1 else None
                    p = rec[2] if len(rec) > 2 else None
                else:  # rec[1] là state
                    state = rec[1]
                    h = rec[0]
                    p = rec[2] if len(rec) > 2 else None
                return state, h, p

            # ----------------------------------------------------

            # Vẽ trạng thái cuối
            last_state, _, _ = _sa_unpack(steps[-1])
            veBanCo(cv3)
            veXe(cv3, last_state)

            # Log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, rec in enumerate(steps):
                state, h, p = _sa_unpack(rec)
                if p is None:
                    text_log.insert("end", f"Bước {i}: h={h} -> {state}\n")
                else:
                    text_log.insert("end", f"Bước {i}: h={h}, p={p:.4f} -> {state}\n")
            text_log.insert("end", f"Kết quả cuối: {last_state}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "Beam":
            """-------------------------------------
                        Chạy Beam
            ------------------------------------"""
            goal = goal_state
            start_time = time.time()
            # truyền heuristic_fn=hes nếu bạn muốn rõ ràng (không bắt buộc vì đã default)
            steps = list(bs(8, goal_state=goal, heuristic_fn=hes, snapshot_top=20))
            elapsed = time.time() - start_time
            times["BS"] = elapsed

            if steps and steps[-1][0] == "done":
                _, last_state, last_cost, _ = steps[-1]
                veBanCo(cv3)
                veXe(cv3, last_state)

            text_log.insert("end", "=== Bắt đầu chạy BS ===\n")
            for i, (kind, state, cost, pq) in enumerate(steps):
                if kind == "frontier":
                    text_log.insert("end", f"Bước {i}: FRONTIER size={len(pq)}\n")
                    for j, (c, st) in enumerate(pq):
                        text_log.insert("end", f"   [{j}] cost={c} state={st}\n")
                else:
                    text_log.insert("end", f"Bước {i}: {kind} -> {state} | cost={cost}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", "=== BS kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "Genetic":
            """-------------------------------------
                        Chạy Genetic Algorithm
            ------------------------------------"""
            goal = goal_state
            n = len(goal)
            start = current_state  # thực ra GA không cần start, chỉ cần n và goal

            # bắt đầu đo thời gian
            start_time = time.time()
            steps = list(gene(n, goal, ca_the=10, ti_le_dot_bien=0.1, max_the_he=100))
            end_time = time.time()
            elapsed = end_time - start_time
            times["GENE"] = elapsed

            # ---- helper: chuẩn hoá 1 step về (state, h) ----
            def _gene_unpack(rec):
                # rec có dạng ("step"/"done", state, fit)
                state = rec[1]
                h = rec[2]  # fitness
                return state, h

            # ------------------------------------------------

            # Vẽ trạng thái cuối
            last_state, _ = _gene_unpack(steps[-1])
            veBanCo(cv3)
            veXe(cv3, last_state)

            # Log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, rec in enumerate(steps):
                state, h = _gene_unpack(rec)
                text_log.insert("end", f"Gen {i}: fitness={h} -> {state}\n")
            text_log.insert("end", f"Kết quả cuối: {last_state}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "And-Or":
            """-------------------------------------
                        Chạy AND–OR Graph Search
            ------------------------------------"""
            goal = tuple(goal_state)  # đảm bảo là tuple hàng→cột
            initial = tuple([-1] * 8)

            start_time = time.time()
            steps = []
            gen = and_or_graph_search_with_logs(initial_state=initial, goal=goal, snapshot_top=20)
            # Thu hết log
            try:
                while True:
                    steps.append(next(gen))
            except StopIteration as e:
                status = e.value  # "success" | "failure"
            elapsed = time.time() - start_time
            times["AND-OR"] = elapsed

            # ----- VẼ KẾT QUẢ -----
            # Ưu tiên vẽ khi có bước DONE
            done_steps = [s for s in steps if s[0].lower() == "done"]
            final_state = None
            if done_steps:
                _, final_state, _, _ = done_steps[-1]
                final_state = tuple(final_state)
            else:
                # Fallback: lấy plan rồi áp lên initial để tạo final_state
                plan = AND_OR_Graph_Search(initial_state=initial, goal=goal)
                if plan is not None:
                    s = list(initial)
                    for (row, col) in plan:
                        s[row] = col
                    final_state = tuple(s)

            if final_state is not None:
                veBanCo(cv3)
                veXe(cv3, final_state)  # ❗ state là hàng→cột, vẽ trực tiếp
                cv3.update()

            # ----- LOG RA GUI -----
            text_log.insert("end", "=== Bắt đầu chạy AND–OR Graph Search ===\n")
            for i, (kind, state, cost, pq) in enumerate(steps):
                KIND = kind.upper()
                if KIND == "FRONTIER":
                    text_log.insert("end", f"Bước {i}: FRONTIER size={len(pq)}\n")
                    for j, (c, st) in enumerate(pq):
                        text_log.insert("end", f"   [{j}] cost={c} state={st}\n")
                elif KIND == "DONE":
                    text_log.insert("end", f"Bước {i}: ✅ DONE -> {state}\n")
                elif KIND == "FAILURE":
                    text_log.insert("end", f"Bước {i}: ❌ FAILURE -> {state}\n")
                elif KIND == "SUCCESS":
                    text_log.insert("end", f"Bước {i}: ✅ SUCCESS -> {state}\n")
                elif KIND == "VISIT":
                    text_log.insert("end", f"Bước {i}: VISIT -> {state}\n")
                else:
                    text_log.insert("end", f"Bước {i}: {KIND} -> {state}\n")

            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", "=== AND–OR Graph Search kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "Belief Search":
            """-------------------------------------
                        Chạy Belief Space Search (Run All)
            ------------------------------------"""
            # --- Khởi tạo ---
            goal = tuple(goal_state)
            N = len(goal_state)  # 🔧 Thêm dòng này để tránh lỗi 'UnboundLocalError'
            initial_belief = {tuple([-1] * N)}  # belief ban đầu: chưa biết vị trí nào

            text_log.insert("end", "=== Bắt đầu chạy Belief Search ===\n")

            start_time = time.time()
            steps = []

            # --- Gọi hàm tìm kiếm ---
            gen = belief_search_with_logs(initial_belief, goal)

            try:
                while True:
                    steps.append(next(gen))
            except StopIteration as e:
                ret = e.value
                if isinstance(ret, tuple):
                    status, expanded_count = ret
                else:
                    status = ret or "failure"
                    expanded_count = 0

            elapsed = time.time() - start_time
            times["BELIEF"] = elapsed

            # --- Vẽ bàn cờ kết quả ---
            final_state = None
            if steps and steps[-1][0].lower() == "goal":
                _, belief_state, _ = steps[-1]
                if belief_state:
                    final_state = next(iter(belief_state))

            if final_state:
                veBanCo(cv3)
                veXe(cv3, final_state)
                cv3.update()

            # --- In log ra text_log ---
            for i, step in enumerate(steps):
                kind = step[0].lower()

                # Hỗ trợ cả 2 format yield khác nhau
                if len(step) == 3:
                    _, belief_state, step_id = step
                else:
                    belief_state = step[1]
                    step_id = i + 1

                if kind == "expand":
                    text_log.insert("end", f"Bước {step_id}: 🔍 Mở rộng {len(belief_state)} trạng thái\n")
                    for s in belief_state:
                        text_log.insert("end", f"   {s}\n")
                elif kind == "goal":
                    text_log.insert("end", f"Bước {step_id}: ✅ ĐẠT ĐÍCH — belief chứa goal\n")
                    for s in belief_state:
                        text_log.insert("end", f"   {s}\n")
                elif kind == "failure":
                    text_log.insert("end", f"Bước {step_id}: ❌ THẤT BẠI — không còn belief mới\n")

            # --- Tổng kết ---
            text_log.insert("end", f"\n🔹 Số belief mở rộng: {expanded_count}\n")
            text_log.insert("end", f"⏱️ Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== Kết thúc Belief Search ({status}) ===\n")
            text_log.see("end")


        elif algo_name == "Partial Observable":
            """-------------------------------------
                        Chạy Partial Observable Search
            ------------------------------------"""
            goal = tuple(goal_state)
            N = len(goal)

            # belief ban đầu: chưa biết vị trí quân nào
            initial_belief = {tuple([-1] * N)}

            # ví dụ chuỗi quan sát (bạn có thể thay đổi tuỳ theo giao diện nhập)
            observations = [
                {0: 0},  # thấy quân ở hàng 0, cột 0
                {1: 1, 2: 2},  # thấy thêm quân hàng 1,2
                {3: 3, 4: 4},  # ...
                {5: 5, 6: 6, 7: 7}
            ]

            text_log.insert("end", "=== Bắt đầu chạy Partial Observable Search ===\n")

            start_time = time.time()
            steps = []
            gen = partial_observable_search_with_logs(initial_belief, goal, observations)

            try:
                while True:
                    steps.append(next(gen))
            except StopIteration as e:
                ret = e.value
                if isinstance(ret, tuple):
                    status, expanded_count = ret
                else:
                    status = ret
                    expanded_count = 0

            elapsed = time.time() - start_time

            # --- Vẽ bàn cờ kết quả ---
            final_state = None
            if steps and steps[-1][0].lower() == "goal":
                _, belief_state, _ = steps[-1]
                if belief_state:
                    final_state = next(iter(belief_state))

            if final_state:
                veBanCo(cv3)
                veXe(cv3, final_state)
                cv3.update()

            # --- In log ---
            for (kind, data, *rest) in steps:
                if kind == "expand":
                    belief, step_id = data, rest[0]
                    text_log.insert("end", f"Bước {step_id}: 🔍 Mở rộng {len(belief)} trạng thái\n")
                elif kind == "observe":
                    obs, remain, step_id = data, rest[0], rest[1]
                    text_log.insert("end", f"Bước {step_id}: 👁 Quan sát {obs} → còn {remain} trạng thái\n")
                elif kind == "goal":
                    belief_state, step_id = data, rest[0]
                    text_log.insert("end", f"Bước {step_id}: ✅ ĐẠT ĐÍCH — belief chứa goal\n")
                    for s in belief_state:
                        text_log.insert("end", f"   {s}\n")
                elif kind == "failure":
                    step_id = rest[0]
                    text_log.insert("end", f"Bước {step_id}: ❌ THẤT BẠI — không còn belief mới\n")

            text_log.insert("end", f"\n🔹 Số belief mở rộng: {expanded_count}\n")
            text_log.insert("end", f"⏱️ Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== Kết thúc Partial Observable Search ({status}) ===\n")
            text_log.see("end")
            times["PARTIAL"] = elapsed


        elif algo_name == "Backtracking":
            """-------------------------------------
                        Chạy Backtracking
            ------------------------------------"""
            start_time = time.time()
            steps = list(btk(8))  # chạy toàn bộ generator
            end_time = time.time()
            elapsed = end_time - start_time
            times["BTK"] = elapsed

            # Vẽ trạng thái cuối cùng
            last_step = steps[-1]
            if isinstance(last_step, tuple) and last_step[0] == "done":
                _, final_pos = last_step
                last_state = pairs_to_state(final_pos, 8)
            else:
                # nếu chưa có "done", lấy state từ bước cuối
                if len(last_step) == 4:
                    _, _, _, pos_pairs = last_step
                    last_state = pairs_to_state(pos_pairs, 8)
                else:
                    last_state = []

            veBanCo(cv3)
            veXe(cv3, last_state)

            # Log
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, step in enumerate(steps):
                if isinstance(step, tuple):
                    if len(step) == 2 and step[0] == "done":
                        # bước cuối đã đủ n quân
                        _, pos_pairs = step
                        state = pairs_to_state(pos_pairs, 8)
                        text_log.insert("end", f"Bước {i}: Hoàn thành -> {state}\n")
                    elif len(step) == 4:
                        action, idx, (row, col), pos_pairs = step
                        state = pairs_to_state(pos_pairs, 8)
                        verb = "Đặt" if action == "place" else "Gỡ"
                        text_log.insert("end", f"Bước {i}: {verb} quân {idx} tại ({row},{col}) -> {state}\n")

            text_log.insert("end", f"Kết quả cuối: {last_state}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "Forward Checking":
            """-------------------------------------
                        Chạy Forward Checking
            ------------------------------------"""
            start_time = time.time()
            steps = list(fwd(8))  # chạy toàn bộ generator
            end_time = time.time()
            elapsed = end_time - start_time
            times["FWD"] = elapsed

            # Xác định trạng thái cuối
            last_step = steps[-1]
            if isinstance(last_step, tuple) and last_step[0] == "done":
                _, assignment = last_step
                last_state = pairs_to_state(list(assignment.values()), 8)
            else:
                # nếu chưa có "done", lấy state từ bước cuối
                if len(last_step) == 4:
                    _, _, _, assignment = last_step
                    last_state = pairs_to_state(list(assignment.values()), 8)
                else:
                    last_state = []

            veBanCo(cv3)
            veXe(cv3, last_state)

            # Log toàn bộ quá trình
            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")
            for i, step in enumerate(steps):
                if isinstance(step, tuple):
                    if len(step) == 2 and step[0] == "done":
                        _, assignment = step
                        state = pairs_to_state(list(assignment.values()), 8)
                        text_log.insert("end", f"Bước {i}: Hoàn thành -> {state}\n")
                    elif len(step) == 4:
                        action, idx, (row, col), assignment = step
                        state = pairs_to_state(list(assignment.values()), 8)
                        verb = "Đặt" if action == "place" else "Gỡ"
                        text_log.insert("end", f"Bước {i}: {verb} quân {idx} tại ({row},{col}) -> {state}\n")

            text_log.insert("end", f"Kết quả cuối: {last_state}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

        elif algo_name == "AC3":
            """-------------------------------------
                        Chạy AC3 (run all)
            ------------------------------------"""
            start_time = time.time()
            steps = list(ac3_steps(8))  # chạy hết generator
            end_time = time.time()
            elapsed = end_time - start_time
            times["AC3"] = elapsed

            text_log.insert("end", f"=== Bắt đầu chạy {algo_name} ===\n")

            for i, step in enumerate(steps):
                if step[0] == "place":
                    _, idx, (row, col), assignment = step
                    state = pairs_to_state(list(assignment.values()), 8)  # ✅ chuyển đổi
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Bước {i}: Đặt quân {idx} tại ({row},{col}) -> {state}\n")

                elif step[0] == "remove":
                    _, idx, (row, col), assignment = step
                    state = pairs_to_state(list(assignment.values()), 8)  # ✅ chuyển đổi
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Bước {i}: Gỡ quân {idx} tại ({row},{col}) -> {state}\n")

                elif step[0] == "done":
                    _, assignment = step
                    state = pairs_to_state(list(assignment.values()), 8)  # ✅ chuyển đổi
                    veBanCo(cv3)
                    veXe(cv3, state)
                    text_log.insert("end", f"Hoàn thành -> {state}\n")

                text_log.see("end")

            text_log.insert("end", f"Số node mở rộng (expand): {expand_nodes.get('AC3', 0)}\n")
            text_log.insert("end", f"Thời gian chạy: {elapsed:.4f} giây\n")
            text_log.insert("end", f"=== {algo_name} kết thúc ===\n")
            text_log.see("end")

    btn_run_all.config(command=run_all)

    root.mainloop()


if __name__ == "__main__":
    main()