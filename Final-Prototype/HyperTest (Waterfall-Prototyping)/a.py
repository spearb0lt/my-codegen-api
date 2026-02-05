import sys
import bisect

def solve():
    """
    Solves a single test case.
    """
    N, Q, L = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))

    # Store robots as (initial_pos, index)
    # This helps in quickly finding robots within a wall segment
    sorted_robots = sorted([(X[i], i + 1) for i in range(N)])
    sorted_robots_X = [x for x, idx in sorted_robots]

    # Active walls start with boundaries 1 and L
    active_walls = [1, L]

    total_ans = 0
    for _ in range(Q):
        line = sys.stdin.readline().split()
        q_type = int(line[0])

        if q_type == 1:
            x = int(line[1])
            # Add a new wall, keeping the list sorted
            bisect.insort(active_walls, x)
        else:
            r, s = int(line[1]), int(line[2])
            X_r = X[r - 1]

            # Find the segment (w_l, w_r) for robot r
            wall_idx_r = bisect.bisect_right(active_walls, X_r)
            w_l = active_walls[wall_idx_r - 1]
            w_r = active_walls[wall_idx_r]

            W = w_r - w_l
            if W == 0:
                continue

            # Find robots within the same segment (w_l, w_r) using the pre-sorted list
            seg_start_idx = bisect.bisect_right(sorted_robots_X, w_l)
            seg_end_idx = bisect.bisect_left(sorted_robots_X, w_r)

            best_j = 0
            best_rem = -1  # Remainder can be 0, so initialize with -1

            M = 2 * W

            # Linear scan through all robots in the segment
            for i in range(seg_start_idx, seg_end_idx):
                X_j, j = sorted_robots[i]

                # Robot r pays off robot j, so j must be greater than r
                if j <= r:
                    continue

                # The latest collision time t <= s is maximized when the following remainder is maximized.
                # K is a constant part for this query and a specific robot r.
                K = X_r - 2 * w_l - 2 * s
                rem = (K + X_j) % M

                # Update best candidate based on remainder (primary key) and index (secondary key)
                if rem > best_rem:
                    best_rem = rem
                    best_j = j
                elif rem == best_rem:
                    if j > best_j:
                        best_j = j

            # If a candidate was found, verify that a collision actually occurs at or before time s.
            if best_j > 0:
                X_best_j = X[best_j - 1]

                # Calculate the time of the first collision between r and best_j
                # num = X_r + X_best_j - 2 * w_l must be > 0 for t_first > 0.
                # Since X_r > w_l and X_best_j > w_l, this holds.
                num = X_r + X_best_j - 2 * w_l

                # First collision corresponds to largest integer k such that t_k > 0
                # t_k = (num - k*M)/2 > 0 => num > k*M => k < num/M
                k_for_t_first = (num - 1) // M if M > 0 else 0
                t_first = (num - k_for_t_first * M) / 2.0

                if t_first <= s:
                    total_ans += best_j

    return total_ans

def main():
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for i in range(1, T + 1):
            ans = solve()
            print(f"Case #{i}: {ans}")
    except (IOError, IndexError):
        return

if __name__ == "__main__":
    main()