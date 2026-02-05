import sys
from bisect import bisect_left, bisect_right

def solve():
    """
    Solves a single test case.
    """
    try:
        line = sys.stdin.readline().strip()
        if not line: return None
        N, Q, L = map(int, line.split())
        X_orig = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return None

    # Store robots as (position, original_index) and sort by position
    robots_sorted_by_pos = sorted([(X_orig[i], i + 1) for i in range(N)])
    
    positions_sorted = [r[0] for r in robots_sorted_by_pos]
    indices_sorted = [r[1] for r in robots_sorted_by_pos]
    
    walls = [1, L]
    total_ans = 0

    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        op = query[0]

        if op == 1:
            x = query[1]
            idx = bisect_left(walls, x)
            # Insert wall only if it doesn't exist
            if idx == len(walls) or walls[idx] != x:
                walls.insert(idx, x)
        else:
            r, s = query[1], query[2]
            r_pos = X_orig[r - 1]
            
            wall_idx = bisect_right(walls, r_pos)
            w_L = walls[wall_idx - 1]
            w_R = walls[wall_idx]
            
            D = w_R - w_L
            if D <= 0:
                continue

            # Find all robots within the same interval (w_L, w_R)
            start_idx = bisect_right(positions_sorted, w_L)
            end_idx = bisect_left(positions_sorted, w_R)

            # Filter candidates: j > r and first collision must be within s seconds
            candidates = []
            # Condition for C_rj <= s, rearranged to avoid floats and use integers
            # (X_r + X_j - 2*w_L)/2 <= s  =>  X_r + X_j - 2*w_L <= 2*s
            pos_limit = 2 * s - r_pos + 2 * w_L
            for i in range(start_idx, end_idx):
                if indices_sorted[i] > r and positions_sorted[i] <= pos_limit:
                    candidates.append((indices_sorted[i], positions_sorted[i]))
            
            if not candidates:
                continue

            best_j_cat1, max_rem_cat1 = 0, -1
            best_j_cat2, max_rem_cat2 = 0, -1
            
            # Target for modulo comparison to determine category
            s_rem_target = 2 * (s % D)

            for j, j_pos in candidates:
                # We need to maximize C_rj mod D. This is equivalent to maximizing
                # (X_r + X_j - 2*w_L) mod 2D.
                rem_val = (r_pos + j_pos - 2 * w_L) % (2 * D)

                if rem_val <= s_rem_target: # Category 1
                    if rem_val > max_rem_cat1:
                        max_rem_cat1 = rem_val
                        best_j_cat1 = j
                    elif rem_val == max_rem_cat1:
                        best_j_cat1 = max(best_j_cat1, j)
                else: # Category 2
                    if rem_val > max_rem_cat2:
                        max_rem_cat2 = rem_val
                        best_j_cat2 = j
                    elif rem_val == max_rem_cat2:
                        best_j_cat2 = max(best_j_cat2, j)

            if best_j_cat1 != 0:
                total_ans += best_j_cat1
            elif best_j_cat2 != 0:
                total_ans += best_j_cat2
                
    return total_ans


def main():
    T_str = sys.stdin.readline()
    if not T_str: return
    T = int(T_str)
    for i in range(1, T + 1):
        ans = solve()
        if ans is None: break
        print(f"Case #{i}: {ans}")

if __name__ == "__main__":
    main()