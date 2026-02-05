import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # 1. Calculate prefix sums S_j
    # S[0] is a dummy, S[j] corresponds to item j
    S = [0] * (N + 1)
    current_sum = 0
    for i in range(N):
        current_sum += A[i]
        S[i + 1] = current_sum

    # 2. Iterative DP to find min cost C
    # dp[c] = max amazingness for cost c
    # parent[c] = (j, prev_c) ->
    #   dp[c] was achieved from dp[prev_c] by adding item j
    dp = {0: 0}
    parent = {}
    
    C = 0
    while True:
        C += 1
        
        best_val = -1
        best_j = -1
        prev_c = -1
        
        for j in range(1, N + 1):
            if C >= j:
                # We can use item j, which has cost j and value S[j]
                # Check if this item j improves the value for cost C
                if C - j in dp:
                    current_val = dp[C - j] + S[j]
                    if current_val > best_val:
                        best_val = current_val
                        best_j = j
                        prev_c = C - j

        if best_val == -1:
            # Should not happen if M > 0 and A_i >= 1
            # But as a guard:
            dp[C] = 0
        else:
            dp[C] = best_val
            parent[C] = (best_j, prev_c)

        if best_val >= M:
            # Found the minimum cost C
            break
            
        # Optimization: If C exceeds a reasonable bound, stop.
        # Given sample 6 (N=8, C=20), a bound like N*N might be needed.
        # But we trust the problem constraints mean C_opt is found quickly.
        # A bound of 2*N*N is very safe but might be slow.
        # Let's try 200000, which covers N=500 case (250k) reasonably
        # and N=8 case (20).
        # This bound is heuristic, but necessary for TLE.
        # The true bound is likely smaller.
        if C > 400000: # Heuristic bound
            break
            
    # 3. Reconstruct d_j counts
    d_counts = [0] * (N + 1)
    curr_c = C
    while curr_c > 0:
        if curr_c not in parent:
             # This should not be reached if a solution exists
             break
        j, prev_c = parent[curr_c]
        d_counts[j] += 1
        curr_c = prev_c

    # 4. Convert d_j counts to x_i heights
    x = [0] * N
    # We use the optimized x[k] = d[k] + x[k+1]
    # x_N = d_N
    # x_{N-1} = d_{N-1} + x_N = d_{N-1} + d_N
    # ...
    # This is a suffix sum of d_counts
    suffix_sum_d = 0
    for k in range(N, 0, -1): # k from N down to 1
        suffix_sum_d += d_counts[k]
        x[k - 1] = suffix_sum_d # x[0]...x[N-1]
        
    return f"{C}\n{' '.join(map(str, x))}"

# Read number of test cases
T = int(sys.stdin.readline())
outputs = []
for i in range(1, T + 1):
    outputs.append(f"Case #{i}: {solve()}")

print('\n'.join(outputs))