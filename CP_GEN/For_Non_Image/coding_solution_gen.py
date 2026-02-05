import sys

# It is a good practice to set a higher recursion limit for deep recursion in digit DP.
# The maximum recursion depth is bounded by K, which is at most 25.
# A default limit is usually fine, but a higher one is safer.
sys.setrecursionlimit(2000)

MOD = 998244353

# Memoization for total strong counts for a given K.
memo_total_strong_counts = {}

def get_total_strong_counts(K):
    """
    Calculates the number of K-strong numbers for each length from 1 to K-1.
    """
    if K in memo_total_strong_counts:
        return memo_total_strong_counts[K]

    # dp_from_nonzero[p] will store a map from (remainder, mask) to count
    # for K-strong numbers of length p.
    dp_from_nonzero = [{} for _ in range(K)]
    
    # Base case: numbers of length 1.
    for d1 in range(1, 10):
        rem = d1 % K
        if rem != 0:  # S_1 must not be S_0 (mod K)
            mask = 1 | (1 << rem)
            state = (rem, mask)
            dp_from_nonzero[1][state] = (dp_from_nonzero[1].get(state, 0) + 1)

    # Build up for lengths p = 2, ..., K-1
    for p in range(1, K - 1):
        if not dp_from_nonzero[p]:
            break
        for state, count in dp_from_nonzero[p].items():
            if count == 0: continue
            rem, mask = state
            for d in range(10):
                new_rem = (rem + d) % K
                if not ((mask >> new_rem) & 1):
                    new_mask = mask | (1 << new_rem)
                    new_state = (new_rem, new_mask)
                    dp_from_nonzero[p + 1][new_state] = (dp_from_nonzero[p + 1].get(new_state, 0) + count) % MOD
    
    counts_by_len = [0] * K
    for p in range(1, K):
        counts_by_len[p] = sum(dp_from_nonzero[p].values()) % MOD
    
    memo_total_strong_counts[K] = counts_by_len
    return counts_by_len

# Memoization for the main digit DP calculation.
memo_calc = {}

def calc(S, K):
    """
    Counts K-strong numbers in [1, S] for a given string S.
    Assumes len(S) < K.
    """
    state_key = (S, K)
    if state_key in memo_calc:
        return memo_calc[state_key]

    N = len(S)
    memo_dp = {}

    def dp(pos, current_sum, mask, is_tight, is_leading):
        if pos == N:
            return 1 if not is_leading else 0
        
        state = (pos, current_sum, mask, is_tight, is_leading)
        if state in memo_dp:
            return memo_dp[state]

        res = 0
        limit = int(S[pos]) if is_tight else 9
        
        for digit in range(limit + 1):
            new_tight = is_tight and (digit == limit)
            
            if is_leading and digit == 0:
                res = (res + dp(pos + 1, 0, 1, new_tight, True)) % MOD
            else:
                _sum = current_sum if not is_leading else 0
                _mask = mask if not is_leading else 1
                
                new_sum = (_sum + digit) % K
                if (_mask >> new_sum) & 1:
                    continue
                
                new_mask = _mask | (1 << new_sum)
                res = (res + dp(pos + 1, new_sum, new_mask, new_tight, False)) % MOD
        
        memo_dp[state] = res
        return res

    # The DP counts strong numbers in [0, S]. We subtract 1 for the number 0.
    # Actually, the base case `1 if not is_leading else 0` handles this.
    ans = dp(0, 0, 1, True, True)
    memo_calc[state_key] = ans
    return ans

def count_strong(S, K):
    """
    Counts K-strong numbers up to S.
    """
    if S == "0":
        return 0
    N = len(S)
    if N >= K:
        counts_by_len = get_total_strong_counts(K)
        total_strong = sum(counts_by_len) % MOD
        return total_strong
    else:
        return calc(S, K)

def solve():
    L_str, R_str, K = input().split()
    K = int(K)

    L_minus_1_str = str(int(L_str) - 1)
    
    R_val = int(R_str) % MOD
    L_minus_1_val = int(L_minus_1_str) % MOD

    strong_R = count_strong(R_str, K)
    weak_R = (R_val - strong_R + MOD) % MOD
    
    strong_L_minus_1 = count_strong(L_minus_1_str, K)
    weak_L_minus_1 = (L_minus_1_val - strong_L_minus_1 + MOD) % MOD
    
    ans = (weak_R - weak_L_minus_1 + MOD) % MOD
    return ans

def main():
    T = int(input())
    for i in range(1, T + 1):
        result = solve()
        print(f"Case #{i}: {result}")

if __name__ == "__main__":
    main()