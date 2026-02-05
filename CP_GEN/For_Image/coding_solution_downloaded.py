import sys
import collections

# A high recursion limit is set for building the segment trees recursively,
# although the maximum depth is logarithmic and well within standard limits.
sys.setrecursionlimit(2 * 10**6)

def solve():
    """
    Solves a single test case using a 0-1 BFS on a specially constructed graph.
    """
    try:
        # Fast I/O
        line = sys.stdin.readline().split()
        if not line: return None
        N, K, M = map(int, line)
    except (IOError, ValueError):
        return None

    routes = []
    sum_L = 0
    for _ in range(M):
        line = list(map(int, sys.stdin.readline().split()))
        route = line[1:]
        routes.append(route)
        sum_L += len(route)

    # Node Allocation:
    # 0 to N-1: Court nodes
    # N to N+sum_L-1: Boarding nodes
    # N+sum_L onwards: Deboarding segment tree nodes
    # A segment tree on L items has at most 2L-1 nodes.
    # Total nodes are N (courts) + sum_L (boarding) + 2*sum_L (ST nodes).
    num_nodes = N + 3 * sum_L
    adj = [[] for _ in range(num_nodes)]
    
    node_idx_counter = N

    for route in routes:
        L = len(route)
        
        boarding_nodes_start = node_idx_counter
        node_idx_counter += L
        
        st_nodes_start = node_idx_counter
        st_leaf_map = [-1] * L
        
        st_node_ptr = st_nodes_start
        
        # Recursively build the segment tree for deboarding stops
        def build_st(l, r):
            nonlocal st_node_ptr
            node_idx = st_node_ptr
            st_node_ptr += 1
            
            if l == r:
                st_leaf_map[l] = node_idx
                return node_idx
            
            mid = (l + r) // 2
            left_child = build_st(l, mid)
            right_child = build_st(mid + 1, r)
            
            # Parent to child edges have 0 cost
            adj[node_idx].append((left_child, 0))
            adj[node_idx].append((right_child, 0))
            return node_idx

        st_root = build_st(0, L - 1)
        node_idx_counter = st_node_ptr

        # Query the segment tree to find nodes covering a range
        def query_st(node_idx, l, r, ql, qr, result):
            if ql > qr or l > qr or r < ql:
                return
            if ql <= l and r <= qr:
                result.append(node_idx)
                return
            
            mid = (l + r) // 2
            if adj[node_idx]:
                left_child, right_child = adj[node_idx][0][0], adj[node_idx][1][0]
                query_st(left_child, l, mid, ql, qr, result)
                query_st(right_child, mid + 1, r, ql, qr, result)

        # Connect the graph components for this route
        for j in range(L):
            court_node = route[j] - 1
            boarding_node = boarding_nodes_start + j
            deboarding_leaf_node = st_leaf_map[j]

            # Edge for boarding (cost 1)
            adj[court_node].append((boarding_node, 1))
            # Edge for deboarding (cost 0)
            adj[deboarding_leaf_node].append((court_node, 0))

            # Edges for riding (cost 0)
            q_start = j + 1
            q_end = min(L - 1, j + K)
            
            if q_start <= q_end:
                target_st_nodes = []
                query_st(st_root, 0, L - 1, q_start, q_end, target_st_nodes)
                for st_node in target_st_nodes:
                    adj[boarding_node].append((st_node, 0))

    # 0-1 BFS (Dijkstra with a deque) to find shortest paths
    dist = [float('inf')] * node_idx_counter
    start_node = 0  # Court 1 is node 0
    dist[start_node] = 0
    dq = collections.deque([(0, start_node)]) # (distance, node)
    
    while dq:
        d, u = dq.popleft()
        
        if d > dist[u]:
            continue
            
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft((dist[v], v))
                else:
                    dq.append((dist[v], v))
    
    # Calculate the final sum
    total_sum = 0
    for i in range(N):
        court_dist = dist[i]
        if court_dist == float('inf'):
            total_sum -= (i + 1) # D(i) = -1 for unreachable courts
        else:
            total_sum += court_dist * (i + 1)
            
    return total_sum

def main():
    try:
        num_test_cases_str = sys.stdin.readline()
        if not num_test_cases_str: return
        T = int(num_test_cases_str)
        for t in range(1, T + 1):
            result = solve()
            if result is None: break
            print(f"Case #{t}: {result}")
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()