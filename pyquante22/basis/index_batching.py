from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

Idx2 = Tuple[int, int]

def batch_pairs_share_i(pairs: Sequence[Idx2], batch_size: int) -> List[List[Idx2]]:
    """
    Group (i,j) pairs into batches of size <= batch_size, trying to maximize
    how many pairs in a batch share the same first index i.

    Guaranteed input constraint: i <= j (not strictly needed here, but assumed true).
    Strategy:
      1) bucket by i
      2) greedily fill each batch with as many from one i as possible
      3) if a bucket is bigger than batch_size, it becomes multiple pure-i batches
      4) leftover space in a batch is filled from other i's (largest remaining first)
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    buckets: dict[int, List[Idx2]] = defaultdict(list)
    for i, j in pairs:
        # optional sanity check:
        # if i > j: raise ValueError(f"violates i<=j: {(i,j)}")
        buckets[i].append((i, j))

    # (Optional) keep js sorted for determinism
    for i in buckets:
        buckets[i].sort(key=lambda t: t[1])

    # remaining counts per i
    remaining = {i: len(lst) for i, lst in buckets.items()}
    active_is = sorted(remaining, key=lambda i: remaining[i], reverse=True)

    batches: List[List[Idx2]] = []
    ptr = {i: 0 for i in buckets}  # read pointer within each bucket

    def take_from_i(i: int, k: int) -> List[Idx2]:
        p = ptr[i]
        out = buckets[i][p:p + k]
        ptr[i] = p + k
        remaining[i] -= k
        return out

    # Greedy: start a batch with the most populous i, pack it, then optionally top off.
    while active_is:
        # refresh to skip empties
        active_is = [i for i in active_is if remaining[i] > 0]
        if not active_is:
            break

        # pick i with most remaining
        active_is.sort(key=lambda i: remaining[i], reverse=True)
        i0 = active_is[0]

        batch: List[Idx2] = []
        # pack as much as possible from i0
        k0 = min(batch_size, remaining[i0])
        batch.extend(take_from_i(i0, k0))

        # top off with other i's, preferring larger remaining buckets
        space = batch_size - len(batch)
        if space:
            # iterate other i's in descending remaining
            for i in active_is[1:]:
                if space == 0:
                    break
                if remaining[i] == 0:
                    continue
                k = min(space, remaining[i])
                batch.extend(take_from_i(i, k))
                space -= k

        batches.append(batch)

    return batches


# --- example ---
if __name__ == "__main__":
    pairs = [(0,0),(0,1),(0,3),(1,1),(1,2),(2,2),(2,5),(2,7),(3,3)]
    out = batch_pairs_share_i(pairs, batch_size=4)
    for b in out:
        print(b)


from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

Idx2 = Tuple[int, int]
Slice2D = Tuple[slice, slice]

def make_2d_slices_fixed_area(
    pairs: Iterable[Idx2],
    batch_size: int,
    *,
    i_block: int = 1,
) -> List[Slice2D]:
    """
    Build 2D rectangular slices (slice_i, slice_j) whose *area* is <= batch_size.
    It's OK if a slice covers (i,j) locations that are NOT in `pairs`.

    Default is i_block=1, so each slice is 1 x batch_size in (i,j) space.
    If you set i_block>1, slices become i_block x (batch_size//i_block).

    Notes:
      - Input guarantees i <= j, but we don't rely on it.
      - Slices are axis-aligned rectangles in the (i,j) index grid.
      - Goal: cover all provided pairs by a small set of such rectangles (heuristic).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if i_block <= 0:
        raise ValueError("i_block must be positive")

    j_block = max(1, batch_size // i_block)   # width
    i_block = min(i_block, batch_size)        # height (area <= batch_size)

    # bucket js by i
    by_i: Dict[int, List[int]] = defaultdict(list)
    for i, j in pairs:
        by_i[i].append(j)

    # sort for determinism
    for i in by_i:
        by_i[i].sort()

    slices: List[Slice2D] = []

    # Process i in blocks of consecutive i's.
    # Within each i-block, cover the union of js with j-windows of width j_block.
    i_vals = sorted(by_i.keys())
    k = 0
    while k < len(i_vals):
        i0 = i_vals[k]
        # choose consecutive i's for this block
        block_is = [i0]
        k += 1
        while k < len(i_vals) and len(block_is) < i_block and i_vals[k] == block_is[-1] + 1:
            block_is.append(i_vals[k])
            k += 1

        # collect j range for this i-block
        j_min = min(by_i[i][0] for i in block_is)
        j_max = max(by_i[i][-1] for i in block_is)

        # align j windows to multiples of j_block (helps reuse/cache)
        start = (j_min // j_block) * j_block
        end = j_max

        for j0 in range(start, end + 1, j_block):
            slices.append((slice(block_is[0], block_is[-1] + 1), slice(j0, j0 + j_block)))

    return slices


# --- example ---
if __name__ == "__main__":
    pairs = [(0,0),(0,1),(0,7),(1,1),(1,2),(3,3),(3,9)]
    sls = make_2d_slices_fixed_area(pairs, batch_size=4, i_block=1)  # 1x4 tiles
    for si, sj in sls:
        print(si, sj)

    # Larger 2D tiles (e.g., 2x2 = area 4)
    sls2 = make_2d_slices_fixed_area(pairs, batch_size=4, i_block=2)
    for si, sj in sls2:
        print("2x2 tile:", si, sj)


from collections import defaultdict
from typing import Iterable, List, Tuple

Idx2 = Tuple[int, int]

def batch_pairs_greedy(pairs: Iterable[Idx2], batch_size: int) -> List[List[Idx2]]:
    """
    Greedy batching to maximize shared first-index i inside each batch.

    1) Bucket by i
    2) Repeatedly take up to batch_size from the currently-largest bucket
    3) If there's slack, top off from the next-largest buckets

    Returns: list of batches, each a list[(i,j)] with len <= batch_size
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    buckets = defaultdict(list)
    for i, j in pairs:
        buckets[i].append((i, j))

    # deterministic ordering within each i (optional)
    for i in buckets:
        buckets[i].sort(key=lambda t: t[1])

    batches: List[List[Idx2]] = []
    while buckets:
        # pick i with most remaining
        i0 = max(buckets, key=lambda i: len(buckets[i]))
        batch = buckets[i0][:batch_size]
        buckets[i0] = buckets[i0][len(batch):]
        if not buckets[i0]:
            del buckets[i0]

        # top off from other i's (largest first)
        need = batch_size - len(batch)
        if need:
            for i in sorted(buckets, key=lambda i: len(buckets[i]), reverse=True):
                if need == 0:
                    break
                take = min(need, len(buckets[i]))
                batch += buckets[i][:take]
                buckets[i] = buckets[i][take:]
                need -= take
                if not buckets[i]:
                    del buckets[i]

        batches.append(batch)

    return batches
