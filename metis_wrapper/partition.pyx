from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libcpp.vector cimport vector
from libcpp.deque cimport deque

cdef enum:
    METIS_OK = 1
    METIS_ERROR_INPUT = -2
    METIS_ERROR_MEMORY = -3
    METIS_ERROR = -4

cdef enum:
    METIS_OPTION_PTYPE = 0
    METIS_OPTION_OBJTYPE = 1
    METIS_OPTION_CTYPE = 2
    METIS_OPTION_IPTYPE = 3
    METIS_OPTION_RTYPE = 4
    METIS_OPTION_DBGLVL = 5
    METIS_OPTION_NITER = 7
    METIS_OPTION_NCUTS = 8
    METIS_OPTION_SEED = 9
    METIS_OPTION_NUMBERING = 18

cdef enum:
    METIS_OBJTYPE_CUT = 0
    METIS_OBJTYPE_VOL = 1

cdef enum:
    METIS_PTYPE_KWAY = 1

cdef extern from "metis.h":
    ctypedef int idx_t
    ctypedef float real_t
    int METIS_NOPTIONS

    int METIS_PartGraphKway(
        idx_t *nvtxs,
        idx_t *ncon,
        idx_t *xadj,
        idx_t *adjncy,
        idx_t *vwgt,
        idx_t *vsize,
        idx_t *adjwgt,
        idx_t *nparts,
        real_t *tpwgts,
        real_t *ubvec,
        idx_t *options,
        idx_t *edgecut,
        idx_t *part
    )
    
    int METIS_SetDefaultOptions(idx_t *options)
    
    int METIS_Free(void *ptr)

def validate_adjlist(list adjlist):
    cdef size_t n = len(adjlist)
    cdef size_t i, j
    cdef object neighbor
    cdef int neighbor_idx
    cdef object weight

    for i in range(n):
        if not isinstance(adjlist[i], list):
            raise ValueError(f"adjlist[{i}] is not a list")
        for j in range(len(adjlist[i])):
            neighbor = adjlist[i][j]
            if isinstance(neighbor, int):
                neighbor_idx = neighbor
                if neighbor_idx < 0 or neighbor_idx >= n:
                    raise ValueError(f"Invalid neighbor index {neighbor_idx} at adjlist[{i}][{j}]")
                if neighbor_idx == i:
                    raise ValueError(f"Self-loop detected at node {i}")
            elif isinstance(neighbor, tuple):
                if len(neighbor) != 2:
                    raise ValueError(f"Neighbor tuple at adjlist[{i}][{j}] must have length 2")
                neighbor_idx, weight = neighbor
                if not isinstance(neighbor_idx, int):
                    raise ValueError(f"First element of neighbor tuple must be int at adjlist[{i}][{j}]")
                if neighbor_idx < 0 or neighbor_idx >= n:
                    raise ValueError(f"Invalid neighbor index {neighbor_idx} at adjlist[{i}][{j}]")
                if neighbor_idx == i:
                    raise ValueError(f"Self-loop detected at node {i}")
                if not (isinstance(weight, int) or isinstance(weight, float)):
                    raise ValueError(f"Second element of neighbor tuple must be a number at adjlist[{i}][{j}]")
                if weight < 0:
                    raise ValueError(f"Weight must be non-negative at adjlist[{i}][{j}]")
            else:
                raise ValueError(f"Neighbor at adjlist[{i}][{j}] must be int or tuple")

cdef void _c_descendants_at_distance_multisource(
    int nparts,
    idx_t nvtxs,
    idx_t* xadj,
    idx_t* adjncy,
    idx_t* part,
    vector[vector[idx_t]]& result,
    int distance=0
):
    cdef deque[idx_t] queue
    cdef deque[int] depths
    cdef vector[bint] visited = vector[bint](nvtxs, False)
    cdef idx_t node
    cdef int d
    cdef vector[idx_t] current
    cdef idx_t child
    
    for i in range(nparts):
        current = vector[idx_t]()
        visited.assign(nvtxs, False)

        for j in range(nvtxs):
            if part[j] == i:
                queue.push_back(j)
                depths.push_back(0)

        while not queue.empty():
            node = queue.front()
            d = depths.front()

            queue.pop_front()
            depths.pop_front()

            if distance != -1 and d > distance:
                continue
            
            current.push_back(node)

            for j in range(xadj[node], xadj[node + 1]):
                child = adjncy[j]
                if not visited[child]:
                    visited[child] = True
                    queue.push_back(child)
                    depths.push_back(d + 1)

        result.push_back(current)

def part_graph_kway_extended(list adjlist, int num_partitions, int distance=0):
    cdef idx_t nvtxs = len(adjlist)
    cdef idx_t ncon = 1
    cdef idx_t *xadj = NULL
    cdef idx_t *adjncy = NULL
    cdef idx_t *vwgt = NULL
    cdef idx_t *vsize = NULL
    cdef idx_t *adjwgt = NULL
    cdef idx_t nparts = num_partitions
    cdef real_t *tpwgts = NULL
    cdef real_t ubvec = 1
    cdef idx_t *options = NULL
    cdef idx_t edgecut
    cdef idx_t *part = NULL

    cdef size_t m = 0
    cdef size_t i, j, e

    cdef vector[vector[idx_t]] result

    validate_adjlist(adjlist)

    for neighbors in adjlist:
        m += len(neighbors)

    try:
        options = <idx_t*>malloc(METIS_NOPTIONS * sizeof(idx_t))
        if options == NULL:
            raise MemoryError("Failed to allocate options")

        xadj = <idx_t*>malloc((nvtxs+1) * sizeof(idx_t))
        if xadj == NULL:
            raise MemoryError("Failed to allocate xadj")

        adjncy = <idx_t*>malloc(m * sizeof(idx_t))
        if adjncy == NULL:
            raise MemoryError("Failed to allocate adjncy")

        part = <idx_t*>malloc(nvtxs * sizeof(idx_t))
        if part == NULL:
            raise MemoryError("Failed to allocate part")

        # Set up xadj, adjncy
        e = 0
        for i in range(nvtxs):
            xadj[i] = e
            for j in range(len(adjlist[i])):
                adjncy[e] = adjlist[i][j]
                e += 1
        xadj[nvtxs] = e

        # METIS calls
        METIS_SetDefaultOptions(options)
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT
        options[METIS_OPTION_NUMBERING] = 0

        METIS_PartGraphKway(
            &nvtxs,
            &ncon,
            xadj,
            adjncy,
            vwgt,
            vsize,
            adjwgt,
            &nparts,
            tpwgts,
            &ubvec,
            options,
            &edgecut,
            part
        )

        _c_descendants_at_distance_multisource(
            nparts,
            nvtxs,
            xadj,
            adjncy,
            part,
            result,
            distance
        )

        core_partitions = [set() for _ in range(nparts)]
        
        for i in range(nvtxs):
            core_partitions[part[i]].add(i)

        return core_partitions
    finally:
        if options != NULL:
            free(options)
        if part != NULL:
            free(part)
        if adjncy != NULL:
            free(adjncy)
        if xadj != NULL:
            free(xadj)