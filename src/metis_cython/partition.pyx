# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.deque  cimport deque

cdef enum:
    METIS_OK            = 1
    METIS_ERROR_INPUT   = -2
    METIS_ERROR_MEMORY  = -3
    METIS_ERROR         = -4

cdef enum:
    METIS_OPTION_PTYPE     = 0
    METIS_OPTION_OBJTYPE   = 1
    METIS_OPTION_CTYPE     = 2
    METIS_OPTION_IPTYPE    = 3
    METIS_OPTION_RTYPE     = 4
    METIS_OPTION_DBGLVL    = 5
    METIS_OPTION_NITER     = 7
    METIS_OPTION_NCUTS     = 8
    METIS_OPTION_SEED      = 9
    METIS_OPTION_NUMBERING = 18

cdef enum:
    METIS_OBJTYPE_CUT = 0
    METIS_OBJTYPE_VOL = 1

cdef enum:
    METIS_PTYPE_KWAY = 1

cdef extern from "metis.h":
    ctypedef int   idx_t
    ctypedef float real_t
    int METIS_NOPTIONS

    int METIS_PartGraphKway(
        idx_t *nvtxs,    idx_t *ncon,
        idx_t *xadj,     idx_t *adjncy,
        idx_t *vwgt,     idx_t *vsize,
        idx_t *adjwgt,   idx_t *nparts,
        real_t *tpwgts,  real_t *ubvec,
        idx_t *options,  idx_t *edgecut,
        idx_t *part
    )
    int METIS_SetDefaultOptions(idx_t *options)


cdef void _c_descendants_at_distance_multisource(
    int                nparts,     idx_t nvtxs,
    idx_t*             xadj,       idx_t* adjncy,
    idx_t*             part,       vector[vector[idx_t]]& result,
    int distance = 0
):
    cdef deque[idx_t] queue
    cdef deque[int]   depths
    cdef vector[bint] visited = vector[bint](nvtxs, False)
    cdef idx_t        node, child
    cdef int          d
    cdef vector[idx_t] current

    for i in range(nparts):
        current = vector[idx_t]()
        visited.assign(nvtxs, False)

        for j in range(nvtxs):
            if part[j] == i:
                queue.push_back(j)
                depths.push_back(0)

        while not queue.empty():
            node = queue.front();  queue.pop_front()
            d    = depths.front(); depths.pop_front()

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

def part_graph_kway_extended(list adjlist,
                              int  num_partitions,
                              int  distance = 0):
    cdef idx_t nvtxs = len(adjlist)
    cdef idx_t ncon  = 1

    cdef idx_t *xadj   = NULL
    cdef idx_t *adjncy = NULL
    cdef idx_t *part   = NULL
    cdef idx_t *options = NULL

    cdef idx_t nparts  = num_partitions
    cdef real_t ubvec  = 1.0
    cdef idx_t edgecut

    cdef size_t m = 0
    for neighbours in adjlist:
        m += len(neighbours)

    cdef vector[vector[idx_t]] result

    cdef size_t prefix = 0
    cdef size_t i, e = 0

    cdef list core_partitions = [set() for _ in range(num_partitions)]

    cdef list extended_partitions = []
    cdef vector[idx_t] bucket

    try:
        options = <idx_t*>malloc(METIS_NOPTIONS * sizeof(idx_t))
        xadj    = <idx_t*>malloc((nvtxs + 1)  * sizeof(idx_t))
        adjncy  = <idx_t*>malloc(m            * sizeof(idx_t))
        part    = <idx_t*>malloc(nvtxs        * sizeof(idx_t))
        if not (options and xadj and adjncy and part):
            raise MemoryError("out of memory")
        
        for i in range(nvtxs):
            xadj[i] = prefix
            prefix += len(adjlist[i])
        xadj[nvtxs] = prefix

        for neighbours in adjlist:
            for n in neighbours:
                adjncy[e] = n
                e += 1

        METIS_SetDefaultOptions(options)
        options[METIS_OPTION_NUMBERING] = 0
        options[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_CUT

        METIS_PartGraphKway(
            &nvtxs, &ncon,
            xadj, adjncy,
            NULL,  NULL, NULL,
            &nparts,
            NULL, &ubvec,
            options,
            &edgecut,
            part)

        for i in range(nvtxs):
            core_partitions[part[i]].add(i)

        _c_descendants_at_distance_multisource(
            num_partitions, nvtxs,
            xadj, adjncy, part,
            result, distance)

        for bucket in result:
            extended_partitions.append(
                set(bucket[j] for j in range(bucket.size()))
            )

        return core_partitions, extended_partitions

    finally:
        if options: free(options)
        if part:    free(part)
        if adjncy:  free(adjncy)
        if xadj:    free(xadj)
