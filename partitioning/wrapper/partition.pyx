cimport metis
from metis cimport idx_t, real_t, METIS_OK, METIS_ERROR_INPUT, METIS_ERROR_MEMORY, METIS_ERROR
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libcpp.vector cimport vector
from libcpp.deque cimport deque

cdef struct METIS_Graph:
    idx_t nvtxs
    idx_t ncon
    idx_t* xadj
    idx_t* adjncy
    idx_t* vwgt
    idx_t* vsize
    idx_t* adjwgt

cdef int _c_adjlist_to_metis(
    METIS_Graph* graph,
    idx_t* xadj_input,
    idx_t* adjncy_input,
    size_t n,
    size_t m,
    idx_t* vwgt_input=NULL,
    idx_t* vsize_input=NULL,
    idx_t* adjwgt_input=NULL,
    idx_t ncon=1
):
    cdef idx_t* xadj = <idx_t*>malloc((n+1) * sizeof(idx_t))
    if xadj == NULL:
        return -1
    
    cdef idx_t* adjncy = <idx_t*>malloc(m * sizeof(idx_t))
    if adjncy == NULL:
        free(xadj)
        return -1

    cdef size_t i
    for i in range(n+1):
        xadj[i] = xadj_input[i]
    
    for i in range(m):
        adjncy[i] = adjncy_input[i]

    cdef idx_t* vwgt = NULL
    cdef idx_t* vsize = NULL
    cdef idx_t* adjwgt = NULL
    
    if vwgt_input != NULL:
        vwgt = <idx_t*>malloc(n * ncon * sizeof(idx_t))
        if vwgt == NULL:
            free(xadj)
            free(adjncy)
            return -1
            
        for i in range(n * ncon):
            vwgt[i] = vwgt_input[i]
    
    if vsize_input != NULL:
        vsize = <idx_t*>malloc(n * sizeof(idx_t))
        if vsize == NULL:
            free(xadj)
            free(adjncy)
            if vwgt != NULL:
                free(vwgt)
            return -1
            
        for i in range(n):
            vsize[i] = vsize_input[i]

    if adjwgt_input != NULL:
        adjwgt = <idx_t*>malloc(m * sizeof(idx_t))
        if adjwgt == NULL:
            free(xadj)
            free(adjncy)
            if vwgt != NULL:
                free(vwgt)
            if vsize != NULL:
                free(vsize)
            return -1
            
        for i in range(m):
            adjwgt[i] = adjwgt_input[i]

    graph.nvtxs = n
    graph.ncon = ncon
    graph.xadj = xadj
    graph.adjncy = adjncy
    graph.vwgt = vwgt
    graph.vsize = vsize
    graph.adjwgt = adjwgt

    return 0

cdef void _c_descendants_at_distance_multisource(
    int nparts,
    idx_t* xadj_input,
    idx_t* adjncy_input,
    size_t n,
    idx_t* part,
    vector[vector[idx_t]]& result,
    int distance=0
):
    cdef deque[idx_t] queue
    cdef deque[int] depths
    cdef vector[bint] visited = vector[bint](n, False)
    cdef idx_t node
    cdef int d
    cdef vector[idx_t] current
    cdef idx_t child
    
    for i in range(nparts):
        current = vector[idx_t]()
        visited.assign(n, False)

        for j in range(n):
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

            for j in range(xadj_input[node], xadj_input[node + 1]):
                child = adjncy_input[j]
                if not visited[child]:
                    visited[child] = True
                    queue.push_back(child)
                    depths.push_back(d + 1)

        result.push_back(current)

cdef int _c_part_graph_kway_extended(
    vector[vector[idx_t]]& result,
    vector[idx_t]& flat_part_result,
    idx_t nparts,  # Changed from int to idx_t
    idx_t* xadj_input,
    idx_t* adjncy_input,
    size_t n,
    size_t m,
    idx_t* vwgt_input=NULL,
    idx_t* vsize_input=NULL,
    idx_t* adjwgt_input=NULL,
    idx_t ncon=1,
    real_t* tpwgts_ptr=NULL,
    real_t ubvec_val=1.03,
    int distance=0
):
    cdef idx_t _edgecut = 0
    cdef METIS_Graph graph
    memset(&graph, 0, sizeof(METIS_Graph))
    if _c_adjlist_to_metis(&graph, xadj_input, adjncy_input, n, m, vwgt_input, vsize_input, adjwgt_input, ncon) < 0:
        return -1
    
    cdef idx_t* part = <idx_t*>malloc(n * sizeof(idx_t))
    if part == NULL:
        free(graph.xadj)
        free(graph.adjncy)
        if graph.vwgt != NULL:
            free(graph.vwgt)
        if graph.vsize != NULL:
            free(graph.vsize)
        if graph.adjwgt != NULL:
            free(graph.adjwgt)
        return -1
        
    cdef int metis_result = metis.METIS_PartGraphKway(
        &graph.nvtxs,
        &graph.ncon,
        graph.xadj,
        graph.adjncy,
        graph.vwgt,
        graph.vsize,
        graph.adjwgt,
        &nparts,
        tpwgts_ptr,
        &ubvec_val,
        NULL,
        &_edgecut,
        part
    )

    # Check if METIS call was successful
    if metis_result != METIS_OK:
        # Clean up and return error
        free(part)
        free(graph.xadj)
        free(graph.adjncy)
        if graph.vwgt != NULL:
            free(graph.vwgt)
        if graph.vsize != NULL:
            free(graph.vsize)
        if graph.adjwgt != NULL:
            free(graph.adjwgt)
        return metis_result

    # Only proceed if METIS call was successful
    for i in range(n):
        flat_part_result.push_back(part[i])

    _c_descendants_at_distance_multisource(nparts, xadj_input, adjncy_input, n, part, result, distance)
    
    free(part)
    free(graph.xadj)
    free(graph.adjncy)
    if graph.vwgt != NULL:
        free(graph.vwgt)
    if graph.vsize != NULL:
        free(graph.vsize)
    if graph.adjwgt != NULL:
        free(graph.adjwgt)

    return 0


def part_graph_kway_extended(list adjlist, int nparts, list nodew=None, list nodesz=None, 
                            list tpwgts=None, float ubvec_val=1.03, int distance=0):
    cdef size_t n = len(adjlist)
    cdef size_t m = 0
    cdef size_t i, j, e
    cdef bint has_weights = False

    for adj in adjlist:
        m += len(adj)

    cdef idx_t* xadj = <idx_t*>malloc((n+1) * sizeof(idx_t))
    if xadj == NULL:
        raise MemoryError("Failed to allocate memory for xadj")
    
    cdef idx_t* adjncy = <idx_t*>malloc(m * sizeof(idx_t))
    if adjncy == NULL:
        free(xadj)
        raise MemoryError("Failed to allocate memory for adjncy")
    
    cdef idx_t* adjwgt = <idx_t*>malloc(m * sizeof(idx_t))
    if adjwgt == NULL:
        free(xadj)
        free(adjncy)
        raise MemoryError("Failed to allocate memory for adjwgt")

    xadj[0] = 0
    e = 0

    for i in range(n):
        adj = adjlist[i]
        for item in adj:
            try:
                neighbor, weight = item
                adjncy[e] = neighbor
                adjwgt[e] = weight
                has_weights = True
            except (TypeError, ValueError):
                adjncy[e] = item
                adjwgt[e] = 1
            
            e += 1
        xadj[i + 1] = e
    
    cdef idx_t* vwgt = NULL
    cdef idx_t ncon = 1
    
    if nodew is not None:
        if isinstance(nodew[0], int):
            vwgt = <idx_t*>malloc(n * sizeof(idx_t))
            if vwgt == NULL:
                free(xadj)
                free(adjncy)
                free(adjwgt)
                raise MemoryError("Failed to allocate memory for vwgt")
                
            for i in range(n):
                vwgt[i] = nodew[i]
        else:
            ncon = len(nodew[0])
            vwgt = <idx_t*>malloc(n * ncon * sizeof(idx_t))
            if vwgt == NULL:
                free(xadj)
                free(adjncy)
                free(adjwgt)
                raise MemoryError("Failed to allocate memory for vwgt")
                
            for i in range(n):
                for j in range(ncon):
                    vwgt[i*ncon + j] = nodew[i][j]

    cdef idx_t* vsize = NULL
    
    if nodesz is not None:
        vsize = <idx_t*>malloc(n * sizeof(idx_t))
        if vsize == NULL:
            free(xadj)
            free(adjncy)
            free(adjwgt)
            if vwgt != NULL:
                free(vwgt)
            raise MemoryError("Failed to allocate memory for vsize")
            
        for i in range(n):
            vsize[i] = nodesz[i]
    
    if not has_weights:
        adjwgt = NULL
    
    cdef real_t* tpwgts_ptr = NULL
    
    if tpwgts is not None:
        tpwgts_ptr = <real_t*>malloc(nparts * ncon * sizeof(real_t))
        if tpwgts_ptr == NULL:
            free(xadj)
            free(adjncy)
            if adjwgt != NULL:
                free(adjwgt)
            if vwgt != NULL:
                free(vwgt)
            if vsize != NULL:
                free(vsize)
            raise MemoryError("Failed to allocate memory for tpwgts")
            
        for i in range(nparts * ncon):
            tpwgts_ptr[i] = tpwgts[i]
    
    cdef vector[vector[idx_t]] result
    cdef vector[idx_t] flat_part_result
    _c_part_graph_kway_extended(
        result,
        flat_part_result,
        nparts, xadj, adjncy, n, m, vwgt, vsize, adjwgt, 
        ncon, tpwgts_ptr, ubvec_val, distance
    )

    free(xadj)
    free(adjncy)
    if adjwgt != NULL:
        free(adjwgt)
    if vwgt != NULL:
        free(vwgt)
    if vsize != NULL:
        free(vsize)
    if tpwgts_ptr != NULL:
        free(tpwgts_ptr)

    cdef list core_partitions = [set() for _ in range(nparts)]
    for i in range(flat_part_result.size()):
        core_partitions[flat_part_result[i]].add(i)

    cdef list extended_partitions = []
    cdef object group  # Python set
    cdef idx_t val

    for i in range(result.size()):
        group = set()  # Python set
        for j in range(result[i].size()):
            val = result[i][j]
            group.add(val)
        extended_partitions.append(group)

    return core_partitions, extended_partitions