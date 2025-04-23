cimport metis
from metis cimport idx_t, real_t, METIS_OK, METIS_ERROR_INPUT, METIS_ERROR_MEMORY, METIS_ERROR
from libc.stdlib cimport malloc, free

cdef struct METIS_Graph:
    idx_t nvtxs
    idx_t ncon
    idx_t* xadj
    idx_t* adjncy
    idx_t* vwgt
    idx_t* vsize
    idx_t* adjwgt


cdef METIS_Graph _c_adjlist_to_metis(
    idx_t* xadj_input,
    idx_t* adjncy_input,
    size_t n,
    size_t m,
    idx_t* vwgt_input=NULL,
    idx_t* vsize_input=NULL,
    idx_t* adjwgt_input=NULL,
    idx_t ncon=1
):
    cdef METIS_Graph graph

    cdef idx_t* xadj = <idx_t*>malloc((n+1) * sizeof(idx_t))
    if xadj == NULL:
        raise MemoryError("Failed to allocate memory for xadj")
    
    cdef idx_t* adjncy = <idx_t*>malloc(m * sizeof(idx_t))
    if adjncy == NULL:
        free(xadj)
        raise MemoryError("Failed to allocate memory for adjncy")

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
            raise MemoryError("Failed to allocate memory for vwgt")
            
        for i in range(n * ncon):
            vwgt[i] = vwgt_input[i]
    
    if vsize_input != NULL:
        vsize = <idx_t*>malloc(n * sizeof(idx_t))
        if vsize == NULL:
            free(xadj)
            free(adjncy)
            if vwgt != NULL:
                free(vwgt)
            raise MemoryError("Failed to allocate memory for vsize")
            
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
            raise MemoryError("Failed to allocate memory for adjwgt")
            
        for i in range(m):
            adjwgt[i] = adjwgt_input[i]

    graph.nvtxs = n
    graph.ncon = ncon
    graph.xadj = xadj
    graph.adjncy = adjncy
    graph.vwgt = vwgt
    graph.vsize = vsize
    graph.adjwgt = adjwgt

    return graph


cdef void _c_descendants_at_distance_multisource(G, sources, distance=None):

cdef void _c_part_graph_kway_extended(
    int nparts,
    idx_t* xadj_input,          # Starting index in adjncy_input for each vertex
    idx_t* adjncy_input,        # Flattened adjacency information
    size_t n,                   # The total number of vertices
    size_t m,                   # The total number of edges
    idx_t* vwgt_input=NULL,     # Optional vertex weights
    idx_t* vsize_input=NULL,    # Optional vertex sizes
    idx_t* adjwgt_input=NULL,   # Optional edge weights
    idx_t ncon=1,               # Number of constraints
    real_t* tpwgts_ptr=NULL,    # Target weights
    real_t ubvec_val=1.03       # Balance constraint factor
):
    cdef idx_t _edgecut = 0

    cdef METIS_Graph graph = _c_adjlist_to_metis(xadj_input, adjncy_input, n, m, vwgt_input, vsize_input, adjwgt_input, ncon)

    cdef idx_t* part = <idx_t*>malloc(n * sizeof(idx_t))

    cdef int result = metis.METIS_PartGraphKway(
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

    print('Result: ' + str(result))

    cdef size_t i
    for i in range(n):
        print(part[i])

    pass

def part_graph_kway_extended(list adjlist, int nparts, list nodew=None, list nodesz=None, 
                            list tpwgts=None, float ubvec_val=1.03):
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
        free(adjwgt)
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
    
    _c_part_graph_kway_extended(
        nparts, xadj, adjncy, n, m, vwgt, vsize, adjwgt, 
        ncon, tpwgts_ptr, ubvec_val
    )
    
    # Free allocated memory
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