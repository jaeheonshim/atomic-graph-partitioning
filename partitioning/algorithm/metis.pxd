ctypedef int idx_t
ctypedef float real_t

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