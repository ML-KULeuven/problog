cdef class Term:
    cdef public int id
    cdef public object probability, location, op_priority, op_spec
    cdef public int __arity, __hash, reprhash
    cdef str repr, __signature
    cdef dict __dict__
    cdef bint _cached_hash
    cdef public object __functor, __args, _cache_is_ground
    cdef public object _cache_list_length, _cache_variables, cache_eq

    cdef str crepr(self)
    cdef long _list_length(self)