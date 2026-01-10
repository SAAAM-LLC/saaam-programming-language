/*
SAAAM Language - Native Runtime Implementation
Dependency-free C runtime with optional CUDA kernel library integration.
*/

#include "saaam_native_runtime.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <time.h>
#endif

#ifndef SAAAM_WITH_CUDA
  #define SAAAM_WITH_CUDA 0
#endif

#ifndef SAAAM_STACK_SIZE_BYTES
  #define SAAAM_STACK_SIZE_BYTES (2u * 1024u * 1024u)
#endif

#ifndef SAAAM_NEURAL_POOL_SIZE_BYTES
  #define SAAAM_NEURAL_POOL_SIZE_BYTES (16u * 1024u * 1024u)
#endif

#ifndef SAAAM_ARENA_BLOCK_SIZE_BYTES
  #define SAAAM_ARENA_BLOCK_SIZE_BYTES (1u * 1024u * 1024u)
#endif

typedef struct {
    uint8_t* memory;
    size_t size;
    size_t used;
    saaam_value_t* freelist;
} saaam_value_pool_t;

typedef struct saaam_arena_block {
    uint8_t* memory;
    size_t size;
    size_t used;
    saaam_value_t* freelist;
    struct saaam_arena_block* next;
} saaam_arena_block_t;

typedef struct {
    saaam_value_t** items;
    size_t count;
    size_t capacity;
} saaam_ptr_vec_t;

typedef struct {
    saaam_event_handler_t handler;
    void* data;
} saaam_event_t;

typedef struct {
    saaam_value_t* a;
    uint64_t a_id;
    uint64_t a_version_seen;
    saaam_value_t* b;
    uint64_t b_id;
    uint64_t b_version_seen;
} saaam_binding_t;

typedef struct {
    const char* operation;
    uint64_t start_ns;
} saaam_profile_frame_t;

struct saaam_runtime {
    bool enable_ternary_logic;
    bool enable_gpu_acceleration;
    bool enable_neuroplastic_optimization;

    uint64_t next_instance_id;

    saaam_value_pool_t stack_pool;
    saaam_value_pool_t neural_pool;
    saaam_arena_block_t* arena_blocks;

    saaam_ptr_vec_t registry_all;
    saaam_ptr_vec_t registry_gc;

    saaam_event_t* events;
    size_t event_count;
    size_t event_capacity;

    saaam_binding_t* bindings;
    size_t binding_count;
    size_t binding_capacity;
    bool binding_in_propagation;

    saaam_profile_frame_t profile_stack[128];
    size_t profile_depth;

    saaam_runtime_stats_t stats;
};

// Forward declarations for internal use.
void saaam_release_value(saaam_runtime_t* runtime, saaam_value_t* value);
saaam_value_t* saaam_morph_value(saaam_runtime_t* runtime, saaam_value_t* value, saaam_type_t target_type);
bool saaam_value_set_int(saaam_runtime_t* runtime, saaam_value_t* value, int64_t new_value);
bool saaam_value_set_float(saaam_runtime_t* runtime, saaam_value_t* value, double new_value);
bool saaam_value_set_bool(saaam_runtime_t* runtime, saaam_value_t* value, bool new_value);
bool saaam_value_set_string(saaam_runtime_t* runtime, saaam_value_t* value, const char* new_value);

static uint64_t saaam_time_ns(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    static bool inited = false;
    if (!inited) {
        QueryPerformanceFrequency(&freq);
        inited = true;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000ULL) / (uint64_t)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static char* saaam_strdup_impl(const char* s, size_t* out_len) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char* p = (char*)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n + 1);
    if (out_len) *out_len = n;
    return p;
}

static size_t saaam_align_up(size_t x, size_t align) {
    size_t mask = align - 1;
    return (x + mask) & ~mask;
}

static bool vec_reserve(saaam_ptr_vec_t* v, size_t cap) {
    if (v->capacity >= cap) return true;
    size_t new_cap = v->capacity ? v->capacity : 1024;
    while (new_cap < cap) new_cap *= 2;
    void* p = realloc(v->items, new_cap * sizeof(v->items[0]));
    if (!p) return false;
    v->items = (saaam_value_t**)p;
    v->capacity = new_cap;
    return true;
}

static void registry_add(saaam_ptr_vec_t* reg, saaam_value_t* v, size_t* index_field) {
    if (!reg || !v || !index_field) return;
    if (!vec_reserve(reg, reg->count + 1)) return;
    *index_field = reg->count;
    reg->items[reg->count++] = v;
}

static void registry_remove(saaam_ptr_vec_t* reg, saaam_value_t* v, size_t* index_field, bool is_main_registry) {
    if (!reg || !v || !index_field) return;
    size_t idx = *index_field;
    if (idx == SIZE_MAX || idx >= reg->count) return;
    size_t last = reg->count - 1;
    if (idx != last) {
        saaam_value_t* moved = reg->items[last];
        reg->items[idx] = moved;
        if (is_main_registry) moved->registry_index = idx;
        else moved->gc_index = idx;
    }
    reg->count = last;
    *index_field = SIZE_MAX;
}

static saaam_value_t* pool_pop(saaam_value_pool_t* p) {
    if (!p || !p->freelist) return NULL;
    saaam_value_t* v = p->freelist;
    p->freelist = v->free_next;
    v->free_next = NULL;
    return v;
}

static void pool_push(saaam_value_pool_t* p, saaam_value_t* v) {
    if (!p || !v) return;
    v->free_next = p->freelist;
    p->freelist = v;
}

static saaam_value_t* pool_alloc(saaam_value_pool_t* p) {
    if (!p) return NULL;
    saaam_value_t* v = pool_pop(p);
    if (v) return v;

    size_t align = (size_t)_Alignof(saaam_value_t);
    size_t off = saaam_align_up(p->used, align);
    if (off + sizeof(saaam_value_t) > p->size) return NULL;
    v = (saaam_value_t*)(p->memory + off);
    p->used = off + sizeof(saaam_value_t);
    return v;
}

static saaam_value_t* arena_alloc(saaam_runtime_t* rt) {
    if (!rt) return NULL;

    for (saaam_arena_block_t* b = rt->arena_blocks; b; b = b->next) {
        if (b->freelist) {
            saaam_value_t* v = b->freelist;
            b->freelist = v->free_next;
            v->free_next = NULL;
            return v;
        }
    }

    if (!rt->arena_blocks || rt->arena_blocks->used + sizeof(saaam_value_t) > rt->arena_blocks->size) {
        saaam_arena_block_t* b = (saaam_arena_block_t*)calloc(1, sizeof(*b));
        if (!b) return NULL;
        b->size = (size_t)SAAAM_ARENA_BLOCK_SIZE_BYTES;
        b->memory = (uint8_t*)malloc(b->size);
        if (!b->memory) {
            free(b);
            return NULL;
        }
        b->next = rt->arena_blocks;
        rt->arena_blocks = b;
    }

    size_t align = (size_t)_Alignof(saaam_value_t);
    size_t off = saaam_align_up(rt->arena_blocks->used, align);
    if (off + sizeof(saaam_value_t) > rt->arena_blocks->size) return NULL;
    saaam_value_t* v = (saaam_value_t*)(rt->arena_blocks->memory + off);
    rt->arena_blocks->used = off + sizeof(saaam_value_t);
    return v;
}

static void arena_push(saaam_runtime_t* rt, saaam_value_t* v) {
    if (!rt || !v) return;
    for (saaam_arena_block_t* b = rt->arena_blocks; b; b = b->next) {
        uint8_t* start = b->memory;
        uint8_t* end = b->memory + b->size;
        uint8_t* p = (uint8_t*)v;
        if (p >= start && p < end) {
            v->free_next = b->freelist;
            b->freelist = v;
            return;
        }
    }
}

static void value_finalize_nested(saaam_runtime_t* rt, saaam_value_t* v) {
    if (!v) return;

    if (v->flags & SAAAM_VALUE_FLAG_ON_GPU) {
        v->gpu_data = NULL;
        v->gpu_bytes = 0;
        v->flags &= ~SAAAM_VALUE_FLAG_ON_GPU;
    }

    saaam_morph_history_t* h = v->morph_history;
    while (h) {
        saaam_morph_history_t* next = h->next;
        free(h);
        h = next;
    }
    v->morph_history = NULL;
    v->morph_count = 0;

    switch (v->type) {
        case SAAAM_TYPE_STRING:
            free(v->data.string.ptr);
            v->data.string.ptr = NULL;
            v->data.string.len = 0;
            break;

        case SAAAM_TYPE_ARRAY:
            if (v->data.array.elements) {
                for (size_t i = 0; i < v->data.array.length; i++) {
                    if (v->data.array.elements[i]) {
                        saaam_release_value(rt, v->data.array.elements[i]);
                        v->data.array.elements[i] = NULL;
                    }
                }
                free(v->data.array.elements);
                v->data.array.elements = NULL;
            }
            v->data.array.length = 0;
            v->data.array.capacity = 0;
            break;

        case SAAAM_TYPE_FUNCTION:
            if (v->data.function.dtor && v->data.function.user_data) {
                v->data.function.dtor(v->data.function.user_data);
            }
            v->data.function.fn = NULL;
            v->data.function.user_data = NULL;
            v->data.function.dtor = NULL;
            break;

        case SAAAM_TYPE_COMPONENT:
            if (v->data.component.slots) {
                for (size_t i = 0; i < v->data.component.slot_count; i++) {
                    if (v->data.component.slots[i]) {
                        saaam_release_value(rt, v->data.component.slots[i]);
                        v->data.component.slots[i] = NULL;
                    }
                }
                free(v->data.component.slots);
                v->data.component.slots = NULL;
                v->data.component.slot_count = 0;
            }
            break;

        default:
            break;
    }
}

static void binding_gc_compact(saaam_runtime_t* rt) {
    if (!rt) return;
    size_t out = 0;
    for (size_t i = 0; i < rt->binding_count; i++) {
        saaam_binding_t b = rt->bindings[i];
        bool a_ok = b.a && b.a->registry_index != SIZE_MAX && b.a->instance_id == b.a_id;
        bool b_ok = b.b && b.b->registry_index != SIZE_MAX && b.b->instance_id == b.b_id;
        if (!a_ok || !b_ok) continue;
        rt->bindings[out++] = b;
    }
    rt->binding_count = out;
}

static bool value_assign_scalar(saaam_runtime_t* rt, saaam_value_t* dst, const saaam_value_t* src) {
    if (!rt || !dst || !src) return false;

    switch (src->type) {
        case SAAAM_TYPE_INT:
            return saaam_value_set_int(rt, dst, src->data.i64);
        case SAAAM_TYPE_FLOAT:
            return saaam_value_set_float(rt, dst, src->data.f64);
        case SAAAM_TYPE_BOOL:
            return saaam_value_set_bool(rt, dst, src->data.boolean);
        case SAAAM_TYPE_TERNARY:
            if ((dst->flags & SAAAM_VALUE_FLAG_NEURAL) && dst->type != SAAAM_TYPE_TERNARY) {
                if (!saaam_morph_value(rt, dst, SAAAM_TYPE_TERNARY)) return false;
            }
            if (dst->type != SAAAM_TYPE_TERNARY) return false;
            dst->data.ternary = src->data.ternary;
            dst->version++;
            return true;
        case SAAAM_TYPE_STRING:
            return saaam_value_set_string(rt, dst, src->data.string.ptr ? src->data.string.ptr : "");
        default:
            return false;
    }
}

static void bindings_propagate(saaam_runtime_t* rt, saaam_value_t* changed) {
    if (!rt || !changed) return;
    if (rt->binding_in_propagation) return;
    rt->binding_in_propagation = true;

    binding_gc_compact(rt);

    for (size_t i = 0; i < rt->binding_count; i++) {
        saaam_binding_t* b = &rt->bindings[i];
        if (b->a == changed && b->a->instance_id == b->a_id) {
            if (b->a_version_seen != b->a->version) {
                (void)value_assign_scalar(rt, b->b, b->a);
                b->a_version_seen = b->a->version;
                b->b_version_seen = b->b->version;
            }
        } else if (b->b == changed && b->b->instance_id == b->b_id) {
            if (b->b_version_seen != b->b->version) {
                (void)value_assign_scalar(rt, b->a, b->b);
                b->a_version_seen = b->a->version;
                b->b_version_seen = b->b->version;
            }
        }
    }

    rt->binding_in_propagation = false;
}

// ---- Public API ----

saaam_runtime_t* saaam_runtime_init(void) {
    saaam_runtime_t* rt = (saaam_runtime_t*)calloc(1, sizeof(*rt));
    if (!rt) return NULL;

    rt->enable_ternary_logic = true;
    rt->enable_gpu_acceleration = false;
    rt->enable_neuroplastic_optimization = true;
    rt->next_instance_id = 1;

    rt->stack_pool.size = (size_t)SAAAM_STACK_SIZE_BYTES;
    rt->stack_pool.memory = (uint8_t*)malloc(rt->stack_pool.size);
    rt->stack_pool.used = 0;
    rt->stack_pool.freelist = NULL;

    rt->neural_pool.size = (size_t)SAAAM_NEURAL_POOL_SIZE_BYTES;
    rt->neural_pool.memory = (uint8_t*)malloc(rt->neural_pool.size);
    rt->neural_pool.used = 0;
    rt->neural_pool.freelist = NULL;

    if (!rt->stack_pool.memory || !rt->neural_pool.memory) {
        saaam_runtime_destroy(rt);
        return NULL;
    }

    rt->registry_all.items = NULL;
    rt->registry_all.count = 0;
    rt->registry_all.capacity = 0;
    rt->registry_gc.items = NULL;
    rt->registry_gc.count = 0;
    rt->registry_gc.capacity = 0;

    rt->events = NULL;
    rt->event_count = 0;
    rt->event_capacity = 0;

    rt->bindings = NULL;
    rt->binding_count = 0;
    rt->binding_capacity = 0;
    rt->binding_in_propagation = false;

    rt->profile_depth = 0;
    memset(&rt->stats, 0, sizeof(rt->stats));

    return rt;
}

void saaam_runtime_destroy(saaam_runtime_t* rt) {
    if (!rt) return;

    while (rt->registry_all.count) {
        saaam_value_t* v = rt->registry_all.items[rt->registry_all.count - 1];
        registry_remove(&rt->registry_all, v, &v->registry_index, true);
        if (v->memory_region == SAAAM_MEMORY_GC) {
            registry_remove(&rt->registry_gc, v, &v->gc_index, false);
        }
        value_finalize_nested(rt, v);

        if (v->memory_region == SAAAM_MEMORY_HEAP || v->memory_region == SAAAM_MEMORY_GC) {
            free(v);
        } else if (v->memory_region == SAAAM_MEMORY_STACK) {
            pool_push(&rt->stack_pool, v);
        } else if (v->memory_region == SAAAM_MEMORY_NEURAL) {
            pool_push(&rt->neural_pool, v);
        } else if (v->memory_region == SAAAM_MEMORY_ARENA) {
            arena_push(rt, v);
        }
    }

    free(rt->registry_all.items);
    free(rt->registry_gc.items);
    free(rt->events);
    free(rt->bindings);

    free(rt->stack_pool.memory);
    free(rt->neural_pool.memory);

    saaam_arena_block_t* b = rt->arena_blocks;
    while (b) {
        saaam_arena_block_t* next = b->next;
        free(b->memory);
        free(b);
        b = next;
    }

    free(rt);
}

void saaam_runtime_set_ternary_logic(saaam_runtime_t* rt, bool enabled) {
    if (!rt) return;
    rt->enable_ternary_logic = enabled;
}

void saaam_runtime_set_gpu_acceleration(saaam_runtime_t* rt, bool enabled) {
    if (!rt) return;
    rt->enable_gpu_acceleration = enabled && saaam_cuda_available();
}

void saaam_runtime_set_neuroplastic_optimization(saaam_runtime_t* rt, bool enabled) {
    if (!rt) return;
    rt->enable_neuroplastic_optimization = enabled;
}

saaam_runtime_stats_t saaam_runtime_get_stats(const saaam_runtime_t* rt) {
    saaam_runtime_stats_t z;
    memset(&z, 0, sizeof(z));
    if (!rt) return z;
    return rt->stats;
}

void saaam_print_performance_stats(const saaam_runtime_t* rt) {
    if (!rt) return;
    fprintf(stdout, "SAAAM Runtime Stats\n");
    fprintf(stdout, "  morph_operations: %llu\n", (unsigned long long)rt->stats.morph_operations);
    fprintf(stdout, "  gc_cycles:        %llu\n", (unsigned long long)rt->stats.gc_cycles);
    fprintf(stdout, "  gpu_transfers:    %llu\n", (unsigned long long)rt->stats.gpu_transfers);
    fprintf(stdout, "  events_emitted:   %llu\n", (unsigned long long)rt->stats.events_emitted);
    fprintf(stdout, "  events_processed: %llu\n", (unsigned long long)rt->stats.events_processed);
}

saaam_value_t* saaam_alloc_value(saaam_runtime_t* rt, saaam_type_t type, saaam_memory_region_t region) {
    if (!rt) return NULL;

    saaam_value_t* v = NULL;
    switch (region) {
        case SAAAM_MEMORY_STACK:
            v = pool_alloc(&rt->stack_pool);
            break;
        case SAAAM_MEMORY_NEURAL:
            v = pool_alloc(&rt->neural_pool);
            break;
        case SAAAM_MEMORY_ARENA:
            v = arena_alloc(rt);
            break;
        case SAAAM_MEMORY_HEAP:
        case SAAAM_MEMORY_GC:
            v = (saaam_value_t*)calloc(1, sizeof(*v));
            break;
        default:
            return NULL;
    }
    if (!v) return NULL;

    memset(v, 0, sizeof(*v));
    v->type = type;
    v->memory_region = region;
    v->flags = SAAAM_VALUE_FLAG_MUTABLE;
    if (region == SAAAM_MEMORY_NEURAL) v->flags |= SAAAM_VALUE_FLAG_NEURAL;

    v->instance_id = rt->next_instance_id++;
    v->version = 1;
    v->ref_count = 1;
    v->registry_index = SIZE_MAX;
    v->gc_index = SIZE_MAX;

    registry_add(&rt->registry_all, v, &v->registry_index);
    if (region == SAAAM_MEMORY_GC) {
        registry_add(&rt->registry_gc, v, &v->gc_index);
    }

    return v;
}

void saaam_retain_value(saaam_runtime_t* runtime, saaam_value_t* value) {
    (void)runtime;
    if (!value) return;
    value->ref_count++;
}

void saaam_free_value(saaam_runtime_t* runtime, saaam_value_t* value) {
    saaam_release_value(runtime, value);
}

void saaam_release_value(saaam_runtime_t* rt, saaam_value_t* v) {
    if (!rt || !v) return;
    if (v->ref_count == 0) return;

    v->ref_count--;
    if (v->ref_count != 0) return;

    registry_remove(&rt->registry_all, v, &v->registry_index, true);
    if (v->memory_region == SAAAM_MEMORY_GC) {
        registry_remove(&rt->registry_gc, v, &v->gc_index, false);
    }

    value_finalize_nested(rt, v);

    if (v->memory_region == SAAAM_MEMORY_HEAP || v->memory_region == SAAAM_MEMORY_GC) {
        free(v);
    } else if (v->memory_region == SAAAM_MEMORY_STACK) {
        pool_push(&rt->stack_pool, v);
    } else if (v->memory_region == SAAAM_MEMORY_NEURAL) {
        pool_push(&rt->neural_pool, v);
    } else if (v->memory_region == SAAAM_MEMORY_ARENA) {
        arena_push(rt, v);
    }
}

saaam_value_t* saaam_create_int(saaam_runtime_t* rt, int64_t value) {
    saaam_value_t* v = saaam_alloc_value(rt, SAAAM_TYPE_INT, SAAAM_MEMORY_STACK);
    if (!v) return NULL;
    v->data.i64 = value;
    return v;
}

saaam_value_t* saaam_create_float(saaam_runtime_t* rt, double value) {
    saaam_value_t* v = saaam_alloc_value(rt, SAAAM_TYPE_FLOAT, SAAAM_MEMORY_STACK);
    if (!v) return NULL;
    v->data.f64 = value;
    return v;
}

saaam_value_t* saaam_create_string(saaam_runtime_t* rt, const char* value) {
    saaam_value_t* v = saaam_alloc_value(rt, SAAAM_TYPE_STRING, SAAAM_MEMORY_HEAP);
    if (!v) return NULL;
    v->data.string.ptr = saaam_strdup_impl(value ? value : "", &v->data.string.len);
    if (!v->data.string.ptr) {
        saaam_release_value(rt, v);
        return NULL;
    }
    return v;
}

saaam_value_t* saaam_create_bool(saaam_runtime_t* rt, bool value) {
    saaam_value_t* v = saaam_alloc_value(rt, SAAAM_TYPE_BOOL, SAAAM_MEMORY_STACK);
    if (!v) return NULL;
    v->data.boolean = value;
    return v;
}

saaam_value_t* saaam_create_neural(saaam_runtime_t* rt, saaam_type_t initial_type) {
    saaam_value_t* v = saaam_alloc_value(rt, initial_type, SAAAM_MEMORY_NEURAL);
    if (!v) return NULL;
    v->flags |= SAAAM_VALUE_FLAG_NEURAL;
    return v;
}

saaam_value_t* saaam_create_function(saaam_runtime_t* rt, saaam_native_fn_t fn, void* user_data, saaam_native_fn_dtor_t dtor) {
    saaam_value_t* v = saaam_alloc_value(rt, SAAAM_TYPE_FUNCTION, SAAAM_MEMORY_HEAP);
    if (!v) return NULL;
    v->data.function.fn = fn;
    v->data.function.user_data = user_data;
    v->data.function.dtor = dtor;
    return v;
}

saaam_value_t* saaam_create_component(saaam_runtime_t* rt) {
    saaam_value_t* v = saaam_alloc_value(rt, SAAAM_TYPE_COMPONENT, SAAAM_MEMORY_HEAP);
    if (!v) return NULL;
    v->data.component.slot_count = (size_t)SAAAM_TYPE_MAX;
    v->data.component.slots = (saaam_value_t**)calloc(v->data.component.slot_count, sizeof(saaam_value_t*));
    if (!v->data.component.slots) {
        saaam_release_value(rt, v);
        return NULL;
    }
    return v;
}

bool saaam_value_set_int(saaam_runtime_t* rt, saaam_value_t* v, int64_t new_value) {
    if (!rt || !v) return false;
    if ((v->flags & SAAAM_VALUE_FLAG_NEURAL) && v->type != SAAAM_TYPE_INT) {
        if (!saaam_morph_value(rt, v, SAAAM_TYPE_INT)) return false;
    }
    if (v->type != SAAAM_TYPE_INT) return false;
    v->data.i64 = new_value;
    v->version++;
    bindings_propagate(rt, v);
    return true;
}

bool saaam_value_set_float(saaam_runtime_t* rt, saaam_value_t* v, double new_value) {
    if (!rt || !v) return false;
    if ((v->flags & SAAAM_VALUE_FLAG_NEURAL) && v->type != SAAAM_TYPE_FLOAT) {
        if (!saaam_morph_value(rt, v, SAAAM_TYPE_FLOAT)) return false;
    }
    if (v->type != SAAAM_TYPE_FLOAT) return false;
    v->data.f64 = new_value;
    v->version++;
    bindings_propagate(rt, v);
    return true;
}

bool saaam_value_set_bool(saaam_runtime_t* rt, saaam_value_t* v, bool new_value) {
    if (!rt || !v) return false;
    if ((v->flags & SAAAM_VALUE_FLAG_NEURAL) && v->type != SAAAM_TYPE_BOOL) {
        if (!saaam_morph_value(rt, v, SAAAM_TYPE_BOOL)) return false;
    }
    if (v->type != SAAAM_TYPE_BOOL) return false;
    v->data.boolean = new_value;
    v->version++;
    bindings_propagate(rt, v);
    return true;
}

bool saaam_value_set_string(saaam_runtime_t* rt, saaam_value_t* v, const char* new_value) {
    if (!rt || !v) return false;
    if ((v->flags & SAAAM_VALUE_FLAG_NEURAL) && v->type != SAAAM_TYPE_STRING) {
        if (!saaam_morph_value(rt, v, SAAAM_TYPE_STRING)) return false;
    }
    if (v->type != SAAAM_TYPE_STRING) return false;
    size_t len = 0;
    char* p = saaam_strdup_impl(new_value ? new_value : "", &len);
    if (!p) return false;
    free(v->data.string.ptr);
    v->data.string.ptr = p;
    v->data.string.len = len;
    v->version++;
    bindings_propagate(rt, v);
    return true;
}

bool saaam_component_set_slot(saaam_runtime_t* rt, saaam_value_t* component, saaam_type_t slot_type, saaam_value_t* injected) {
    if (!rt || !component || component->type != SAAAM_TYPE_COMPONENT) return false;
    if (!component->data.component.slots || component->data.component.slot_count != (size_t)SAAAM_TYPE_MAX) return false;
    if (slot_type <= SAAAM_TYPE_UNKNOWN || slot_type >= SAAAM_TYPE_MAX) return false;
    size_t idx = (size_t)slot_type;
    if (injected) saaam_retain_value(rt, injected);
    if (component->data.component.slots[idx]) saaam_release_value(rt, component->data.component.slots[idx]);
    component->data.component.slots[idx] = injected;
    component->version++;
    return true;
}

saaam_value_t* saaam_component_get_slot(const saaam_value_t* component, saaam_type_t slot_type) {
    if (!component || component->type != SAAAM_TYPE_COMPONENT) return NULL;
    if (!component->data.component.slots || component->data.component.slot_count != (size_t)SAAAM_TYPE_MAX) return NULL;
    if (slot_type <= SAAAM_TYPE_UNKNOWN || slot_type >= SAAAM_TYPE_MAX) return NULL;
    return component->data.component.slots[(size_t)slot_type];
}

static int saaam_stricmp_ascii(const char* a, const char* b) {
    if (a == b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        ca = (unsigned char)tolower(ca);
        cb = (unsigned char)tolower(cb);
        if (ca != cb) return (int)ca - (int)cb;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

bool saaam_can_morph(saaam_type_t from, saaam_type_t to) {
    if (from == to) return true;
    switch (from) {
        case SAAAM_TYPE_INT:
            return (to == SAAAM_TYPE_FLOAT || to == SAAAM_TYPE_STRING || to == SAAAM_TYPE_BOOL || to == SAAAM_TYPE_TERNARY);
        case SAAAM_TYPE_FLOAT:
            return (to == SAAAM_TYPE_INT || to == SAAAM_TYPE_STRING || to == SAAAM_TYPE_BOOL);
        case SAAAM_TYPE_STRING:
            return (to == SAAAM_TYPE_INT || to == SAAAM_TYPE_FLOAT || to == SAAAM_TYPE_BOOL);
        case SAAAM_TYPE_BOOL:
            return (to == SAAAM_TYPE_INT || to == SAAAM_TYPE_STRING || to == SAAAM_TYPE_TERNARY);
        case SAAAM_TYPE_TERNARY:
            return (to == SAAAM_TYPE_BOOL || to == SAAAM_TYPE_STRING || to == SAAAM_TYPE_INT);
        default:
            return false;
    }
}

static bool parse_bool_str(const char* s, bool* out) {
    if (!s || !out) return false;
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return false;
    if (saaam_stricmp_ascii(s, "true") == 0 || saaam_stricmp_ascii(s, "yes") == 0 ||
        saaam_stricmp_ascii(s, "on") == 0 || strcmp(s, "1") == 0) {
        *out = true;
        return true;
    }
    if (saaam_stricmp_ascii(s, "false") == 0 || saaam_stricmp_ascii(s, "no") == 0 ||
        saaam_stricmp_ascii(s, "off") == 0 || strcmp(s, "0") == 0) {
        *out = false;
        return true;
    }
    return false;
}

static void record_morph(saaam_value_t* v, saaam_type_t from, saaam_type_t to) {
    if (!v || !(v->flags & SAAAM_VALUE_FLAG_NEURAL)) return;
    saaam_morph_history_t* h = (saaam_morph_history_t*)malloc(sizeof(*h));
    if (!h) return;
    h->from_type = from;
    h->to_type = to;
    h->timestamp_ns = saaam_time_ns();
    h->next = v->morph_history;
    v->morph_history = h;
    v->morph_count++;
}

saaam_value_t* saaam_morph_value(saaam_runtime_t* rt, saaam_value_t* v, saaam_type_t target) {
    if (!rt || !v) return NULL;
    if (!(v->flags & SAAAM_VALUE_FLAG_NEURAL)) return NULL;
    if (!saaam_can_morph(v->type, target)) return NULL;
    if (v->type == target) return v;

    saaam_type_t from = v->type;
    record_morph(v, from, target);
    rt->stats.morph_operations++;

    if (target == SAAAM_TYPE_STRING) {
        char buf[256];
        buf[0] = '\0';
        switch (from) {
            case SAAAM_TYPE_INT:
                snprintf(buf, sizeof(buf), "%lld", (long long)v->data.i64);
                break;
            case SAAAM_TYPE_FLOAT:
                snprintf(buf, sizeof(buf), "%.17g", v->data.f64);
                break;
            case SAAAM_TYPE_BOOL:
                snprintf(buf, sizeof(buf), "%s", v->data.boolean ? "true" : "false");
                break;
            case SAAAM_TYPE_TERNARY:
                snprintf(buf, sizeof(buf), "%s",
                         (v->data.ternary == SAAAM_TERNARY_TRUE) ? "true" :
                         (v->data.ternary == SAAAM_TERNARY_FALSE) ? "false" : "unknown");
                break;
            default:
                return NULL;
        }
        if (from == SAAAM_TYPE_STRING) free(v->data.string.ptr);
        v->type = SAAAM_TYPE_STRING;
        v->data.string.ptr = saaam_strdup_impl(buf, &v->data.string.len);
        if (!v->data.string.ptr) {
            v->data.string.len = 0;
            v->type = from;
            return NULL;
        }
        v->version++;
        bindings_propagate(rt, v);
        return v;
    }

    if (from == SAAAM_TYPE_STRING && v->data.string.ptr) {
        const char* s = v->data.string.ptr;
        errno = 0;
        if (target == SAAAM_TYPE_INT) {
            char* end = NULL;
            long long val = strtoll(s, &end, 10);
            if (errno != 0 || end == s) return NULL;
            free(v->data.string.ptr);
            v->data.string.ptr = NULL;
            v->data.string.len = 0;
            v->type = SAAAM_TYPE_INT;
            v->data.i64 = (int64_t)val;
        } else if (target == SAAAM_TYPE_FLOAT) {
            char* end = NULL;
            double val = strtod(s, &end);
            if (errno != 0 || end == s) return NULL;
            free(v->data.string.ptr);
            v->data.string.ptr = NULL;
            v->data.string.len = 0;
            v->type = SAAAM_TYPE_FLOAT;
            v->data.f64 = val;
        } else if (target == SAAAM_TYPE_BOOL) {
            bool b = false;
            if (!parse_bool_str(s, &b)) return NULL;
            free(v->data.string.ptr);
            v->data.string.ptr = NULL;
            v->data.string.len = 0;
            v->type = SAAAM_TYPE_BOOL;
            v->data.boolean = b;
        } else {
            return NULL;
        }
        v->version++;
        bindings_propagate(rt, v);
        return v;
    }

    if (target == SAAAM_TYPE_INT) {
        if (from == SAAAM_TYPE_FLOAT) v->data.i64 = (int64_t)v->data.f64;
        else if (from == SAAAM_TYPE_BOOL) v->data.i64 = v->data.boolean ? 1 : 0;
        else if (from == SAAAM_TYPE_TERNARY) v->data.i64 = (v->data.ternary == SAAAM_TERNARY_TRUE) ? 1 :
                                                          (v->data.ternary == SAAAM_TERNARY_FALSE) ? 0 : -1;
        else return NULL;
        v->type = SAAAM_TYPE_INT;
    } else if (target == SAAAM_TYPE_FLOAT) {
        if (from == SAAAM_TYPE_INT) v->data.f64 = (double)v->data.i64;
        else if (from == SAAAM_TYPE_BOOL) v->data.f64 = v->data.boolean ? 1.0 : 0.0;
        else return NULL;
        v->type = SAAAM_TYPE_FLOAT;
    } else if (target == SAAAM_TYPE_BOOL) {
        if (from == SAAAM_TYPE_INT) v->data.boolean = (v->data.i64 != 0);
        else if (from == SAAAM_TYPE_FLOAT) v->data.boolean = (v->data.f64 != 0.0);
        else if (from == SAAAM_TYPE_TERNARY) v->data.boolean = (v->data.ternary == SAAAM_TERNARY_TRUE);
        else return NULL;
        v->type = SAAAM_TYPE_BOOL;
    } else if (target == SAAAM_TYPE_TERNARY) {
        if (!rt->enable_ternary_logic) return NULL;
        if (from == SAAAM_TYPE_BOOL) v->data.ternary = v->data.boolean ? SAAAM_TERNARY_TRUE : SAAAM_TERNARY_FALSE;
        else if (from == SAAAM_TYPE_INT) v->data.ternary = (v->data.i64 == 0) ? SAAAM_TERNARY_FALSE :
                                                    (v->data.i64 == 1) ? SAAAM_TERNARY_TRUE : SAAAM_TERNARY_UNKNOWN;
        else return NULL;
        v->type = SAAAM_TYPE_TERNARY;
    } else {
        return NULL;
    }

    v->version++;
    bindings_propagate(rt, v);
    return v;
}

saaam_ternary_t saaam_ternary_and(saaam_ternary_t a, saaam_ternary_t b) {
    if (a == SAAAM_TERNARY_FALSE || b == SAAAM_TERNARY_FALSE) return SAAAM_TERNARY_FALSE;
    if (a == SAAAM_TERNARY_TRUE && b == SAAAM_TERNARY_TRUE) return SAAAM_TERNARY_TRUE;
    return SAAAM_TERNARY_UNKNOWN;
}

saaam_ternary_t saaam_ternary_or(saaam_ternary_t a, saaam_ternary_t b) {
    if (a == SAAAM_TERNARY_TRUE || b == SAAAM_TERNARY_TRUE) return SAAAM_TERNARY_TRUE;
    if (a == SAAAM_TERNARY_FALSE && b == SAAAM_TERNARY_FALSE) return SAAAM_TERNARY_FALSE;
    return SAAAM_TERNARY_UNKNOWN;
}

saaam_ternary_t saaam_ternary_not(saaam_ternary_t a) {
    if (a == SAAAM_TERNARY_TRUE) return SAAAM_TERNARY_FALSE;
    if (a == SAAAM_TERNARY_FALSE) return SAAAM_TERNARY_TRUE;
    return SAAAM_TERNARY_UNKNOWN;
}

bool saaam_synapse_bind(saaam_runtime_t* rt, saaam_value_t* left, saaam_value_t* right) {
    if (!rt || !left || !right) return false;

    if (rt->binding_capacity == rt->binding_count) {
        size_t new_cap = rt->binding_capacity ? rt->binding_capacity * 2 : 128;
        void* p = realloc(rt->bindings, new_cap * sizeof(rt->bindings[0]));
        if (!p) return false;
        rt->bindings = (saaam_binding_t*)p;
        rt->binding_capacity = new_cap;
    }

    saaam_binding_t b;
    memset(&b, 0, sizeof(b));
    b.a = left;
    b.a_id = left->instance_id;
    b.a_version_seen = left->version;
    b.b = right;
    b.b_id = right->instance_id;
    b.b_version_seen = right->version;
    rt->bindings[rt->binding_count++] = b;

    // Initial sync: right mirrors left.
    (void)value_assign_scalar(rt, right, left);
    return true;
}

saaam_value_t* saaam_synapse_flow(saaam_runtime_t* rt, saaam_value_t* input, saaam_value_t* function) {
    if (!rt || !input || !function) return NULL;
    if (function->type != SAAAM_TYPE_FUNCTION || !function->data.function.fn) return NULL;
    return function->data.function.fn(rt, input, function->data.function.user_data);
}

bool saaam_synapse_inject(saaam_runtime_t* rt, saaam_value_t* dependency, saaam_value_t* target) {
    if (!rt || !dependency || !target) return false;
    if (target->type != SAAAM_TYPE_COMPONENT) return false;
    return saaam_component_set_slot(rt, target, dependency->type, dependency);
}

void saaam_gc_collect(saaam_runtime_t* rt) {
    if (!rt) return;
    rt->stats.gc_cycles++;

    // RC-based sweep: free GC values that have been released to 0.
    // Cycles require higher-level handling (or a future cycle collector).
    size_t i = 0;
    while (i < rt->registry_gc.count) {
        saaam_value_t* v = rt->registry_gc.items[i];
        if (v && v->ref_count == 0) {
            registry_remove(&rt->registry_all, v, &v->registry_index, true);
            registry_remove(&rt->registry_gc, v, &v->gc_index, false);
            value_finalize_nested(rt, v);
            free(v);
            continue;
        }
        i++;
    }
}

bool saaam_emit_event(saaam_runtime_t* rt, saaam_event_handler_t handler, void* data) {
    if (!rt || !handler) return false;
    if (rt->event_count == rt->event_capacity) {
        size_t new_cap = rt->event_capacity ? rt->event_capacity * 2 : 256;
        void* p = realloc(rt->events, new_cap * sizeof(rt->events[0]));
        if (!p) return false;
        rt->events = (saaam_event_t*)p;
        rt->event_capacity = new_cap;
    }
    rt->events[rt->event_count++] = (saaam_event_t){ .handler = handler, .data = data };
    rt->stats.events_emitted++;
    return true;
}

size_t saaam_process_events(saaam_runtime_t* rt) {
    if (!rt) return 0;
    size_t processed = 0;
    for (size_t i = 0; i < rt->event_count; i++) {
        saaam_event_t ev = rt->events[i];
        ev.handler(rt, ev.data);
        processed++;
    }
    rt->event_count = 0;
    rt->stats.events_processed += processed;
    return processed;
}

void saaam_profile_start(saaam_runtime_t* rt, const char* operation) {
    if (!rt || !operation) return;
    if (rt->profile_depth >= (sizeof(rt->profile_stack) / sizeof(rt->profile_stack[0]))) return;
    rt->profile_stack[rt->profile_depth++] = (saaam_profile_frame_t){ .operation = operation, .start_ns = saaam_time_ns() };
}

void saaam_profile_end(saaam_runtime_t* rt, const char* operation) {
    if (!rt || !operation) return;
    if (rt->profile_depth == 0) return;
    saaam_profile_frame_t frame = rt->profile_stack[--rt->profile_depth];
    uint64_t dur = saaam_time_ns() - frame.start_ns;
    fprintf(stdout, "PROFILE %s: %llu ns\n", operation, (unsigned long long)dur);
}

#if !SAAAM_WITH_CUDA
bool saaam_cuda_available(void) { return false; }
bool saaam_cuda_check_device(void) { return false; }
bool saaam_cuda_ternary_matmul(const int8_t* weights, const float* input, float* output, int M, int N, int K) {
    (void)weights; (void)input; (void)output; (void)M; (void)N; (void)K;
    return false;
}
bool saaam_cuda_concept_attention(const float* queries, const float* keys, const float* values, float* output, int num_concepts, int embedding_dim) {
    (void)queries; (void)keys; (void)values; (void)output; (void)num_concepts; (void)embedding_dim;
    return false;
}
bool saaam_cuda_activation_inplace(float* data, size_t n, saaam_activation_t act) {
    (void)data; (void)n; (void)act;
    return false;
}
#endif
