/*
SAAAM Language - Native Runtime API

No external deps. Pure C/CUDA (optional) execution substrate.
*/

#ifndef SAAAM_NATIVE_RUNTIME_H
#define SAAAM_NATIVE_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- ABI / Export ----

#define SAAAM_RUNTIME_ABI_VERSION 1u

#if defined(_WIN32) && !defined(SAAAM_STATIC)
  #if defined(SAAAM_BUILD_DLL)
    #define SAAAM_API __declspec(dllexport)
  #else
    #define SAAAM_API __declspec(dllimport)
  #endif
#else
  #define SAAAM_API
#endif

// ---- Core Types ----

typedef enum {
    SAAAM_TYPE_UNKNOWN = 0,
    SAAAM_TYPE_INT,
    SAAAM_TYPE_FLOAT,
    SAAAM_TYPE_STRING,
    SAAAM_TYPE_BOOL,
    SAAAM_TYPE_NEURAL,
    SAAAM_TYPE_TERNARY,
    SAAAM_TYPE_ARRAY,
    SAAAM_TYPE_COMPONENT,
    SAAAM_TYPE_FUNCTION,
    SAAAM_TYPE_MAX
} saaam_type_t;

typedef enum {
    SAAAM_MEMORY_STACK = 0,
    SAAAM_MEMORY_HEAP,
    SAAAM_MEMORY_NEURAL,
    SAAAM_MEMORY_GPU,
    SAAAM_MEMORY_ARENA,
    SAAAM_MEMORY_GC
} saaam_memory_region_t;

typedef enum {
    SAAAM_TERNARY_TRUE = 1,
    SAAAM_TERNARY_FALSE = 0,
    SAAAM_TERNARY_UNKNOWN = -1
} saaam_ternary_t;

typedef enum {
    SAAAM_VALUE_FLAG_NEURAL   = 1u << 0,
    SAAAM_VALUE_FLAG_MUTABLE  = 1u << 1,
    SAAAM_VALUE_FLAG_MARKED   = 1u << 2,
    SAAAM_VALUE_FLAG_ON_GPU   = 1u << 3
} saaam_value_flags_t;

typedef struct saaam_morph_history {
    saaam_type_t from_type;
    saaam_type_t to_type;
    uint64_t timestamp_ns;
    struct saaam_morph_history* next;
} saaam_morph_history_t;

typedef struct saaam_runtime saaam_runtime_t;

typedef struct saaam_value saaam_value_t;

typedef saaam_value_t* (*saaam_native_fn_t)(saaam_runtime_t* runtime, saaam_value_t* input, void* user_data);
typedef void (*saaam_native_fn_dtor_t)(void* user_data);

typedef struct saaam_value {
    saaam_type_t type;
    saaam_memory_region_t memory_region;
    uint32_t flags;

    uint64_t instance_id;  // unique per allocation
    uint64_t version;      // increments on mutation

    size_t ref_count;
    size_t registry_index; // SIZE_MAX when unregistered
    size_t gc_index;       // SIZE_MAX when not in GC list

    saaam_morph_history_t* morph_history;
    uint32_t morph_count;

    void* gpu_data;
    size_t gpu_bytes;

    struct saaam_value* free_next; // freelist linkage for pool regions

    union {
        int64_t i64;
        double f64;
        bool boolean;
        saaam_ternary_t ternary;
        void* ptr;

        struct {
            char* ptr;
            size_t len;
        } string;

        struct {
            saaam_value_t** elements; // retained references
            size_t length;
            size_t capacity;
        } array;

        struct {
            saaam_native_fn_t fn;
            void* user_data;
            saaam_native_fn_dtor_t dtor;
        } function;

        struct {
            saaam_value_t** slots; // indexed by saaam_type_t
            size_t slot_count;     // == SAAAM_TYPE_MAX
        } component;
    } data;
} saaam_value_t;

// ---- Runtime ----

typedef void (*saaam_event_handler_t)(saaam_runtime_t* runtime, void* data);

typedef struct {
    uint64_t morph_operations;
    uint64_t gc_cycles;
    uint64_t gpu_transfers;
    uint64_t events_emitted;
    uint64_t events_processed;
} saaam_runtime_stats_t;

SAAAM_API saaam_runtime_t* saaam_runtime_init(void);
SAAAM_API void saaam_runtime_destroy(saaam_runtime_t* runtime);

SAAAM_API void saaam_runtime_set_ternary_logic(saaam_runtime_t* runtime, bool enabled);
SAAAM_API void saaam_runtime_set_gpu_acceleration(saaam_runtime_t* runtime, bool enabled);
SAAAM_API void saaam_runtime_set_neuroplastic_optimization(saaam_runtime_t* runtime, bool enabled);

SAAAM_API saaam_runtime_stats_t saaam_runtime_get_stats(const saaam_runtime_t* runtime);
SAAAM_API void saaam_print_performance_stats(const saaam_runtime_t* runtime);

// ---- Values / Memory ----

SAAAM_API saaam_value_t* saaam_alloc_value(saaam_runtime_t* runtime, saaam_type_t type, saaam_memory_region_t region);
SAAAM_API void saaam_retain_value(saaam_runtime_t* runtime, saaam_value_t* value);
SAAAM_API void saaam_release_value(saaam_runtime_t* runtime, saaam_value_t* value);

// Back-compat alias: "free" means release a reference.
SAAAM_API void saaam_free_value(saaam_runtime_t* runtime, saaam_value_t* value);

SAAAM_API saaam_value_t* saaam_create_int(saaam_runtime_t* runtime, int64_t value);
SAAAM_API saaam_value_t* saaam_create_float(saaam_runtime_t* runtime, double value);
SAAAM_API saaam_value_t* saaam_create_string(saaam_runtime_t* runtime, const char* value);
SAAAM_API saaam_value_t* saaam_create_bool(saaam_runtime_t* runtime, bool value);
SAAAM_API saaam_value_t* saaam_create_neural(saaam_runtime_t* runtime, saaam_type_t initial_type);
SAAAM_API saaam_value_t* saaam_create_function(saaam_runtime_t* runtime, saaam_native_fn_t fn, void* user_data, saaam_native_fn_dtor_t dtor);
SAAAM_API saaam_value_t* saaam_create_component(saaam_runtime_t* runtime);

SAAAM_API bool saaam_value_set_int(saaam_runtime_t* runtime, saaam_value_t* value, int64_t new_value);
SAAAM_API bool saaam_value_set_float(saaam_runtime_t* runtime, saaam_value_t* value, double new_value);
SAAAM_API bool saaam_value_set_bool(saaam_runtime_t* runtime, saaam_value_t* value, bool new_value);
SAAAM_API bool saaam_value_set_string(saaam_runtime_t* runtime, saaam_value_t* value, const char* new_value);

SAAAM_API bool saaam_component_set_slot(saaam_runtime_t* runtime, saaam_value_t* component, saaam_type_t slot_type, saaam_value_t* injected);
SAAAM_API saaam_value_t* saaam_component_get_slot(const saaam_value_t* component, saaam_type_t slot_type);

// ---- Neuroplastic Ops ----

SAAAM_API bool saaam_can_morph(saaam_type_t from, saaam_type_t to);
SAAAM_API saaam_value_t* saaam_morph_value(saaam_runtime_t* runtime, saaam_value_t* value, saaam_type_t target_type);

// ---- Ternary Logic ----

SAAAM_API saaam_ternary_t saaam_ternary_and(saaam_ternary_t a, saaam_ternary_t b);
SAAAM_API saaam_ternary_t saaam_ternary_or(saaam_ternary_t a, saaam_ternary_t b);
SAAAM_API saaam_ternary_t saaam_ternary_not(saaam_ternary_t a);

// ---- Synapse Ops ----

SAAAM_API bool saaam_synapse_bind(saaam_runtime_t* runtime, saaam_value_t* left, saaam_value_t* right);
SAAAM_API saaam_value_t* saaam_synapse_flow(saaam_runtime_t* runtime, saaam_value_t* input, saaam_value_t* function);
SAAAM_API bool saaam_synapse_inject(saaam_runtime_t* runtime, saaam_value_t* dependency, saaam_value_t* target);

// ---- GC ----

SAAAM_API void saaam_gc_collect(saaam_runtime_t* runtime);

// ---- Events ----

SAAAM_API bool saaam_emit_event(saaam_runtime_t* runtime, saaam_event_handler_t handler, void* data);
SAAAM_API size_t saaam_process_events(saaam_runtime_t* runtime);

// ---- Profiling ----

SAAAM_API void saaam_profile_start(saaam_runtime_t* runtime, const char* operation);
SAAAM_API void saaam_profile_end(saaam_runtime_t* runtime, const char* operation);

// ---- CUDA Kernel Library (optional) ----

typedef enum {
    SAAAM_ACT_RELU = 0,
    SAAAM_ACT_TANH = 1,
    SAAAM_ACT_GELU = 2,
    SAAAM_ACT_SILU = 3,
    SAAAM_ACT_RMSNORM_SILU = 4
} saaam_activation_t;

SAAAM_API bool saaam_cuda_available(void);
SAAAM_API bool saaam_cuda_check_device(void);

SAAAM_API bool saaam_cuda_ternary_matmul(const int8_t* weights, const float* input, float* output, int M, int N, int K);
SAAAM_API bool saaam_cuda_concept_attention(const float* queries, const float* keys, const float* values, float* output, int num_concepts, int embedding_dim);
SAAAM_API bool saaam_cuda_activation_inplace(float* data, size_t n, saaam_activation_t act);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SAAAM_NATIVE_RUNTIME_H
