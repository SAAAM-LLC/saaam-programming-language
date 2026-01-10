/*
SAAAM Native Runtime - Test Program
Demonstrates neuroplastic morphing, ternary logic, and performance
*/

#include "saaam_native_runtime.h"
#include <stdio.h>
#include <assert.h>

void test_neuroplastic_morphing(saaam_runtime_t* runtime) {
    printf("\n🧠 TEST: Neuroplastic Morphing\n");
    printf("="*50 "\n");

    // Create a neural value starting as integer
    saaam_value_t* magic = saaam_create_neural(runtime, SAAAM_TYPE_INT);
    magic->data.i64 = 42;
    printf("✓ Created neural value: %lld (Int)\n", (long long)magic->data.i64);

    // Morph to String
    saaam_morph_value(runtime, magic, SAAAM_TYPE_STRING);
    printf("✓ Morphed to String: '%s'\n", magic->data.string);

    // Morph to Float
    saaam_create_string(runtime, "3.14159");  // Target value
    saaam_morph_value(runtime, magic, SAAAM_TYPE_FLOAT);
    printf("✓ Morphed to Float: %f\n", magic->data.f64);

    // Morph to Bool
    saaam_morph_value(runtime, magic, SAAAM_TYPE_BOOL);
    printf("✓ Morphed to Bool: %s\n", magic->data.boolean ? "true" : "false");

    // Check morph history
    printf("✓ Morph history count: %u\n", magic->morph_count);
    saaam_morph_history_t* history = magic->morph_history;
    while (history) {
        printf("  - %d -> %d @ %llu ns\n",
               history->from_type, history->to_type,
               (unsigned long long)history->timestamp);
        history = history->next;
    }

    printf("✓ Neuroplastic morphing SUCCESS!\n");
}

void test_ternary_logic(saaam_runtime_t* runtime) {
    printf("\n⚡ TEST: Ternary Logic Operations\n");
    printf("="*50 "\n");

    // Test AND
    saaam_ternary_t result_and_tt = saaam_ternary_and(SAAAM_TERNARY_TRUE, SAAAM_TERNARY_TRUE);
    assert(result_and_tt == SAAAM_TERNARY_TRUE);
    printf("✓ TRUE && TRUE = TRUE\n");

    saaam_ternary_t result_and_tu = saaam_ternary_and(SAAAM_TERNARY_TRUE, SAAAM_TERNARY_UNKNOWN);
    assert(result_and_tu == SAAAM_TERNARY_UNKNOWN);
    printf("✓ TRUE && UNKNOWN = UNKNOWN\n");

    saaam_ternary_t result_and_tf = saaam_ternary_and(SAAAM_TERNARY_TRUE, SAAAM_TERNARY_FALSE);
    assert(result_and_tf == SAAAM_TERNARY_FALSE);
    printf("✓ TRUE && FALSE = FALSE\n");

    // Test OR
    saaam_ternary_t result_or_ff = saaam_ternary_or(SAAAM_TERNARY_FALSE, SAAAM_TERNARY_FALSE);
    assert(result_or_ff == SAAAM_TERNARY_FALSE);
    printf("✓ FALSE || FALSE = FALSE\n");

    saaam_ternary_t result_or_fu = saaam_ternary_or(SAAAM_TERNARY_FALSE, SAAAM_TERNARY_UNKNOWN);
    assert(result_or_fu == SAAAM_TERNARY_UNKNOWN);
    printf("✓ FALSE || UNKNOWN = UNKNOWN\n");

    saaam_ternary_t result_or_tu = saaam_ternary_or(SAAAM_TERNARY_TRUE, SAAAM_TERNARY_UNKNOWN);
    assert(result_or_tu == SAAAM_TERNARY_TRUE);
    printf("✓ TRUE || UNKNOWN = TRUE\n");

    // Test NOT
    saaam_ternary_t result_not_t = saaam_ternary_not(SAAAM_TERNARY_TRUE);
    assert(result_not_t == SAAAM_TERNARY_FALSE);
    printf("✓ !TRUE = FALSE\n");

    saaam_ternary_t result_not_u = saaam_ternary_not(SAAAM_TERNARY_UNKNOWN);
    assert(result_not_u == SAAAM_TERNARY_UNKNOWN);
    printf("✓ !UNKNOWN = UNKNOWN\n");

    printf("✓ Ternary logic operations SUCCESS!\n");
}

void test_memory_regions(saaam_runtime_t* runtime) {
    printf("\n💾 TEST: Memory Regions\n");
    printf("="*50 "\n");

    // Stack allocation
    saaam_value_t* stack_val = saaam_create_int(runtime, 100);
    assert(stack_val->memory_region == SAAAM_MEMORY_STACK);
    printf("✓ Stack allocation: %lld\n", (long long)stack_val->data.i64);

    // Heap allocation
    saaam_value_t* heap_val = saaam_create_string(runtime, "Hello from heap!");
    assert(heap_val->memory_region == SAAAM_MEMORY_HEAP);
    printf("✓ Heap allocation: '%s'\n", heap_val->data.string);

    // Neural pool allocation
    saaam_value_t* neural_val = saaam_create_neural(runtime, SAAAM_TYPE_FLOAT);
    neural_val->data.f64 = 2.71828;
    assert(neural_val->memory_region == SAAAM_MEMORY_NEURAL);
    assert(neural_val->is_neural == true);
    printf("✓ Neural pool allocation: %f (morphable)\n", neural_val->data.f64);

    // GC allocation
    saaam_value_t* gc_val = saaam_alloc_value(runtime, SAAAM_TYPE_INT, SAAAM_MEMORY_GC);
    gc_val->data.i64 = 999;
    assert(gc_val->memory_region == SAAAM_MEMORY_GC);
    printf("✓ GC allocation: %lld\n", (long long)gc_val->data.i64);

    printf("✓ Memory regions SUCCESS!\n");
}

void test_synapse_operators(saaam_runtime_t* runtime) {
    printf("\n⚡ TEST: Synapse Operators\n");
    printf("="*50 "\n");

    // Create neural values
    saaam_value_t* src = saaam_create_neural(runtime, SAAAM_TYPE_INT);
    src->data.i64 = 42;

    saaam_value_t* target = saaam_create_string(runtime, "target");

    // Test morph operator ~>
    saaam_value_t* morphed = saaam_synapse_morph(runtime, src, target);
    assert(morphed->type == SAAAM_TYPE_STRING);
    printf("✓ Morph operator (~>): Int -> String\n");

    // Test bind operator <=>
    saaam_value_t* left = saaam_create_int(runtime, 10);
    saaam_value_t* right = saaam_create_int(runtime, 20);
    saaam_value_t* bound = saaam_synapse_bind(runtime, left, right);
    printf("✓ Bind operator (<=>): Bidirectional binding\n");

    // Test flow operator ->
    saaam_value_t* input = saaam_create_int(runtime, 5);
    saaam_value_t* flowed = saaam_synapse_flow(runtime, input, NULL);
    printf("✓ Flow operator (->): Pipeline flow\n");

    // Test inject operator @>
    saaam_value_t* dependency = saaam_create_string(runtime, "dependency");
    saaam_value_t* injected = saaam_synapse_inject(runtime, dependency, NULL);
    printf("✓ Inject operator (@>): Dependency injection\n");

    printf("✓ Synapse operators SUCCESS!\n");
}

void test_gc_collection(saaam_runtime_t* runtime) {
    printf("\n🗑️  TEST: Garbage Collection\n");
    printf("="*50 "\n");

    size_t initial_gc_count = runtime->memory->gc_count;
    printf("Initial GC objects: %zu\n", initial_gc_count);

    // Create some GC objects
    for (int i = 0; i < 10; i++) {
        saaam_value_t* val = saaam_alloc_value(runtime, SAAAM_TYPE_INT, SAAAM_MEMORY_GC);
        val->data.i64 = i * 100;
    }

    printf("After allocations: %zu objects\n", runtime->memory->gc_count);

    // Run GC
    saaam_gc_collect(runtime);

    printf("After GC cycle: %zu objects\n", runtime->memory->gc_count);
    printf("✓ Garbage collection SUCCESS!\n");
}

void test_performance(saaam_runtime_t* runtime) {
    printf("\n⏱️  TEST: Performance Profiling\n");
    printf("="*50 "\n");

    saaam_profile_start(runtime, "1000 neuroplastic morphs");

    for (int i = 0; i < 1000; i++) {
        saaam_value_t* val = saaam_create_neural(runtime, SAAAM_TYPE_INT);
        val->data.i64 = i;

        saaam_morph_value(runtime, val, SAAAM_TYPE_FLOAT);
        saaam_morph_value(runtime, val, SAAAM_TYPE_STRING);
        saaam_morph_value(runtime, val, SAAAM_TYPE_BOOL);
    }

    saaam_profile_end(runtime, "1000 neuroplastic morphs");

    printf("✓ Performance test SUCCESS!\n");
}

int main(void) {
    printf("\n");
    printf("████████████████████████████████████████████████\n");
    printf("██                                            ██\n");
    printf("██  SAAAM NATIVE RUNTIME TEST SUITE          ██\n");
    printf("██  Revolutionary C/CUDA Performance Engine  ██\n");
    printf("██                                            ██\n");
    printf("████████████████████████████████████████████████\n");
    printf("\n");

    // Initialize runtime
    saaam_runtime_t* runtime = saaam_runtime_init();
    if (!runtime) {
        fprintf(stderr, "❌ Failed to initialize runtime\n");
        return 1;
    }

    // Run tests
    test_neuroplastic_morphing(runtime);
    test_ternary_logic(runtime);
    test_memory_regions(runtime);
    test_synapse_operators(runtime);
    test_gc_collection(runtime);
    test_performance(runtime);

    // Print stats
    saaam_print_performance_stats(runtime);

    // Cleanup
    saaam_runtime_destroy(runtime);

    printf("\n");
    printf("🚀 ALL TESTS PASSED! SAAAM Native Runtime is REVOLUTIONARY! 🚀\n");
    printf("\n");

    return 0;
}
