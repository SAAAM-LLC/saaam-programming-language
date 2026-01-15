/*
SAAAM Native Runtime - C Test Suite
Builds without external dependencies.
*/

#include "saaam_native_runtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void test_morphing(void) {
    saaam_runtime_t* rt = saaam_runtime_init();
    assert(rt);

    saaam_value_t* v = saaam_create_neural(rt, SAAAM_TYPE_INT);
    assert(v);
    assert(v->type == SAAAM_TYPE_INT);
    assert(v->memory_region == SAAAM_MEMORY_NEURAL);
    assert((v->flags & SAAAM_VALUE_FLAG_NEURAL) != 0);

    assert(saaam_value_set_int(rt, v, 42));
    assert(saaam_morph_value(rt, v, SAAAM_TYPE_STRING));
    assert(v->type == SAAAM_TYPE_STRING);
    assert(v->data.string.ptr);
    assert(strcmp(v->data.string.ptr, "42") == 0);

    assert(saaam_morph_value(rt, v, SAAAM_TYPE_FLOAT));
    assert(v->type == SAAAM_TYPE_FLOAT);
    assert(fabs(v->data.f64 - 42.0) < 1e-9);

    assert(saaam_morph_value(rt, v, SAAAM_TYPE_BOOL));
    assert(v->type == SAAAM_TYPE_BOOL);
    assert(v->data.boolean == true);

    assert(v->morph_count >= 3);

    saaam_runtime_destroy(rt);
}

static void test_binding(void) {
    saaam_runtime_t* rt = saaam_runtime_init();
    assert(rt);

    saaam_value_t* a = saaam_create_neural(rt, SAAAM_TYPE_INT);
    saaam_value_t* b = saaam_create_neural(rt, SAAAM_TYPE_INT);
    assert(a && b);

    assert(saaam_value_set_int(rt, a, 7));
    assert(saaam_synapse_bind(rt, a, b));

    // initial sync
    assert(b->type == SAAAM_TYPE_INT);
    assert(b->data.i64 == 7);

    // propagate a -> b
    assert(saaam_value_set_int(rt, a, 123));
    assert(b->data.i64 == 123);

    // propagate b -> a
    assert(saaam_value_set_int(rt, b, -9));
    assert(a->data.i64 == -9);

    saaam_runtime_destroy(rt);
}

static saaam_value_t* fn_double_int(saaam_runtime_t* rt, saaam_value_t* input, void* user_data) {
    (void)user_data;
    if (!rt || !input) return NULL;
    if (input->type != SAAAM_TYPE_INT) return NULL;
    return saaam_create_int(rt, input->data.i64 * 2);
}

static void test_flow(void) {
    saaam_runtime_t* rt = saaam_runtime_init();
    assert(rt);

    saaam_value_t* f = saaam_create_function(rt, fn_double_int, NULL, NULL);
    saaam_value_t* x = saaam_create_int(rt, 21);
    assert(f && x);

    saaam_value_t* y = saaam_synapse_flow(rt, x, f);
    assert(y);
    assert(y->type == SAAAM_TYPE_INT);
    assert(y->data.i64 == 42);

    saaam_runtime_destroy(rt);
}

static void test_inject(void) {
    saaam_runtime_t* rt = saaam_runtime_init();
    assert(rt);

    saaam_value_t* c = saaam_create_component(rt);
    saaam_value_t* dep = saaam_create_string(rt, "service");
    assert(c && dep);

    assert(saaam_synapse_inject(rt, dep, c));
    saaam_value_t* got = saaam_component_get_slot(c, SAAAM_TYPE_STRING);
    assert(got == dep);

    saaam_runtime_destroy(rt);
}

static void on_event(saaam_runtime_t* rt, void* data) {
    (void)rt;
    int* counter = (int*)data;
    (*counter)++;
}

static void test_events(void) {
    saaam_runtime_t* rt = saaam_runtime_init();
    assert(rt);

    int counter = 0;
    assert(saaam_emit_event(rt, on_event, &counter));
    assert(saaam_emit_event(rt, on_event, &counter));
    size_t processed = saaam_process_events(rt);
    assert(processed == 2);
    assert(counter == 2);

    saaam_runtime_destroy(rt);
}

static void test_gc(void) {
    saaam_runtime_t* rt = saaam_runtime_init();
    assert(rt);

    saaam_value_t* g = saaam_alloc_value(rt, SAAAM_TYPE_STRING, SAAAM_MEMORY_GC);
    assert(g);
    assert(g->memory_region == SAAAM_MEMORY_GC);
    assert(g->type == SAAAM_TYPE_STRING);

    assert(saaam_value_set_string(rt, g, "ephemeral"));
    saaam_release_value(rt, g); // eligible for collection

    size_t freed = saaam_gc_collect(rt);
    assert(freed >= 1);

    saaam_runtime_destroy(rt);
}

int main(void) {
    test_morphing();
    test_binding();
    test_flow();
    test_inject();
    test_events();
    test_gc();

    printf("OK: native runtime tests passed\n");
    return 0;
}

