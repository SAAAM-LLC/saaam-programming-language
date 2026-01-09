#!/usr/bin/env python3
"""
SAAAM Parser Test Suite
Let's make sure this neural beast WORKS!
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokens import TokenType
from lexer import Lexer, tokenize
from ast_nodes import *
from parser import Parser, parse, parse_expression_only, print_ast, format_expression, validate_syntax, is_complete_input


def test_basic_expressions():
    """Test basic expression parsing."""
    print("🧪 Testing basic expressions...")

    # Integer literals
    ast = parse_expression_only("42")
    assert isinstance(ast, IntegerLiteral), f"Expected IntegerLiteral, got {type(ast)}"
    assert ast.value == 42

    # Float literals
    ast = parse_expression_only("3.14")
    assert isinstance(ast, FloatLiteral)
    assert ast.value == 3.14

    # String literals
    ast = parse_expression_only('"hello world"')
    assert isinstance(ast, StringLiteral)
    assert ast.value == "hello world"

    # Boolean literals
    ast = parse_expression_only("true")
    assert isinstance(ast, BoolLiteral)
    assert ast.value == True

    ast = parse_expression_only("false")
    assert isinstance(ast, BoolLiteral)
    assert ast.value == False

    # None literal
    ast = parse_expression_only("None")
    assert isinstance(ast, NoneLiteral)

    print("  ✅ Basic expressions passed!")


def test_binary_operations():
    """Test binary operator parsing."""
    print("🧪 Testing binary operations...")

    # Simple addition
    ast = parse_expression_only("1 + 2")
    assert isinstance(ast, BinaryOp)
    assert ast.operator == "+"
    assert isinstance(ast.left, IntegerLiteral)
    assert isinstance(ast.right, IntegerLiteral)

    # Precedence test: 1 + 2 * 3 should parse as 1 + (2 * 3)
    ast = parse_expression_only("1 + 2 * 3")
    assert isinstance(ast, BinaryOp)
    assert ast.operator == "+"
    assert isinstance(ast.right, BinaryOp)
    assert ast.right.operator == "*"

    # SAAAM synapse operators
    ast = parse_expression_only("x ~> transform()")
    assert isinstance(ast, BinaryOp)
    assert ast.operator == "~>"

    # Bidirectional binding
    ast = parse_expression_only("a <=> b")
    assert isinstance(ast, BinaryOp)
    assert ast.operator == "<=>"

    print("  ✅ Binary operations passed!")


def test_unary_operations():
    """Test unary operator parsing."""
    print("🧪 Testing unary operations...")

    ast = parse_expression_only("-42")
    assert isinstance(ast, UnaryOp)
    assert ast.operator == "-"

    ast = parse_expression_only("!true")
    assert isinstance(ast, UnaryOp)
    assert ast.operator == "!"

    ast = parse_expression_only("&x")
    assert isinstance(ast, UnaryOp)
    assert ast.operator == "&"

    ast = parse_expression_only("&var x")
    assert isinstance(ast, UnaryOp)
    assert ast.operator == "&mut"

    ast = parse_expression_only("*ptr")
    assert isinstance(ast, UnaryOp)
    assert ast.operator == "*"

    print("  ✅ Unary operations passed!")


def test_function_calls():
    """Test function call parsing."""
    print("🧪 Testing function calls...")

    ast = parse_expression_only("foo()")
    assert isinstance(ast, Call)
    assert isinstance(ast.callee, Identifier)
    assert ast.callee.name == "foo"
    assert len(ast.arguments) == 0

    ast = parse_expression_only("add(1, 2)")
    assert isinstance(ast, Call)
    assert len(ast.arguments) == 2

    # Method call
    ast = parse_expression_only("obj.method()")
    assert isinstance(ast, Call)
    assert isinstance(ast.callee, Member)

    # Chained calls
    ast = parse_expression_only("a.b().c()")
    assert isinstance(ast, Call)

    print("  ✅ Function calls passed!")


def test_array_and_map():
    """Test array and map literal parsing."""
    print("🧪 Testing arrays and maps...")

    # Empty array
    ast = parse_expression_only("[]")
    assert isinstance(ast, ArrayLiteral)
    assert len(ast.elements) == 0

    # Array with elements
    ast = parse_expression_only("[1, 2, 3]")
    assert isinstance(ast, ArrayLiteral)
    assert len(ast.elements) == 3

    # List comprehension
    ast = parse_expression_only("[x * 2 for x in items]")
    assert isinstance(ast, Comprehension)
    assert ast.variable == "x"

    # Comprehension with condition
    ast = parse_expression_only("[x for x in nums if x > 0]")
    assert isinstance(ast, Comprehension)
    assert ast.condition is not None

    print("  ✅ Arrays and maps passed!")


def test_tuples():
    """Test tuple parsing."""
    print("🧪 Testing tuples...")

    # Empty tuple
    ast = parse_expression_only("()")
    assert isinstance(ast, TupleLiteral)
    assert len(ast.elements) == 0

    # Single-element tuple (with trailing comma)
    ast = parse_expression_only("(1, 2, 3)")
    assert isinstance(ast, TupleLiteral)
    assert len(ast.elements) == 3

    # Grouped expression (not a tuple)
    ast = parse_expression_only("(1 + 2)")
    assert isinstance(ast, BinaryOp)

    print("  ✅ Tuples passed!")


def test_if_expression():
    """Test if expression parsing."""
    print("🧪 Testing if expressions...")

    code = "if x > 0 { 1 } else { -1 }"
    ast = parse_expression_only(code)
    assert isinstance(ast, IfExpr)
    assert isinstance(ast.condition, BinaryOp)
    assert ast.else_branch is not None

    print("  ✅ If expressions passed!")


def test_match_expression():
    """Test match expression parsing."""
    print("🧪 Testing match expressions...")

    code = """match value {
        0 => "zero"
        1 => "one"
        _ => "other"
    }"""
    ast = parse_expression_only(code)
    assert isinstance(ast, MatchExpr)
    assert len(ast.arms) == 3

    print("  ✅ Match expressions passed!")


def test_lambda():
    """Test lambda parsing."""
    print("🧪 Testing lambdas...")

    ast = parse_expression_only("fn(x) => x * 2")
    assert isinstance(ast, Lambda)
    assert len(ast.params) == 1
    assert ast.params[0].name == "x"

    ast = parse_expression_only("fn(a, b) => a + b")
    assert isinstance(ast, Lambda)
    assert len(ast.params) == 2

    print("  ✅ Lambdas passed!")


def test_jsx():
    """Test JSX parsing."""
    print("🧪 Testing JSX...")

    ast = parse_expression_only("<div/>")
    assert isinstance(ast, JSXElement)
    assert ast.tag == "div"
    assert ast.is_self_closing

    ast = parse_expression_only('<button on={click}>Click</button>')
    assert isinstance(ast, JSXElement)
    assert ast.tag == "button"
    assert not ast.is_self_closing
    assert len(ast.attributes) == 1

    print("  ✅ JSX passed!")


def test_option_result():
    """Test Option/Result type constructors."""
    print("🧪 Testing Option/Result...")

    ast = parse_expression_only("Some(42)")
    assert isinstance(ast, ConstructExpr)
    assert ast.type_name.name == "Some"

    ast = parse_expression_only("Ok(value)")
    assert isinstance(ast, ConstructExpr)
    assert ast.type_name.name == "Ok"

    ast = parse_expression_only('Err("failed")')
    assert isinstance(ast, ConstructExpr)
    assert ast.type_name.name == "Err"

    print("  ✅ Option/Result passed!")


def test_range():
    """Test range expression parsing."""
    print("🧪 Testing ranges...")

    ast = parse_expression_only("1..10")
    assert isinstance(ast, RangeExpr)
    assert not ast.inclusive

    ast = parse_expression_only("0..=100")
    assert isinstance(ast, RangeExpr)
    assert ast.inclusive

    print("  ✅ Ranges passed!")


def test_try_operator():
    """Test ? operator parsing."""
    print("🧪 Testing try operator...")

    ast = parse_expression_only("result?")
    assert isinstance(ast, TryExpr)

    ast = parse_expression_only("foo()?.bar()?")
    assert isinstance(ast, TryExpr)

    print("  ✅ Try operator passed!")


def test_statements():
    """Test statement parsing."""
    print("🧪 Testing statements...")

    # Variable declarations
    prog = parse("let x = 42")
    assert len(prog.declarations) == 1
    assert isinstance(prog.declarations[0], VarDecl)
    assert prog.declarations[0].name == "x"
    assert not prog.declarations[0].is_mutable

    prog = parse("var y: Int = 0")
    decl = prog.declarations[0]
    assert decl.is_mutable
    assert decl.var_type is not None

    # Neural variable
    prog = parse("neural z = 10")
    assert prog.declarations[0].is_neural

    # GC variable
    prog = parse("gc data = vec")
    assert prog.declarations[0].is_gc

    print("  ✅ Statements passed!")


def test_function_declaration():
    """Test function declaration parsing."""
    print("🧪 Testing function declarations...")

    code = """
    fn add(a: Int, b: Int) -> Int {
        a + b
    }
    """
    prog = parse(code)
    assert len(prog.declarations) == 1
    fn = prog.declarations[0]
    assert isinstance(fn, FunctionDecl)
    assert fn.name == "add"
    assert len(fn.params) == 2
    assert fn.return_type is not None

    # Generic function
    code = """
    fn first<T>(list: Array<T>) -> T {
        list[0]
    }
    """
    prog = parse(code)
    fn = prog.declarations[0]
    assert len(fn.type_params) == 1
    assert fn.type_params[0] == "T"

    # Async function
    code = """
    async fn fetch(url: String) -> Result<Data, Error> {
        response <~ http.get(url)
    }
    """
    prog = parse(code)
    fn = prog.declarations[0]
    assert fn.is_async

    print("  ✅ Function declarations passed!")


def test_struct_declaration():
    """Test struct declaration parsing."""
    print("🧪 Testing struct declarations...")

    code = """
    struct Point {
        x: Float,
        y: Float,
    }
    """
    prog = parse(code)
    struct = prog.declarations[0]
    assert isinstance(struct, StructDecl)
    assert struct.name == "Point"
    assert len(struct.fields) == 2

    # Generic struct
    code = """
    struct Container<T> {
        value: T,
        count: Int,
    }
    """
    prog = parse(code)
    struct = prog.declarations[0]
    assert len(struct.type_params) == 1

    print("  ✅ Struct declarations passed!")


def test_enum_declaration():
    """Test enum declaration parsing."""
    print("🧪 Testing enum declarations...")

    code = """
    enum Color {
        Red,
        Green,
        Blue,
        RGB(r: Int, g: Int, b: Int),
    }
    """
    prog = parse(code)
    enum = prog.declarations[0]
    assert isinstance(enum, EnumDecl)
    assert enum.name == "Color"
    assert len(enum.variants) == 4
    assert enum.variants[3].name == "RGB"
    assert len(enum.variants[3].fields) == 3

    print("  ✅ Enum declarations passed!")


def test_trait_declaration():
    """Test trait declaration parsing."""
    print("🧪 Testing trait declarations...")

    code = """
    trait Drawable {
        fn draw(&self)
        fn bounds(&self) -> Rect
    }
    """
    prog = parse(code)
    trait = prog.declarations[0]
    assert isinstance(trait, TraitDecl)
    assert trait.name == "Drawable"
    assert len(trait.methods) == 2

    # Trait with super traits
    code = """
    trait Foo: Bar + Baz {
        fn method(&self)
    }
    """
    prog = parse(code)
    trait = prog.declarations[0]
    assert len(trait.super_traits) == 2

    print("  ✅ Trait declarations passed!")


def test_impl_declaration():
    """Test impl block parsing."""
    print("🧪 Testing impl blocks...")

    code = """
    impl Point {
        fn new(x: Float, y: Float) -> Point {
            Point { x: x, y: y }
        }
    }
    """
    prog = parse(code)
    impl = prog.declarations[0]
    assert isinstance(impl, ImplDecl)
    assert impl.trait_name is None

    # Trait impl
    code = """
    impl Drawable for Circle {
        fn draw(&self) {
            draw_circle(self.center, self.radius)
        }
    }
    """
    prog = parse(code)
    impl = prog.declarations[0]
    assert impl.trait_name == "Drawable"

    print("  ✅ Impl blocks passed!")


def test_component_declaration():
    """Test component declaration parsing."""
    print("🧪 Testing component declarations...")

    code = """
    component Counter {
        state count: Int = 0

        fn increment() {
            count += 1
        }

        render {
            <div>{count}</div>
        }
    }
    """
    prog = parse(code)
    comp = prog.declarations[0]
    assert isinstance(comp, ComponentDecl)
    assert comp.name == "Counter"
    assert len(comp.state) == 1
    assert len(comp.methods) == 1
    assert comp.render is not None

    print("  ✅ Component declarations passed!")


def test_imports():
    """Test import statement parsing."""
    print("🧪 Testing imports...")

    prog = parse("use std::io")
    assert len(prog.imports) == 1
    imp = prog.imports[0]
    assert imp.path == ["std", "io"]

    prog = parse("use std::io::*")
    assert prog.imports[0].is_wildcard

    prog = parse("use std::io::{File, Read, Write}")
    assert len(prog.imports[0].items) == 3

    prog = parse("use external_crate as ext")
    assert prog.imports[0].alias == "ext"

    print("  ✅ Imports passed!")


def test_control_flow():
    """Test control flow statement parsing."""
    print("🧪 Testing control flow...")

    # For loop
    code = """
    for i in 0..10 {
        print(i)
    }
    """
    prog = parse(code)
    stmt = prog.declarations[0]
    assert isinstance(stmt, ForStmt)

    # While loop
    code = """
    while x < 10 {
        x += 1
    }
    """
    prog = parse(code)
    stmt = prog.declarations[0]
    assert isinstance(stmt, WhileStmt)

    # Loop
    code = """
    loop {
        if done { break }
    }
    """
    prog = parse(code)
    stmt = prog.declarations[0]
    assert isinstance(stmt, LoopStmt)

    print("  ✅ Control flow passed!")


def test_pattern_matching():
    """Test pattern parsing."""
    print("🧪 Testing patterns...")

    code = """
    match opt {
        Some(x) => x,
        None => 0,
    }
    """
    ast = parse_expression_only(code)
    assert isinstance(ast, MatchExpr)
    assert len(ast.arms) == 2

    # Struct pattern
    code = """
    match point {
        Point { x, y } => x + y,
        _ => 0,
    }
    """
    ast = parse_expression_only(code)

    print("  ✅ Patterns passed!")


def test_concurrency():
    """Test concurrency construct parsing."""
    print("🧪 Testing concurrency...")

    # Spawn
    code = """
    spawn {
        heavy_computation()
    }
    """
    prog = parse(code)
    stmt = prog.declarations[0]
    assert isinstance(stmt, SpawnStmt)

    # Parallel
    code = """
    parallel {
        a = compute_a()
        b = compute_b()
    }
    """
    prog = parse(code)
    stmt = prog.declarations[0]
    assert isinstance(stmt, ParallelStmt)
    assert len(stmt.tasks) == 2

    print("  ✅ Concurrency passed!")


def test_error_handling():
    """Test try-catch parsing."""
    print("🧪 Testing error handling...")

    code = """
    try {
        risky()
    } catch e: Error {
        handle(e)
    } finally {
        cleanup()
    }
    """
    prog = parse(code)
    stmt = prog.declarations[0]
    assert isinstance(stmt, TryStmt)
    assert len(stmt.catches) == 1
    assert stmt.finally_block is not None

    print("  ✅ Error handling passed!")


def test_ast_utilities():
    """Test AST utility functions."""
    print("🧪 Testing AST utilities...")

    # Test print_ast
    ast = parse_expression_only("1 + 2 * 3")
    output = print_ast(ast)
    assert "BinaryOp" in output

    # Test format_expression
    ast = parse_expression_only("foo(1, 2)")
    formatted = format_expression(ast)
    assert "foo" in formatted
    assert "1" in formatted

    # Test validate_syntax
    errors = validate_syntax("let x = 42")
    assert len(errors) == 0

    errors = validate_syntax("let x = ")
    assert len(errors) > 0

    # Test is_complete_input
    assert is_complete_input("let x = 42")
    assert not is_complete_input("let x = {")
    assert not is_complete_input('"unclosed string')

    print("  ✅ AST utilities passed!")


def test_complete_program():
    """Test parsing a complete SAAAM program."""
    print("🧪 Testing complete program...")

    code = '''
module myapp::core

use std::io::{File, Read}
use std::collections::HashMap

component Counter {
    state count: Int = 0

    props {
        let initial: Int = 0
        let step: Int = 1
    }

    on mount {
        count = initial
    }

    fn increment() {
        count += step
    }

    fn decrement() {
        count -= step
    }

    render {
        <div class="counter">
            <span>{count}</span>
            <button on={increment}>+</button>
            <button on={decrement}>-</button>
        </div>
    }
}

struct Point {
    x: Float,
    y: Float,
}

impl Point {
    fn new(x: Float, y: Float) -> Point {
        Point { x: x, y: y }
    }

    fn distance(&self, other: &Point) -> Float {
        let dx = self.x - other.x
        let dy = self.y - other.y
        (dx * dx + dy * dy) -> sqrt()
    }
}

enum Status {
    Active,
    Inactive,
    Pending(String),
}

trait Serialize {
    fn to_json(&self) -> String
}

async fn fetch_data(url: String) -> String {
    let response <~ http.get(url)
    response.json()
}

fn main() {
    let points = [Point::new(0.0, 0.0), Point::new(3.0, 4.0)]
    let dist = points[0].distance(&points[1])
    print("Distance: {dist}")

    parallel {
        data1 = fetch_data("http://api.example.com/1")
        data2 = fetch_data("http://api.example.com/2")
    }

    match dist {
        0.0 => print("Same point!")
        d if d < 1.0 => print("Very close")
        d if d < 10.0 => print("Nearby")
        _ => print("Far away")
    }
}
'''

    prog = parse(code)
    assert prog.module_path == "myapp::core"
    assert len(prog.imports) == 2
    assert len(prog.declarations) > 5

    print("  ✅ Complete program passed!")


def run_all_tests():
    """Run all parser tests."""
    print("\n" + "="*60)
    print("🔥 SAAAM PARSER TEST SUITE 🔥")
    print("="*60 + "\n")

    tests = [
        test_basic_expressions,
        test_binary_operations,
        test_unary_operations,
        test_function_calls,
        test_array_and_map,
        test_tuples,
        test_if_expression,
        test_match_expression,
        test_lambda,
        test_jsx,
        test_option_result,
        test_range,
        test_try_operator,
        test_statements,
        test_function_declaration,
        test_struct_declaration,
        test_enum_declaration,
        test_trait_declaration,
        test_impl_declaration,
        test_component_declaration,
        test_imports,
        test_control_flow,
        test_pattern_matching,
        test_concurrency,
        test_error_handling,
        test_ast_utilities,
        test_complete_program,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {test.__name__}")
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! THE NEURAL NETWORK IS FIRING! 🎉\n")
    else:
        print(f"\n⚠️  {failed} test(s) need attention.\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
