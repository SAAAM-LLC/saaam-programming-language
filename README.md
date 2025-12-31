███████╗ █████╗  █████╗  █████╗ ███╗   ███╗
██╔════╝██╔══██╗██╔══██╗██╔══██╗████╗ ████║
███████╗███████║███████║███████║██╔████╔██║
╚════██║██╔══██║██╔══██║██╔══██║██║╚██╔╝██║
███████║██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝

# THE SAAAM LANGUAGE SPECIFICATION v1.0

## PHILOSOPHY

SAAAM is not just another programming language.

We took:
- **Python's** elegance and readability
- **React's** reactive component model
- **TypeScript's** type safety and inference
- **C++'s** raw performance and control
- **Rust's** ownership and fearless concurrency

And FUSED them into something that transcends them all.

---

## CORE CONCEPTS

### 1. NEUROPLASTIC TYPING
Types in SAAAM aren't static boxes - they're living, breathing entities that EVOLVE.

```saaam
# Types can morph based on context
neural x = 42              # Inferred as Int
x ~> "hello"               # Morphs to String (neuroplastic assignment)
x = 3.14                   # Morphs to Float

# Strict typing when you need it  
strict y: Int = 42
y = "hello"                # ERROR! Strict types don't morph
```

### 2. SYNAPSE OPERATORS
Special operators that create neural-like connections between code components.

```saaam
# ~> Morph operator: transforms and assigns
value ~> transform()

# <=> Bidirectional binding (React-style reactivity)
input <=> display

# -> Flow operator: pipes data through
data -> filter() -> map() -> reduce()

# => Lambda/arrow (like JS/TS)
fn = (x) => x * 2

# |> Parallel pipe: executes in parallel
tasks |> [process_a, process_b, process_c]

# <~ Reverse flow: pulls data
result <~ async_source()

# :: Namespace/trait implementation
MyType :: Trait

# @> Inject operator: dependency injection
service @> container
```

### 3. COMPONENT-FIRST ARCHITECTURE
Everything is a component. Not just UI - ALL code.

```saaam
component Counter {
    # State (reactive by default)
    state count: Int = 0
    
    # Props (immutable inputs)
    props {
        initial: Int = 0
        step: Int = 1
    }
    
    # Lifecycle hooks
    on mount {
        count = initial
    }
    
    # Methods
    fn increment() {
        count += step
    }
    
    # Render (optional - for UI components)
    render {
        <div>
            <span>{count}</span>
            <button on:click={increment}>+</button>
        </div>
    }
}
```

### 4. OWNERSHIP & BORROWING (Rust-inspired, Python-friendly)

```saaam
# Ownership by default
let data = Vector::new()
process(data)              # data MOVES to process
# data is no longer valid here

# Borrowing with &
let data = Vector::new()
read(&data)                # Immutable borrow
data.push(42)              # Still valid!

# Mutable borrow with &mut
modify(&mut data)

# Escape hatch: gc keyword for garbage-collected references
gc shared_data = Vector::new()  # GC-managed, Python-style
```

### 5. PATTERN MATCHING ON STEROIDS

```saaam
match value {
    0 => "zero"
    1..=10 => "small"
    n if n < 0 => "negative"
    Int(x) where x > 100 => "big: {x}"
    String(s) if s.starts_with("test") => "test string"
    [first, ...rest] => "array with {first}"
    {name, age} => "person: {name}, {age}"
    Some(x) => "got {x}"
    None => "nothing"
    _ => "default"
}
```

### 6. NEURAL BLOCKS
Code segments that can learn and adapt.

```saaam
neural block Optimizer {
    # Weights that persist and evolve
    weights: Tensor<f32>
    
    # Training mode
    train mode {
        fn forward(input: Tensor) -> Tensor {
            weights @ input -> activate()
        }
        
        fn backward(grad: Tensor) {
            weights -= learning_rate * grad
        }
    }
    
    # Inference mode (optimized, no gradients)
    infer mode {
        fn forward(input: Tensor) -> Tensor {
            weights @ input -> activate()
        }
    }
}
```

### 7. ASYNC/CONCURRENCY PRIMITIVES

```saaam
# Async functions
async fn fetch_data(url: String) -> Result<Data, Error> {
    response <~ http.get(url)
    return response.json()
}

# Parallel execution
parallel {
    task1 = compute_a()
    task2 = compute_b()
    task3 = compute_c()
}
# All three run simultaneously

# Channels (Go-style)
chan messages: Channel<String>

spawn {
    messages.send("hello")
}

msg <~ messages.recv()

# Actor model
actor Counter {
    state count = 0
    
    receive Increment => count += 1
    receive Decrement => count -= 1
    receive GetCount(reply) => reply.send(count)
}
```

### 8. MACROS & METAPROGRAMMING

```saaam
# Declarative macros
macro vec!($($elem:expr),*) {
    {
        let mut temp = Vector::new()
        $(temp.push($elem))*
        temp
    }
}

# Procedural macros (compile-time code generation)
#[derive(Serialize, Deserialize)]
struct User {
    name: String
    age: Int
}

# Quote/unquote for AST manipulation
fn generate_getter(field: Ident) -> AST {
    quote {
        fn get_$(field)(&self) -> &Self::$(field)Type {
            &self.$(field)
        }
    }
}
```

### 9. EFFECTS SYSTEM

```saaam
# Declare what effects a function can have
fn pure_compute(x: Int) -> Int {
    x * 2  # No effects allowed
}

fn io_function(path: String) -> String with IO {
    File.read(path)  # IO effect
}

fn stateful_op() with State<Counter> {
    state.count += 1
}

# Combine effects
fn complex() with IO, State<Config>, Async {
    config <~ load_config()
    data <~ fetch(config.url)
    state.update(data)
}
```

### 10. FIRST-CLASS TYPES

```saaam
# Types are values you can manipulate
let MyType = struct {
    x: Int
    y: Int
}

let ListOf = (T: Type) => struct {
    items: Array<T>
    length: Int
}

let IntList = ListOf(Int)

# Runtime type introspection
fn describe(value: Any) -> String {
    match type_of(value) {
        Int => "integer"
        String => "string"
        Array<T> => "array of {T.name}"
        _ => "unknown"
    }
}
```

---

## 📝 SYNTAX REFERENCE

### Variables & Constants
```saaam
let x = 42              # Immutable by default
var y = 42              # Mutable
const PI = 3.14159      # Compile-time constant
neural z = 0            # Neuroplastic (can change type)
gc shared = data        # Garbage-collected
strict s: Int = 42      # Strict type, no inference changes
```

### Functions
```saaam
# Basic function
fn add(a: Int, b: Int) -> Int {
    a + b
}

# Generic function
fn first<T>(list: Array<T>) -> T {
    list[0]
}

# With default parameters
fn greet(name: String = "World") -> String {
    "Hello, {name}!"
}

# Variadic
fn sum(...nums: Int) -> Int {
    nums.reduce(0, +)
}

# Lambda
let double = (x) => x * 2

# Multi-expression lambda
let process = (x) => {
    let y = x * 2
    let z = y + 1
    z
}
```

### Control Flow
```saaam
# If expression (returns value)
let result = if x > 0 { "positive" } else { "non-positive" }

# Match expression
let msg = match status {
    200 => "OK"
    404 => "Not Found"
    500 => "Server Error"
    _ => "Unknown"
}

# Loops
for item in collection {
    process(item)
}

for i in 0..10 {
    print(i)
}

while condition {
    do_work()
}

loop {
    if done { break }
}

# Comprehensions
let squares = [x * x for x in 1..10]
let evens = [x for x in nums if x % 2 == 0]
let mapping = {k: v * 2 for k, v in pairs}
```

### Types
```saaam
# Primitive types
Int, Int8, Int16, Int32, Int64
UInt, UInt8, UInt16, UInt32, UInt64  
Float, Float32, Float64
Bool
Char
String

# Compound types
Array<T>
Map<K, V>
Set<T>
Tuple<A, B, C>
Option<T>           # Some(value) or None
Result<T, E>        # Ok(value) or Err(error)

# Custom types
struct Point {
    x: Float
    y: Float
}

enum Color {
    Red
    Green  
    Blue
    RGB(r: Int, g: Int, b: Int)
    Hex(String)
}

# Type aliases
type ID = String
type Handler = fn(Request) -> Response
```

### Traits (Interfaces)
```saaam
trait Drawable {
    fn draw(&self)
    fn bounds(&self) -> Rect
    
    # Default implementation
    fn center(&self) -> Point {
        self.bounds().center()
    }
}

# Implementation
impl Drawable for Circle {
    fn draw(&self) {
        # drawing logic
    }
    
    fn bounds(&self) -> Rect {
        Rect::from_center(self.center, self.radius * 2)
    }
}

# Trait bounds
fn render<T: Drawable>(item: T) {
    item.draw()
}

# Multiple bounds
fn process<T: Serialize + Clone + Debug>(item: T) {
    # ...
}
```

### Error Handling
```saaam
# Result-based (Rust-style)
fn divide(a: Float, b: Float) -> Result<Float, DivisionError> {
    if b == 0.0 {
        Err(DivisionError::DivideByZero)
    } else {
        Ok(a / b)
    }
}

# The ? operator for propagation
fn compute() -> Result<Float, Error> {
    let x = parse_float(input)?
    let y = divide(x, 2.0)?
    Ok(y * 10.0)
}

# Try-catch for compatibility
try {
    risky_operation()
} catch e: NetworkError {
    handle_network_error(e)
} catch e: IOError {
    handle_io_error(e)
} finally {
    cleanup()
}
```

### Modules
```saaam
# File: math/vector.saaam
module math::vector

pub struct Vector3 {
    pub x: Float
    pub y: Float
    pub z: Float
}

pub fn dot(a: Vector3, b: Vector3) -> Float {
    a.x * b.x + a.y * b.y + a.z * b.z
}

# Private function
fn internal_helper() { }

# File: main.saaam
use math::vector::{Vector3, dot}
use std::io::*
use external_crate as ext
```

---

## 🚀 MEMORY MODEL

### Ownership Rules
1. Each value has exactly ONE owner
2. When owner goes out of scope, value is dropped
3. Values can be MOVED or BORROWED

### Memory Regions
```saaam
# Stack (default for small types)
let x = 42

# Heap (explicit or automatic for large types)
let data = heap Vector::new()

# Arena allocation
arena game_frame {
    let entities = Vector::new()
    let particles = Vector::new()
    # All dropped when arena ends
}

# GC region (for complex graphs)
gc region shared_state {
    let graph = Graph::new()
    let nodes = graph.nodes  # Can have cycles
}
```

---

## ⚡ PERFORMANCE HINTS

```saaam
# Inline hint
#[inline]
fn fast_add(a: Int, b: Int) -> Int { a + b }

# Force inline
#[inline(always)]
fn critical_path(x: Int) -> Int { x * 2 }

# SIMD hint
#[simd]
fn vector_multiply(a: Vec4, b: Vec4) -> Vec4 {
    a * b
}

# GPU compute
#[gpu]
fn matrix_multiply(a: Matrix, b: Matrix) -> Matrix {
    # Runs on GPU
}

# Compile-time evaluation
const fn factorial(n: Int) -> Int {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

# Cache hint
#[cache(size = 64)]
struct CacheAligned {
    data: Array<Float, 16>
}
```

---

## 🌐 BUILT-IN DOMAINS

### Neural/ML First-Class Support
```saaam
use neural::*

let model = Sequential [
    Dense(784, 256, activation: ReLU)
    Dropout(0.3)
    Dense(256, 10, activation: Softmax)
]

train model with optimizer: Adam(lr: 0.001) {
    for batch in data_loader {
        let pred = model.forward(batch.x)
        let loss = cross_entropy(pred, batch.y)
        loss.backward()
        optimizer.step()
    }
}
```

### Reactive UI
```saaam
use ui::*

component App {
    state items: Array<String> = []
    state input: String = ""
    
    fn add_item() {
        if input.len() > 0 {
            items.push(input)
            input = ""
        }
    }
    
    render {
        <div class="app">
            <input bind:value={input} on:enter={add_item}/>
            <button on:click={add_item}>Add</button>
            <ul>
                {for item in items {
                    <li>{item}</li>
                }}
            </ul>
        </div>
    }
}
```

### Game Engine Primitives
```saaam
use engine::*

entity Player {
    component Transform { position: Vec3, rotation: Quat }
    component Physics { velocity: Vec3, mass: Float }
    component Sprite { texture: Texture }
    
    system Movement(dt: Float) {
        physics.velocity += gravity * dt
        transform.position += physics.velocity * dt
    }
}
```

---

## 📂 FILE EXTENSION

All SAAAM source files use the `.saaam` extension.

```
project/
├── main.saaam
├── lib/
│   ├── utils.saaam
│   └── types.saaam
├── components/
│   ├── button.saaam
│   └── input.saaam
└── tests/
    └── test_main.saaam
```

---

## 🔧 COMPILATION TARGETS

SAAAM compiles to:
- Native machine code (via LLVM)
- WebAssembly
- JavaScript (for browser)
- Python (for interop)
- CUDA (for GPU kernels)

---
