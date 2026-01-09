"""
SAAAM Language - Runtime System
The EXECUTION ENGINE: Where code becomes REALITY!

This runtime handles:
- Neuroplastic type morphing at runtime 🧠
- Synapse operator execution ⚡
- Component reactivity system 🔄
- Ownership/GC hybrid memory management 💾
- Concurrency (async/spawn/parallel) 🚀
- Neural block execution 🤖
"""

import asyncio
import threading
import weakref
import gc
import time
from typing import Any, Dict, List, Set, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import copy
import json
from concurrent.futures import ThreadPoolExecutor, Future

from .ast_nodes import *
from .semantic_analyzer import SAAMType, TypeKind, BuiltinTypes


class RuntimeError(Exception):
    """When the neural pathways short-circuit during execution."""
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}" if location else message)


class MemoryRegion(Enum):
    """Memory management regions in SAAAM."""
    STACK = auto()      # Stack allocated (default for small types)
    HEAP = auto()       # Heap allocated (owned)
    GC = auto()         # Garbage collected
    ARENA = auto()      # Arena allocated
    NEURAL = auto()     # Neural memory (can morph)


@dataclass
class SAAMValue:
    """
    A runtime value in SAAAM - can MORPH, EVOLVE, and be REACTIVE! 🧠
    """
    data: Any
    type: SAAMType
    region: MemoryRegion = MemoryRegion.STACK
    
    # Neuroplastic properties 🧠
    is_neural: bool = False
    morph_history: List[SAAMType] = field(default_factory=list)
    
    # Ownership properties
    is_owned: bool = True
    is_borrowed: bool = False
    borrow_count: int = 0
    
    # Reactivity properties
    is_reactive: bool = False
    observers: Set['ReactiveObserver'] = field(default_factory=set)
    
    # Memory management
    ref_count: int = 1
    mark_bit: bool = False  # For GC marking
    
    def __post_init__(self):
        if self.is_neural:
            self.region = MemoryRegion.NEURAL
    
    def morph_to(self, new_type: SAAMType, new_data: Any = None) -> 'SAAMValue':
        """NEUROPLASTIC MORPHING! 🧠⚡"""
        if not self.is_neural:
            raise RuntimeError("Cannot morph non-neural value")
        
        if not self.type.can_morph_to(new_type):
            raise RuntimeError(f"Cannot morph {self.type} to {new_type}")
        
        # Record morph history
        self.morph_history.append(self.type)
        
        # Perform the morph
        old_data = self.data
        self.type = new_type
        
        if new_data is not None:
            self.data = new_data
        else:
            # Try to convert data
            self.data = self._convert_data(old_data, new_type)
        
        # Notify observers of the change
        self._notify_observers()
        
        return self
    
    def _convert_data(self, data: Any, target_type: SAAMType) -> Any:
        """Convert data during morphing."""
        if target_type.name == "String":
            return str(data)
        elif target_type.name == "Int":
            return int(float(str(data))) if isinstance(data, str) else int(data)
        elif target_type.name == "Float":
            return float(data)
        elif target_type.name == "Bool":
            return bool(data)
        else:
            return data
    
    def borrow(self, mutable: bool = False) -> 'SAAMReference':
        """Create a borrow reference."""
        if self.is_borrowed and mutable:
            raise RuntimeError("Cannot create mutable borrow while other borrows exist")
        
        self.borrow_count += 1
        return SAAMReference(self, mutable)
    
    def add_observer(self, observer: 'ReactiveObserver'):
        """Add a reactive observer."""
        self.observers.add(observer)
        self.is_reactive = True
    
    def remove_observer(self, observer: 'ReactiveObserver'):
        """Remove a reactive observer."""
        self.observers.discard(observer)
        if not self.observers:
            self.is_reactive = False
    
    def _notify_observers(self):
        """Notify all observers of changes."""
        for observer in list(self.observers):  # Copy to avoid modification during iteration
            observer.on_value_changed(self)
    
    def clone(self) -> 'SAAMValue':
        """Create a deep clone."""
        new_value = SAAMValue(
            data=copy.deepcopy(self.data),
            type=copy.deepcopy(self.type),
            region=self.region,
            is_neural=self.is_neural,
            morph_history=self.morph_history.copy(),
            is_reactive=False,  # Don't copy reactivity
            observers=set()
        )
        return new_value
    
    def __str__(self) -> str:
        return f"SAAMValue({self.data}: {self.type})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SAAMReference:
    """A borrowed reference to a SAAAM value."""
    target: SAAMValue
    is_mutable: bool = False
    
    def __del__(self):
        """Decrease borrow count when reference is dropped."""
        if hasattr(self, 'target'):
            self.target.borrow_count -= 1


class ReactiveObserver(ABC):
    """Observer for reactive value changes."""
    
    @abstractmethod
    def on_value_changed(self, value: SAAMValue):
        """Called when a reactive value changes."""
        pass


class ComponentObserver(ReactiveObserver):
    """Observer for component state changes."""
    
    def __init__(self, component: 'ComponentInstance', state_name: str):
        self.component = component
        self.state_name = state_name
    
    def on_value_changed(self, value: SAAMValue):
        """Trigger component re-render on state change."""
        self.component.mark_for_update()


class MemoryManager:
    """
    HYBRID MEMORY MANAGEMENT! 🧠💾
    Combines Rust-style ownership with garbage collection.
    """
    
    def __init__(self):
        self.heap_objects: Set[SAAMValue] = set()
        self.gc_objects: Set[SAAMValue] = set()
        self.arena_stacks: List[List[SAAMValue]] = []
        
        # GC state
        self.gc_threshold = 1000
        self.gc_count = 0
        
        # Neural memory pool
        self.neural_pool: List[SAAMValue] = []
    
    def allocate(self, data: Any, saaam_type: SAAMType, region: MemoryRegion = MemoryRegion.STACK) -> SAAMValue:
        """Allocate memory for a value."""
        value = SAAMValue(
            data=data,
            type=saaam_type,
            region=region,
            is_neural=(region == MemoryRegion.NEURAL or saaam_type.is_neural)
        )
        
        if region == MemoryRegion.HEAP:
            self.heap_objects.add(value)
        elif region == MemoryRegion.GC:
            self.gc_objects.add(value)
            value.is_owned = False
        elif region == MemoryRegion.NEURAL:
            self.neural_pool.append(value)
        elif region == MemoryRegion.ARENA and self.arena_stacks:
            self.arena_stacks[-1].append(value)
        
        self.gc_count += 1
        if self.gc_count >= self.gc_threshold:
            self.run_gc()
        
        return value
    
    def push_arena(self):
        """Push a new arena frame."""
        self.arena_stacks.append([])
    
    def pop_arena(self):
        """Pop arena frame and deallocate all objects."""
        if self.arena_stacks:
            arena = self.arena_stacks.pop()
            # All arena objects are automatically freed
    
    def run_gc(self):
        """Run garbage collection."""
        # Mark phase
        for obj in self.gc_objects:
            obj.mark_bit = False
        
        # Sweep phase (simplified - would need root marking in real implementation)
        live_objects = set()
        for obj in self.gc_objects:
            if obj.ref_count > 0 or obj.is_reactive:
                obj.mark_bit = True
                live_objects.add(obj)
        
        # Free unmarked objects
        self.gc_objects = live_objects
        self.gc_count = 0
    
    def deallocate(self, value: SAAMValue):
        """Explicitly deallocate a value."""
        if value.region == MemoryRegion.HEAP:
            self.heap_objects.discard(value)
        elif value.region == MemoryRegion.GC:
            self.gc_objects.discard(value)
        elif value.region == MemoryRegion.NEURAL:
            try:
                self.neural_pool.remove(value)
            except ValueError:
                pass


class AsyncRuntime:
    """Async runtime for SAAAM concurrency."""
    
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.channels: Dict[str, 'Channel'] = {}
        self.actors: Dict[str, 'Actor'] = {}
    
    async def spawn_task(self, coro):
        """Spawn an async task."""
        return asyncio.create_task(coro)
    
    def spawn_thread(self, func, *args):
        """Spawn a thread."""
        return self.executor.submit(func, *args)
    
    def create_channel(self, name: str, capacity: int = 0) -> 'Channel':
        """Create a new channel."""
        channel = Channel(capacity)
        self.channels[name] = channel
        return channel
    
    def get_channel(self, name: str) -> Optional['Channel']:
        """Get a channel by name."""
        return self.channels.get(name)


class Channel:
    """Go-style channel for async communication."""
    
    def __init__(self, capacity: int = 0):
        self.capacity = capacity
        self.queue = asyncio.Queue(maxsize=capacity)
    
    async def send(self, value: SAAMValue):
        """Send a value through the channel."""
        await self.queue.put(value)
    
    async def recv(self) -> SAAMValue:
        """Receive a value from the channel."""
        return await self.queue.get()
    
    def try_send(self, value: SAAMValue) -> bool:
        """Try to send without blocking."""
        try:
            self.queue.put_nowait(value)
            return True
        except asyncio.QueueFull:
            return False
    
    def try_recv(self) -> Optional[SAAMValue]:
        """Try to receive without blocking."""
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


class Actor:
    """Actor model implementation."""
    
    def __init__(self, name: str):
        self.name = name
        self.mailbox = Channel()
        self.state: Dict[str, SAAMValue] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    async def send(self, message_type: str, data: Any):
        """Send a message to this actor."""
        message = SAAMValue(
            data={"type": message_type, "data": data},
            type=SAAMType(TypeKind.MAP, "Message")
        )
        await self.mailbox.send(message)
    
    async def run(self):
        """Run the actor message loop."""
        self.running = True
        while self.running:
            message = await self.mailbox.recv()
            msg_data = message.data
            msg_type = msg_data.get("type")
            
            if msg_type in self.message_handlers:
                handler = self.message_handlers[msg_type]
                await handler(msg_data.get("data"), message)
    
    def stop(self):
        """Stop the actor."""
        self.running = False


class ComponentInstance:
    """Runtime instance of a component."""
    
    def __init__(self, component_decl: ComponentDecl, props: Dict[str, SAAMValue]):
        self.decl = component_decl
        self.props = props
        self.state: Dict[str, SAAMValue] = {}
        self.methods: Dict[str, Callable] = {}
        self.needs_update = False
        self.rendered_jsx = None
        
        # Initialize state with observers
        for state_var in component_decl.state:
            initial_value = props.get(state_var.name, SAAMValue(None, BuiltinTypes.NONE))
            initial_value.is_reactive = True
            
            # Add observer for re-rendering
            observer = ComponentObserver(self, state_var.name)
            initial_value.add_observer(observer)
            
            self.state[state_var.name] = initial_value
    
    def mark_for_update(self):
        """Mark component for re-render."""
        self.needs_update = True
    
    def render(self):
        """Render the component."""
        if self.decl.render and self.needs_update:
            # Would execute JSX rendering here
            self.needs_update = False
            return f"<{self.decl.name}>{json.dumps({k: str(v) for k, v in self.state.items()})}</{self.decl.name}>"
        return self.rendered_jsx


class NeuralBlock:
    """Runtime neural block for ML operations."""
    
    def __init__(self, decl: NeuralBlockDecl):
        self.decl = decl
        self.weights: Dict[str, SAAMValue] = {}
        self.mode = "infer"  # "train" or "infer"
        
        # Initialize weights
        for weight_var in decl.weights:
            # Create neural tensor (simplified)
            weight_data = [0.0] * 100  # Placeholder
            weight_value = SAAMValue(
                data=weight_data,
                type=SAAMType(TypeKind.NEURAL, "Tensor"),
                region=MemoryRegion.NEURAL,
                is_neural=True
            )
            self.weights[weight_var.name] = weight_value
    
    def set_mode(self, mode: str):
        """Set training or inference mode."""
        if mode not in {"train", "infer"}:
            raise RuntimeError(f"Invalid neural block mode: {mode}")
        self.mode = mode
    
    def forward(self, inputs: List[SAAMValue]) -> List[SAAMValue]:
        """Forward pass through neural block."""
        # Simplified neural computation
        # In reality, this would involve tensor operations
        outputs = []
        for inp in inputs:
            # Apply weights (simplified)
            result_data = inp.data * 0.5  # Placeholder computation
            result = SAAMValue(
                data=result_data,
                type=inp.type,
                region=MemoryRegion.NEURAL,
                is_neural=True
            )
            outputs.append(result)
        
        return outputs
    
    def backward(self, gradients: List[SAAMValue]):
        """Backward pass for training."""
        if self.mode != "train":
            raise RuntimeError("Cannot run backward pass in inference mode")
        
        # Update weights based on gradients (simplified)
        for weight_name, weight in self.weights.items():
            if isinstance(weight.data, list):
                for i in range(len(weight.data)):
                    weight.data[i] -= 0.01 * (gradients[0].data if gradients else 0.1)


class Environment:
    """Execution environment - symbol table at runtime."""
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.parent = parent
        self.bindings: Dict[str, SAAMValue] = {}
        self.is_function_env = False
        self.is_component_env = False
    
    def define(self, name: str, value: SAAMValue):
        """Define a new binding."""
        self.bindings[name] = value
    
    def get(self, name: str) -> Optional[SAAMValue]:
        """Get a binding by name."""
        if name in self.bindings:
            return self.bindings[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            return None
    
    def set(self, name: str, value: SAAMValue) -> bool:
        """Set an existing binding."""
        if name in self.bindings:
            self.bindings[name] = value
            return True
        elif self.parent:
            return self.parent.set(name, value)
        else:
            return False


class SAAMInterpreter:
    """
    The SAAAM EXECUTION ENGINE! 🚀
    
    Executes SAAAM code with:
    - Neuroplastic type morphing
    - Synapse operators
    - Component reactivity
    - Async/concurrency
    - Memory management
    """
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.async_runtime = AsyncRuntime()
        self.global_env = Environment()
        self.current_env = self.global_env
        
        # Runtime state
        self.components: Dict[str, ComponentInstance] = {}
        self.neural_blocks: Dict[str, NeuralBlock] = {}
        self.running_tasks: List[asyncio.Task] = []
        
        # Builtin functions
        self._setup_builtins()
    
    def _setup_builtins(self):
        """Setup builtin functions and values."""
        builtins = {
            "print": SAAMValue(
                data=self._builtin_print,
                type=SAAMType(TypeKind.FUNCTION, "print"),
                region=MemoryRegion.STACK
            ),
            "len": SAAMValue(
                data=self._builtin_len,
                type=SAAMType(TypeKind.FUNCTION, "len"),
                region=MemoryRegion.STACK
            ),
            "str": SAAMValue(
                data=self._builtin_str,
                type=SAAMType(TypeKind.FUNCTION, "str"),
                region=MemoryRegion.STACK
            ),
            "int": SAAMValue(
                data=self._builtin_int,
                type=SAAMType(TypeKind.FUNCTION, "int"),
                region=MemoryRegion.STACK
            ),
            "float": SAAMValue(
                data=self._builtin_float,
                type=SAAMType(TypeKind.FUNCTION, "float"),
                region=MemoryRegion.STACK
            ),
        }
        
        for name, value in builtins.items():
            self.global_env.define(name, value)
    
    def _builtin_print(self, *args: SAAMValue) -> SAAMValue:
        """Built-in print function."""
        output = " ".join(str(arg.data) for arg in args)
        print(output)
        return SAAMValue(None, BuiltinTypes.NONE)
    
    def _builtin_len(self, value: SAAMValue) -> SAAMValue:
        """Built-in len function."""
        if hasattr(value.data, '__len__'):
            length = len(value.data)
            return SAAMValue(length, BuiltinTypes.INT)
        else:
            raise RuntimeError(f"Object of type {value.type} has no len()")
    
    def _builtin_str(self, value: SAAMValue) -> SAAMValue:
        """Built-in str function."""
        return SAAMValue(str(value.data), BuiltinTypes.STRING)
    
    def _builtin_int(self, value: SAAMValue) -> SAAMValue:
        """Built-in int function."""
        try:
            return SAAMValue(int(value.data), BuiltinTypes.INT)
        except (ValueError, TypeError):
            raise RuntimeError(f"Cannot convert {value.type} to int")
    
    def _builtin_float(self, value: SAAMValue) -> SAAMValue:
        """Built-in float function."""
        try:
            return SAAMValue(float(value.data), BuiltinTypes.FLOAT)
        except (ValueError, TypeError):
            raise RuntimeError(f"Cannot convert {value.type} to float")
    
    def execute(self, program: Program) -> SAAMValue:
        """Execute a SAAAM program."""
        try:
            result = SAAMValue(None, BuiltinTypes.NONE)
            
            # Execute all declarations
            for decl in program.declarations:
                result = self.execute_declaration(decl)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")
    
    def execute_declaration(self, node: Declaration) -> SAAMValue:
        """Execute a declaration."""
        if isinstance(node, VarDecl):
            return self.execute_var_decl(node)
        elif isinstance(node, FunctionDecl):
            return self.execute_function_decl(node)
        elif isinstance(node, ComponentDecl):
            return self.execute_component_decl(node)
        elif isinstance(node, NeuralBlockDecl):
            return self.execute_neural_block_decl(node)
        else:
            raise RuntimeError(f"Unknown declaration type: {type(node)}")
    
    def execute_var_decl(self, node: VarDecl) -> SAAMValue:
        """Execute variable declaration."""
        # Determine memory region
        region = MemoryRegion.STACK
        if node.is_gc:
            region = MemoryRegion.GC
        elif node.is_neural:
            region = MemoryRegion.NEURAL
        elif getattr(node, 'is_heap', False):
            region = MemoryRegion.HEAP
        
        # Get initial value
        if node.initializer:
            value = self.execute_expression(node.initializer)
            
            # Handle neuroplastic morphing during initialization
            if node.is_neural and not value.is_neural:
                value = self.memory_manager.allocate(
                    data=value.data,
                    saaam_type=value.type,
                    region=MemoryRegion.NEURAL
                )
        else:
            # Default value
            value = SAAMValue(None, BuiltinTypes.NONE, region)
        
        if node.is_neural:
            value.is_neural = True
        
        # Define in current environment
        self.current_env.define(node.name, value)
        
        return value
    
    def execute_function_decl(self, node: FunctionDecl) -> SAAMValue:
        """Execute function declaration."""
        # Create callable function
        def saaam_function(*args: SAAMValue) -> SAAMValue:
            # Create new environment for function
            func_env = Environment(self.current_env)
            func_env.is_function_env = True
            
            # Bind parameters
            for i, param in enumerate(node.params):
                if i < len(args):
                    func_env.define(param.name, args[i])
                elif param.default_value:
                    default = self.execute_expression(param.default_value)
                    func_env.define(param.name, default)
                else:
                    raise RuntimeError(f"Missing argument for parameter {param.name}")
            
            # Execute function body
            old_env = self.current_env
            self.current_env = func_env
            
            try:
                if node.body:
                    return self.execute_block(node.body)
                else:
                    return SAAMValue(None, BuiltinTypes.NONE)
            finally:
                self.current_env = old_env
        
        # Create function value
        func_value = SAAMValue(
            data=saaam_function,
            type=SAAMType(TypeKind.FUNCTION, node.name)
        )
        
        self.current_env.define(node.name, func_value)
        return func_value
    
    def execute_component_decl(self, node: ComponentDecl) -> SAAMValue:
        """Execute component declaration."""
        # Component is a factory function
        def create_component(**props) -> ComponentInstance:
            prop_values = {}
            for prop_name, prop_value in props.items():
                if not isinstance(prop_value, SAAMValue):
                    prop_values[prop_name] = SAAMValue(prop_value, BuiltinTypes.ANY)
                else:
                    prop_values[prop_name] = prop_value
            
            component = ComponentInstance(node, prop_values)
            self.components[f"{node.name}_{id(component)}"] = component
            return component
        
        comp_value = SAAMValue(
            data=create_component,
            type=SAAMType(TypeKind.COMPONENT, node.name)
        )
        
        self.current_env.define(node.name, comp_value)
        return comp_value
    
    def execute_neural_block_decl(self, node: NeuralBlockDecl) -> SAAMValue:
        """Execute neural block declaration."""
        neural_block = NeuralBlock(node)
        self.neural_blocks[node.name] = neural_block
        
        block_value = SAAMValue(
            data=neural_block,
            type=SAAMType(TypeKind.NEURAL, node.name)
        )
        
        self.current_env.define(node.name, block_value)
        return block_value
    
    def execute_expression(self, node: Expression) -> SAAMValue:
        """Execute any expression."""
        if isinstance(node, IntegerLiteral):
            return SAAMValue(node.value, BuiltinTypes.INT)
        elif isinstance(node, FloatLiteral):
            return SAAMValue(node.value, BuiltinTypes.FLOAT)
        elif isinstance(node, StringLiteral):
            return SAAMValue(node.value, BuiltinTypes.STRING)
        elif isinstance(node, BoolLiteral):
            return SAAMValue(node.value, BuiltinTypes.BOOL)
        elif isinstance(node, CharLiteral):
            return SAAMValue(node.value, BuiltinTypes.CHAR)
        elif isinstance(node, NoneLiteral):
            return SAAMValue(None, BuiltinTypes.NONE)
        elif isinstance(node, Identifier):
            return self.execute_identifier(node)
        elif isinstance(node, BinaryOp):
            return self.execute_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self.execute_unary_op(node)
        elif isinstance(node, Call):
            return self.execute_call(node)
        elif isinstance(node, ArrayLiteral):
            return self.execute_array_literal(node)
        elif isinstance(node, Block):
            return self.execute_block(node)
        elif isinstance(node, IfExpr):
            return self.execute_if_expr(node)
        else:
            raise RuntimeError(f"Unknown expression type: {type(node)}")
    
    def execute_identifier(self, node: Identifier) -> SAAMValue:
        """Execute identifier lookup."""
        value = self.current_env.get(node.name)
        if value is None:
            raise RuntimeError(f"Undefined variable: {node.name}")
        return value
    
    def execute_binary_op(self, node: BinaryOp) -> SAAMValue:
        """Execute binary operation - THE SYNAPSE OPERATORS! ⚡🧠"""
        # Handle neuroplastic morph operator ~>
        if node.operator == "~>":
            return self.execute_morph_op(node)
        
        # Handle bidirectional bind <=>
        elif node.operator == "<=>":
            return self.execute_bind_op(node)
        
        # Handle flow operator ->
        elif node.operator == "->":
            return self.execute_flow_op(node)
        
        # Handle inject operator @>
        elif node.operator == "@>":
            return self.execute_inject_op(node)
        
        # Regular operators
        left = self.execute_expression(node.left)
        right = self.execute_expression(node.right)
        
        if node.operator == "+":
            return self._add_values(left, right)
        elif node.operator == "-":
            return self._subtract_values(left, right)
        elif node.operator == "*":
            return self._multiply_values(left, right)
        elif node.operator == "/":
            return self._divide_values(left, right)
        elif node.operator == "=":
            return self.execute_assignment(node.left, right)
        elif node.operator == "==":
            return SAAMValue(left.data == right.data, BuiltinTypes.BOOL)
        elif node.operator == "!=":
            return SAAMValue(left.data != right.data, BuiltinTypes.BOOL)
        elif node.operator == "<":
            return SAAMValue(left.data < right.data, BuiltinTypes.BOOL)
        elif node.operator == ">":
            return SAAMValue(left.data > right.data, BuiltinTypes.BOOL)
        elif node.operator == "<=":
            return SAAMValue(left.data <= right.data, BuiltinTypes.BOOL)
        elif node.operator == ">=":
            return SAAMValue(left.data >= right.data, BuiltinTypes.BOOL)
        else:
            raise RuntimeError(f"Unknown operator: {node.operator}")
    
    def execute_morph_op(self, node: BinaryOp) -> SAAMValue:
        """Execute neuroplastic morph operator ~> 🧠⚡"""
        left = self.execute_expression(node.left)
        right = self.execute_expression(node.right)
        
        if not left.is_neural:
            raise RuntimeError("Cannot morph non-neural value")
        
        # Perform the morph!
        morphed = left.morph_to(right.type, right.data)
        
        # Update variable binding if left is identifier
        if isinstance(node.left, Identifier):
            self.current_env.set(node.left.name, morphed)
        
        return morphed
    
    def execute_bind_op(self, node: BinaryOp) -> SAAMValue:
        """Execute bidirectional bind operator <=> for reactivity."""
        left = self.execute_expression(node.left)
        right = self.execute_expression(node.right)
        
        # Create bidirectional reactive binding
        if isinstance(node.left, Identifier) and isinstance(node.right, Identifier):
            left_name = node.left.name
            right_name = node.right.name
            
            # Create mutual observers
            class MutualObserver(ReactiveObserver):
                def __init__(self, env: Environment, target_name: str):
                    self.env = env
                    self.target_name = target_name
                
                def on_value_changed(self, value: SAAMValue):
                    # Update the bound variable
                    self.env.set(self.target_name, value.clone())
            
            left.add_observer(MutualObserver(self.current_env, right_name))
            right.add_observer(MutualObserver(self.current_env, left_name))
        
        return left
    
    def execute_flow_op(self, node: BinaryOp) -> SAAMValue:
        """Execute flow operator -> for pipelines."""
        left = self.execute_expression(node.left)
        
        # Right side should be a function call
        if isinstance(node.right, Call):
            # Insert left as first argument
            call_node = Call(
                callee=node.right.callee,
                arguments=[node.left] + node.right.arguments,
                location=node.location
            )
            return self.execute_call(call_node)
        else:
            raise RuntimeError("Flow operator requires function call on right side")
    
    def execute_inject_op(self, node: BinaryOp) -> SAAMValue:
        """Execute inject operator @> for dependency injection."""
        left = self.execute_expression(node.left)
        right = self.execute_expression(node.right)
        
        # Simple DI implementation - inject left into right
        # In a real implementation, this would use a DI container
        return left
    
    def execute_assignment(self, target: Expression, value: SAAMValue) -> SAAMValue:
        """Execute assignment."""
        if isinstance(target, Identifier):
            # Check if variable exists and is mutable
            existing = self.current_env.get(target.name)
            if existing and existing.is_owned and not existing.is_borrowed:
                # Direct assignment
                self.current_env.set(target.name, value)
            else:
                raise RuntimeError(f"Cannot assign to {target.name}")
        else:
            raise RuntimeError("Invalid assignment target")
        
        return value
    
    def execute_call(self, node: Call) -> SAAMValue:
        """Execute function call."""
        func_value = self.execute_expression(node.callee)
        
        if not callable(func_value.data):
            raise RuntimeError(f"Cannot call non-function value")
        
        # Evaluate arguments
        args = [self.execute_expression(arg) for arg in node.arguments]
        
        # Call function
        try:
            result = func_value.data(*args)
            if isinstance(result, SAAMValue):
                return result
            else:
                # Wrap non-SAAMValue returns
                return SAAMValue(result, BuiltinTypes.ANY)
        except Exception as e:
            raise RuntimeError(f"Function call failed: {e}")
    
    def execute_array_literal(self, node: ArrayLiteral) -> SAAMValue:
        """Execute array literal."""
        elements = [self.execute_expression(elem) for elem in node.elements]
        array_data = [elem.data for elem in elements]
        
        # Infer element type
        if elements:
            elem_type = elements[0].type
        else:
            elem_type = BuiltinTypes.ANY
        
        array_type = BuiltinTypes.array(elem_type)
        return SAAMValue(array_data, array_type)
    
    def execute_block(self, node: Block) -> SAAMValue:
        """Execute block of statements."""
        result = SAAMValue(None, BuiltinTypes.NONE)
        
        # Execute statements
        for stmt in node.statements:
            if isinstance(stmt, ExprStmt):
                result = self.execute_expression(stmt.expr)
            elif isinstance(stmt, VarDecl):
                self.execute_var_decl(stmt)
            else:
                raise RuntimeError(f"Unknown statement type: {type(stmt)}")
        
        # Execute final expression
        if node.final_expr:
            result = self.execute_expression(node.final_expr)
        
        return result
    
    def execute_if_expr(self, node: IfExpr) -> SAAMValue:
        """Execute if expression."""
        condition = self.execute_expression(node.condition)
        
        if condition.data:
            return self.execute_expression(node.then_branch)
        elif node.else_branch:
            return self.execute_expression(node.else_branch)
        else:
            return SAAMValue(None, BuiltinTypes.NONE)
    
    # === ARITHMETIC HELPERS ===
    
    def _add_values(self, left: SAAMValue, right: SAAMValue) -> SAAMValue:
        """Add two values."""
        if left.type.name in {"Int", "Float"} and right.type.name in {"Int", "Float"}:
            result = left.data + right.data
            result_type = BuiltinTypes.FLOAT if "Float" in {left.type.name, right.type.name} else BuiltinTypes.INT
            return SAAMValue(result, result_type)
        elif left.type.name == "String" or right.type.name == "String":
            result = str(left.data) + str(right.data)
            return SAAMValue(result, BuiltinTypes.STRING)
        else:
            raise RuntimeError(f"Cannot add {left.type} and {right.type}")
    
    def _subtract_values(self, left: SAAMValue, right: SAAMValue) -> SAAMValue:
        """Subtract two values."""
        if left.type.name in {"Int", "Float"} and right.type.name in {"Int", "Float"}:
            result = left.data - right.data
            result_type = BuiltinTypes.FLOAT if "Float" in {left.type.name, right.type.name} else BuiltinTypes.INT
            return SAAMValue(result, result_type)
        else:
            raise RuntimeError(f"Cannot subtract {right.type} from {left.type}")
    
    def _multiply_values(self, left: SAAMValue, right: SAAMValue) -> SAAMValue:
        """Multiply two values."""
        if left.type.name in {"Int", "Float"} and right.type.name in {"Int", "Float"}:
            result = left.data * right.data
            result_type = BuiltinTypes.FLOAT if "Float" in {left.type.name, right.type.name} else BuiltinTypes.INT
            return SAAMValue(result, result_type)
        else:
            raise RuntimeError(f"Cannot multiply {left.type} and {right.type}")
    
    def _divide_values(self, left: SAAMValue, right: SAAMValue) -> SAAMValue:
        """Divide two values."""
        if left.type.name in {"Int", "Float"} and right.type.name in {"Int", "Float"}:
            if right.data == 0:
                raise RuntimeError("Division by zero")
            result = left.data / right.data
            return SAAMValue(result, BuiltinTypes.FLOAT)
        else:
            raise RuntimeError(f"Cannot divide {left.type} by {right.type}")
    
    # === ASYNC EXECUTION ===
    
    async def execute_async(self, program: Program) -> SAAMValue:
        """Execute program with async support."""
        return self.execute(program)
    
    async def spawn_task(self, block: Block) -> asyncio.Task:
        """Spawn an async task."""
        async def task_runner():
            return self.execute_block(block)
        
        task = asyncio.create_task(task_runner())
        self.running_tasks.append(task)
        return task
    
    def run_parallel(self, tasks: List[Block]) -> List[SAAMValue]:
        """Run tasks in parallel."""
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(self.execute_block, task) for task in tasks]
            results = [future.result() for future in futures]
        return results


def run_saaam_code(source_code: str) -> SAAMValue:
    """
    Run SAAAM source code and return the result.
    """
    from .parser import parse
    from .semantic_analyzer import analyze_program
    
    try:
        # Parse the code
        program = parse(source_code)
        
        # Semantic analysis
        success, diagnostics = analyze_program(program)
        if not success:
            raise RuntimeError(f"Semantic errors: {diagnostics['errors']}")
        
        # Execute!
        interpreter = SAAMInterpreter()
        result = interpreter.execute(program)
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Execution failed: {e}")


def run_saaam_file(filepath: str) -> SAAMValue:
    """Run a .saaam file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    return run_saaam_code(source_code)


if __name__ == "__main__":
    # Test the runtime with SAAAM's revolutionary features! 🚀
    test_code = """
    # Neuroplastic typing in action! 🧠
    neural x = 42
    print("x is:", x)
    
    x ~> "Hello SAAAM!"
    print("x morphed to:", x)
    
    x ~> 3.14159
    print("x morphed again to:", x)
    
    # Function definition and call
    fn greet(name: String) -> String {
        "Hello, " + name + "!"
    }
    
    let message = greet("SAAAM")
    print(message)
    
    # Flow operator pipeline 🚀
    # message -> print()
    """
    
    try:
        print("🚀 EXECUTING SAAAM CODE! 🚀\n")
        result = run_saaam_code(test_code)
        print(f"\n✅ Execution completed! Final result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
