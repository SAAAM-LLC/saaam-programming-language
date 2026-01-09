"""
SAAAM Language - Semantic Analyzer
The Neural Intelligence: Where syntax becomes MEANING and types become ALIVE.

This is where SAAAM's revolutionary features come to life:
- Neuroplastic typing that evolves
- Synapse operator semantics
- Component reactivity analysis
- Ownership/borrowing validation
- Pattern exhaustiveness checking
"""

from typing import Dict, List, Set, Optional, Union, Any, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import copy

from .ast_nodes import *
from .tokens import SourceLocation


class SemanticError(Exception):
    """When the neural pathways can't compute."""
    def __init__(self, message: str, location: Optional[SourceLocation] = None, 
                 suggestions: List[str] = None):
        self.message = message
        self.location = location
        self.suggestions = suggestions or []
        super().__init__(f"{location}: {message}" if location else message)


class TypeKind(Enum):
    """The fundamental nature of types in SAAAM."""
    PRIMITIVE = auto()      # Int, Float, String, etc.
    STRUCT = auto()         # User-defined structs
    ENUM = auto()           # Enums with variants
    TRAIT = auto()          # Interfaces/traits
    FUNCTION = auto()       # Function types
    ARRAY = auto()          # [T] or Array<T>
    MAP = auto()            # Map<K, V>
    TUPLE = auto()          # (A, B, C)
    OPTION = auto()         # Option<T>
    RESULT = auto()         # Result<T, E>
    REFERENCE = auto()      # &T or &mut T
    NEURAL = auto()         # Neuroplastic types 🧠
    COMPONENT = auto()      # React-style components
    CHANNEL = auto()        # Async channels
    ACTOR = auto()          # Actor types
    UNKNOWN = auto()        # During inference
    ERROR = auto()          # Type error sentinel


@dataclass
class SAAMType:
    """
    The living, breathing type system of SAAAM.
    Types can MORPH, EVOLVE, and ADAPT!
    """
    kind: TypeKind
    name: str = ""
    args: List['SAAMType'] = field(default_factory=list)
    
    # Neuroplastic properties 🧠
    is_neural: bool = False
    possible_morphs: Set['SAAMType'] = field(default_factory=set)
    morph_history: List['SAAMType'] = field(default_factory=list)
    
    # Ownership properties (Rust-inspired)
    is_owned: bool = True
    is_mutable: bool = False
    is_reference: bool = False
    is_gc: bool = False
    
    # Component properties
    is_reactive: bool = False
    component_state: Optional[Dict[str, 'SAAMType']] = None
    
    def __str__(self) -> str:
        result = self.name
        if self.args:
            args_str = ', '.join(str(arg) for arg in self.args)
            result += f"<{args_str}>"
        
        if self.is_neural:
            result += " neural"
        if self.is_reference:
            prefix = "&mut " if self.is_mutable else "&"
            result = prefix + result
        if self.is_gc:
            result = "gc " + result
            
        return result
    
    def __hash__(self):
        return hash((self.kind, self.name, tuple(self.args)))
    
    def __eq__(self, other):
        if not isinstance(other, SAAMType):
            return False
        return (self.kind == other.kind and 
                self.name == other.name and 
                self.args == other.args)
    
    def can_morph_to(self, target: 'SAAMType') -> bool:
        """Check if this type can neuroplastically morph to target."""
        if not self.is_neural:
            return False
        
        if target in self.possible_morphs:
            return True
        
        # Built-in neuroplastic morphing rules
        if self.kind == TypeKind.NEURAL and target.kind in {
            TypeKind.PRIMITIVE, TypeKind.STRUCT, TypeKind.ENUM
        }:
            return True
        
        # Numerical morphing: Int <-> Float
        if (self.name == "Int" and target.name == "Float" or
            self.name == "Float" and target.name == "Int"):
            return True
        
        # String morphing to/from most types
        if self.name == "String" or target.name == "String":
            return True
        
        return False
    
    def morph_to(self, target: 'SAAMType') -> 'SAAMType':
        """Perform neuroplastic morphing."""
        if not self.can_morph_to(target):
            raise SemanticError(f"Cannot morph {self} to {target}")
        
        new_type = copy.deepcopy(target)
        new_type.is_neural = self.is_neural
        new_type.morph_history = self.morph_history + [self]
        new_type.possible_morphs = self.possible_morphs | {target}
        
        return new_type
    
    def is_compatible_with(self, other: 'SAAMType') -> bool:
        """Check type compatibility for assignments, etc."""
        if self == other:
            return True
        
        # Neuroplastic compatibility
        if self.is_neural and self.can_morph_to(other):
            return True
        if other.is_neural and other.can_morph_to(self):
            return True
        
        # Reference compatibility
        if (self.is_reference and other.is_reference and
            self.args and other.args and
            self.args[0].is_compatible_with(other.args[0])):
            # &T is compatible with &mut T for reads
            if not self.is_mutable and other.is_mutable:
                return True
        
        # Option unwrapping: Option<T> is compatible with T
        if (self.kind == TypeKind.OPTION and self.args and
            self.args[0].is_compatible_with(other)):
            return True
        
        # Subtyping for structs/traits (would need inheritance info)
        
        return False


# Built-in types
class BuiltinTypes:
    """The fundamental types that power SAAAM."""
    
    INT = SAAMType(TypeKind.PRIMITIVE, "Int")
    INT8 = SAAMType(TypeKind.PRIMITIVE, "Int8")
    INT16 = SAAMType(TypeKind.PRIMITIVE, "Int16")
    INT32 = SAAMType(TypeKind.PRIMITIVE, "Int32")
    INT64 = SAAMType(TypeKind.PRIMITIVE, "Int64")
    
    UINT = SAAMType(TypeKind.PRIMITIVE, "UInt")
    UINT8 = SAAMType(TypeKind.PRIMITIVE, "UInt8")
    UINT16 = SAAMType(TypeKind.PRIMITIVE, "UInt16")
    UINT32 = SAAMType(TypeKind.PRIMITIVE, "UInt32")
    UINT64 = SAAMType(TypeKind.PRIMITIVE, "UInt64")
    
    FLOAT = SAAMType(TypeKind.PRIMITIVE, "Float")
    FLOAT32 = SAAMType(TypeKind.PRIMITIVE, "Float32")
    FLOAT64 = SAAMType(TypeKind.PRIMITIVE, "Float64")
    
    BOOL = SAAMType(TypeKind.PRIMITIVE, "Bool")
    CHAR = SAAMType(TypeKind.PRIMITIVE, "Char")
    STRING = SAAMType(TypeKind.PRIMITIVE, "String")
    
    NONE = SAAMType(TypeKind.PRIMITIVE, "None")
    ANY = SAAMType(TypeKind.PRIMITIVE, "Any")
    
    # Neural type - can morph into anything!
    NEURAL = SAAMType(TypeKind.NEURAL, "Neural", is_neural=True)
    
    ERROR = SAAMType(TypeKind.ERROR, "ERROR")
    UNKNOWN = SAAMType(TypeKind.UNKNOWN, "UNKNOWN")
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional[SAAMType]:
        """Get builtin type by name."""
        mapping = {
            "Int": cls.INT, "Int8": cls.INT8, "Int16": cls.INT16,
            "Int32": cls.INT32, "Int64": cls.INT64,
            "UInt": cls.UINT, "UInt8": cls.UINT8, "UInt16": cls.UINT16,
            "UInt32": cls.UINT32, "UInt64": cls.UINT64,
            "Float": cls.FLOAT, "Float32": cls.FLOAT32, "Float64": cls.FLOAT64,
            "Bool": cls.BOOL, "Char": cls.CHAR, "String": cls.STRING,
            "None": cls.NONE, "Any": cls.ANY,
            "Neural": cls.NEURAL,
        }
        return mapping.get(name)
    
    @classmethod
    def option(cls, inner: SAAMType) -> SAAMType:
        """Create Option<T> type."""
        return SAAMType(TypeKind.OPTION, "Option", [inner])
    
    @classmethod
    def result(cls, ok: SAAMType, err: SAAMType) -> SAAMType:
        """Create Result<T, E> type."""
        return SAAMType(TypeKind.RESULT, "Result", [ok, err])
    
    @classmethod
    def array(cls, element: SAAMType) -> SAAMType:
        """Create Array<T> type."""
        return SAAMType(TypeKind.ARRAY, "Array", [element])
    
    @classmethod
    def map(cls, key: SAAMType, value: SAAMType) -> SAAMType:
        """Create Map<K, V> type."""
        return SAAMType(TypeKind.MAP, "Map", [key, value])
    
    @classmethod
    def reference(cls, inner: SAAMType, mutable: bool = False) -> SAAMType:
        """Create reference type &T or &mut T."""
        ref_type = SAAMType(TypeKind.REFERENCE, "ref", [inner])
        ref_type.is_reference = True
        ref_type.is_mutable = mutable
        return ref_type


@dataclass
class Symbol:
    """A symbol in the symbol table."""
    name: str
    type: SAAMType
    location: SourceLocation
    is_mutable: bool = False
    is_used: bool = False
    is_defined: bool = True
    scope_level: int = 0
    
    # Component-specific
    is_state: bool = False
    is_prop: bool = False
    is_reactive: bool = False


@dataclass
class Scope:
    """A scope in the symbol table."""
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    parent: Optional['Scope'] = None
    level: int = 0
    is_function: bool = False
    is_component: bool = False
    is_neural_block: bool = False
    
    def define(self, symbol: Symbol):
        """Define a symbol in this scope."""
        if symbol.name in self.symbols:
            existing = self.symbols[symbol.name]
            if not existing.is_neural_redefinable():
                raise SemanticError(
                    f"Symbol '{symbol.name}' already defined at {existing.location}",
                    symbol.location
                )
        
        symbol.scope_level = self.level
        self.symbols[symbol.name] = symbol
    
    def resolve(self, name: str) -> Optional[Symbol]:
        """Resolve a symbol by name, searching parent scopes."""
        if name in self.symbols:
            self.symbols[name].is_used = True
            return self.symbols[name]
        
        if self.parent:
            return self.parent.resolve(name)
        
        return None


class SymbolTable:
    """The neural symbol table - tracks all identifiers and their types."""
    
    def __init__(self):
        self.global_scope = Scope()
        self.current_scope = self.global_scope
        self.scope_stack: List[Scope] = [self.global_scope]
        
        # Populate builtins
        self._populate_builtins()
    
    def _populate_builtins(self):
        """Add builtin types and functions."""
        builtins = [
            ("Int", BuiltinTypes.INT),
            ("Float", BuiltinTypes.FLOAT),
            ("String", BuiltinTypes.STRING),
            ("Bool", BuiltinTypes.BOOL),
            ("Char", BuiltinTypes.CHAR),
            ("None", BuiltinTypes.NONE),
            ("Any", BuiltinTypes.ANY),
            ("Neural", BuiltinTypes.NEURAL),
        ]
        
        for name, typ in builtins:
            self.define(Symbol(
                name=name,
                type=typ,
                location=SourceLocation("<builtin>", 0, 0, 0),
                is_defined=True
            ))
    
    def push_scope(self, is_function: bool = False, is_component: bool = False,
                   is_neural_block: bool = False) -> Scope:
        """Push a new scope."""
        scope = Scope(
            parent=self.current_scope,
            level=len(self.scope_stack),
            is_function=is_function,
            is_component=is_component,
            is_neural_block=is_neural_block
        )
        self.scope_stack.append(scope)
        self.current_scope = scope
        return scope
    
    def pop_scope(self) -> Scope:
        """Pop the current scope."""
        if len(self.scope_stack) <= 1:
            raise SemanticError("Cannot pop global scope")
        
        scope = self.scope_stack.pop()
        self.current_scope = self.scope_stack[-1]
        return scope
    
    def define(self, symbol: Symbol):
        """Define a symbol in current scope."""
        self.current_scope.define(symbol)
    
    def resolve(self, name: str) -> Optional[Symbol]:
        """Resolve a symbol by name."""
        return self.current_scope.resolve(name)
    
    def is_in_function(self) -> bool:
        """Check if we're inside a function."""
        for scope in reversed(self.scope_stack):
            if scope.is_function:
                return True
        return False
    
    def is_in_component(self) -> bool:
        """Check if we're inside a component."""
        for scope in reversed(self.scope_stack):
            if scope.is_component:
                return True
        return False


class SemanticAnalyzer:
    """
    The NEURAL BRAIN of SAAAM! 🧠
    
    Understands:
    - Neuroplastic typing and morphing
    - Synapse operator semantics  
    - Component reactivity
    - Ownership/borrowing
    - Pattern exhaustiveness
    - Type inference
    """
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []
        self.warnings: List[str] = []
        
        # Type inference state
        self.type_constraints: List[tuple] = []
        self.infer_context: Dict[str, SAAMType] = {}
        
        # Component analysis state
        self.current_component: Optional[ComponentDecl] = None
        self.reactive_bindings: Dict[str, Set[str]] = {}
        
        # Neural block analysis
        self.current_neural_block: Optional[NeuralBlockDecl] = None
        
        # Flow analysis for ownership
        self.ownership_graph: Dict[str, Set[str]] = {}
        self.borrowed_vars: Dict[str, SourceLocation] = {}
    
    def error(self, message: str, location: Optional[SourceLocation] = None,
              suggestions: List[str] = None):
        """Record a semantic error."""
        self.errors.append(SemanticError(message, location, suggestions))
    
    def warning(self, message: str, location: Optional[SourceLocation] = None):
        """Record a warning."""
        loc_str = f"{location}: " if location else ""
        self.warnings.append(f"{loc_str}{message}")
    
    def analyze(self, program: Program) -> bool:
        """
        Analyze the entire program.
        Returns True if no errors, False if errors found.
        """
        try:
            self.visit_program(program)
            self.check_unused_variables()
            self.resolve_type_constraints()
            return len(self.errors) == 0
        except Exception as e:
            self.error(f"Internal error during analysis: {e}")
            return False
    
    def get_diagnostics(self) -> Dict[str, List[str]]:
        """Get all diagnostics."""
        return {
            "errors": [str(e) for e in self.errors],
            "warnings": self.warnings
        }
    
    # === PROGRAM ANALYSIS ===
    
    def visit_program(self, node: Program):
        """Analyze the entire program."""
        # First pass: collect type definitions
        for decl in node.declarations:
            if isinstance(decl, (StructDecl, EnumDecl, TraitDecl, TypeAlias)):
                self.collect_type_declaration(decl)
        
        # Second pass: analyze all declarations
        for decl in node.declarations:
            self.visit_declaration(decl)
    
    def collect_type_declaration(self, decl: Declaration):
        """Collect type declarations for forward reference."""
        if isinstance(decl, StructDecl):
            struct_type = SAAMType(TypeKind.STRUCT, decl.name)
            self.symbol_table.define(Symbol(
                name=decl.name,
                type=struct_type,
                location=decl.location
            ))
        
        elif isinstance(decl, EnumDecl):
            enum_type = SAAMType(TypeKind.ENUM, decl.name)
            self.symbol_table.define(Symbol(
                name=decl.name,
                type=enum_type,
                location=decl.location
            ))
        
        elif isinstance(decl, TraitDecl):
            trait_type = SAAMType(TypeKind.TRAIT, decl.name)
            self.symbol_table.define(Symbol(
                name=decl.name,
                type=trait_type,
                location=decl.location
            ))
    
    # === DECLARATION ANALYSIS ===
    
    def visit_declaration(self, node: Declaration):
        """Visit any declaration."""
        if isinstance(node, VarDecl):
            self.visit_var_decl(node)
        elif isinstance(node, FunctionDecl):
            self.visit_function_decl(node)
        elif isinstance(node, StructDecl):
            self.visit_struct_decl(node)
        elif isinstance(node, EnumDecl):
            self.visit_enum_decl(node)
        elif isinstance(node, TraitDecl):
            self.visit_trait_decl(node)
        elif isinstance(node, ImplDecl):
            self.visit_impl_decl(node)
        elif isinstance(node, ComponentDecl):
            self.visit_component_decl(node)
        elif isinstance(node, NeuralBlockDecl):
            self.visit_neural_block_decl(node)
        else:
            self.error(f"Unknown declaration type: {type(node)}", node.location)
    
    def visit_var_decl(self, node: VarDecl):
        """Analyze variable declaration."""
        # Infer or validate type
        var_type = BuiltinTypes.UNKNOWN
        
        if node.var_type:
            var_type = self.resolve_type_annotation(node.var_type)
        
        if node.initializer:
            init_type = self.visit_expression(node.initializer)
            
            if node.var_type:
                # Check compatibility
                if not self.check_assignment_compatibility(var_type, init_type, node.location):
                    return
            else:
                # Infer type from initializer
                var_type = init_type
        
        # Handle neuroplastic typing
        if node.is_neural or (node.initializer and var_type.is_neural):
            var_type = copy.deepcopy(var_type)
            var_type.is_neural = True
        
        # Handle ownership modifiers
        if node.is_gc:
            var_type = copy.deepcopy(var_type)
            var_type.is_gc = True
        
        # Create symbol
        symbol = Symbol(
            name=node.name,
            type=var_type,
            location=node.location,
            is_mutable=node.is_mutable,
            is_state=getattr(node, 'is_state', False),
            is_prop=getattr(node, 'is_prop', False)
        )
        
        # Component reactivity
        if self.symbol_table.is_in_component() and symbol.is_state:
            symbol.is_reactive = True
        
        self.symbol_table.define(symbol)
    
    def visit_function_decl(self, node: FunctionDecl):
        """Analyze function declaration."""
        # Create function type
        param_types = []
        return_type = BuiltinTypes.NONE
        
        if node.return_type:
            return_type = self.resolve_type_annotation(node.return_type)
        
        # Push function scope
        self.symbol_table.push_scope(is_function=True)
        
        try:
            # Add parameters to scope
            for param in node.params:
                param_type = BuiltinTypes.ANY
                if param.param_type:
                    param_type = self.resolve_type_annotation(param.param_type)
                
                param_symbol = Symbol(
                    name=param.name,
                    type=param_type,
                    location=param.location or node.location,
                    is_mutable=param.is_mutable
                )
                self.symbol_table.define(param_symbol)
                param_types.append(param_type)
            
            # Analyze function body
            if node.body:
                body_type = self.visit_block(node.body)
                
                # Check return type compatibility
                if return_type != BuiltinTypes.NONE:
                    if not self.check_assignment_compatibility(return_type, body_type, node.location):
                        self.error(
                            f"Function body type {body_type} incompatible with return type {return_type}",
                            node.location
                        )
            
            # Create function symbol
            func_type = SAAMType(TypeKind.FUNCTION, node.name, param_types + [return_type])
            func_symbol = Symbol(
                name=node.name,
                type=func_type,
                location=node.location
            )
            
            # Define in parent scope (before popping)
            parent_scope = self.symbol_table.current_scope.parent
            if parent_scope:
                parent_scope.define(func_symbol)
        
        finally:
            self.symbol_table.pop_scope()
    
    def visit_component_decl(self, node: ComponentDecl):
        """Analyze React-style component declaration."""
        self.current_component = node
        
        # Create component type
        comp_type = SAAMType(TypeKind.COMPONENT, node.name)
        comp_type.component_state = {}
        
        # Push component scope
        self.symbol_table.push_scope(is_component=True)
        
        try:
            # Analyze state variables
            for state_var in node.state:
                state_var.is_state = True
                self.visit_var_decl(state_var)
                comp_type.component_state[state_var.name] = self.symbol_table.resolve(state_var.name).type
            
            # Analyze props
            for prop_var in node.props:
                prop_var.is_prop = True
                self.visit_var_decl(prop_var)
            
            # Analyze methods
            for method in node.methods:
                self.visit_function_decl(method)
            
            # Analyze render function (JSX)
            if node.render:
                self.visit_jsx_element(node.render)
            
            # Analyze lifecycle hooks
            for hook_name, hook_body in node.lifecycle.items():
                self.visit_block(hook_body)
            
            # Check reactivity patterns
            self.analyze_component_reactivity(node)
            
            # Define component symbol
            comp_symbol = Symbol(
                name=node.name,
                type=comp_type,
                location=node.location
            )
            parent_scope = self.symbol_table.current_scope.parent
            if parent_scope:
                parent_scope.define(comp_symbol)
        
        finally:
            self.symbol_table.pop_scope()
            self.current_component = None
    
    def visit_neural_block_decl(self, node: NeuralBlockDecl):
        """Analyze neural block declaration - the BRAIN POWER! 🧠"""
        self.current_neural_block = node
        
        # Create neural block type
        neural_type = SAAMType(TypeKind.NEURAL, node.name)
        neural_type.is_neural = True
        
        self.symbol_table.push_scope(is_neural_block=True)
        
        try:
            # Analyze weights (neural variables)
            for weight_var in node.weights:
                weight_var.is_neural = True
                self.visit_var_decl(weight_var)
            
            # Analyze train mode
            if node.train_mode:
                self.visit_block(node.train_mode)
            
            # Analyze inference mode
            if node.infer_mode:
                self.visit_block(node.infer_mode)
            
            # Define neural block symbol
            neural_symbol = Symbol(
                name=node.name,
                type=neural_type,
                location=node.location
            )
            parent_scope = self.symbol_table.current_scope.parent
            if parent_scope:
                parent_scope.define(neural_symbol)
        
        finally:
            self.symbol_table.pop_scope()
            self.current_neural_block = None
    
    def visit_struct_decl(self, node: StructDecl):
        """Analyze struct declaration."""
        # Struct type already created in first pass
        struct_symbol = self.symbol_table.resolve(node.name)
        if not struct_symbol:
            self.error(f"Struct {node.name} not found", node.location)
            return
        
        struct_type = struct_symbol.type
        
        # Analyze fields
        field_types = {}
        for field in node.fields:
            field_type = self.resolve_type_annotation(field.field_type)
            field_types[field.name] = field_type
            
            if field.default_value:
                default_type = self.visit_expression(field.default_value)
                if not self.check_assignment_compatibility(field_type, default_type, field.location):
                    self.error(
                        f"Default value type {default_type} incompatible with field type {field_type}",
                        field.location
                    )
        
        # Update struct type with field information
        struct_type.component_state = field_types
    
    def visit_enum_decl(self, node: EnumDecl):
        """Analyze enum declaration."""
        enum_symbol = self.symbol_table.resolve(node.name)
        if not enum_symbol:
            self.error(f"Enum {node.name} not found", node.location)
            return
        
        # Analyze variants
        for variant in node.variants:
            # Create variant constructor type
            field_types = []
            for field_name, field_type in variant.fields:
                resolved_type = self.resolve_type_annotation(field_type)
                field_types.append(resolved_type)
            
            # Variant constructor is a function that returns the enum type
            variant_type = SAAMType(TypeKind.FUNCTION, variant.name, field_types + [enum_symbol.type])
            
            variant_symbol = Symbol(
                name=variant.name,
                type=variant_type,
                location=node.location
            )
            self.symbol_table.define(variant_symbol)
    
    def visit_trait_decl(self, node: TraitDecl):
        """Analyze trait declaration."""
        self.symbol_table.push_scope()
        
        try:
            # Analyze trait methods
            for method in node.methods:
                self.visit_function_decl(method)
        finally:
            self.symbol_table.pop_scope()
    
    def visit_impl_decl(self, node: ImplDecl):
        """Analyze implementation block."""
        target_type = self.resolve_type_annotation(node.target_type)
        
        self.symbol_table.push_scope()
        
        try:
            # Add 'self' parameter
            self_symbol = Symbol(
                name="self",
                type=target_type,
                location=node.location
            )
            self.symbol_table.define(self_symbol)
            
            # Analyze methods
            for method in node.methods:
                self.visit_function_decl(method)
        finally:
            self.symbol_table.pop_scope()
    
    # === EXPRESSION ANALYSIS ===
    
    def visit_expression(self, node: Expression) -> SAAMType:
        """Visit any expression and return its type."""
        if isinstance(node, IntegerLiteral):
            return BuiltinTypes.INT
        elif isinstance(node, FloatLiteral):
            return BuiltinTypes.FLOAT
        elif isinstance(node, StringLiteral):
            return BuiltinTypes.STRING
        elif isinstance(node, BoolLiteral):
            return BuiltinTypes.BOOL
        elif isinstance(node, CharLiteral):
            return BuiltinTypes.CHAR
        elif isinstance(node, NoneLiteral):
            return BuiltinTypes.NONE
        elif isinstance(node, Identifier):
            return self.visit_identifier(node)
        elif isinstance(node, BinaryOp):
            return self.visit_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self.visit_unary_op(node)
        elif isinstance(node, Call):
            return self.visit_call(node)
        elif isinstance(node, Member):
            return self.visit_member(node)
        elif isinstance(node, Index):
            return self.visit_index(node)
        elif isinstance(node, ArrayLiteral):
            return self.visit_array_literal(node)
        elif isinstance(node, MapLiteral):
            return self.visit_map_literal(node)
        elif isinstance(node, TupleLiteral):
            return self.visit_tuple_literal(node)
        elif isinstance(node, Block):
            return self.visit_block(node)
        elif isinstance(node, IfExpr):
            return self.visit_if_expr(node)
        elif isinstance(node, MatchExpr):
            return self.visit_match_expr(node)
        elif isinstance(node, Lambda):
            return self.visit_lambda(node)
        elif isinstance(node, JSXElement):
            return self.visit_jsx_element(node)
        else:
            self.error(f"Unknown expression type: {type(node)}", node.location)
            return BuiltinTypes.ERROR
    
    def visit_identifier(self, node: Identifier) -> SAAMType:
        """Visit identifier and return its type."""
        symbol = self.symbol_table.resolve(node.name)
        if not symbol:
            # Try to suggest similar names
            suggestions = self.suggest_similar_names(node.name)
            self.error(
                f"Undefined identifier: {node.name}",
                node.location,
                suggestions
            )
            return BuiltinTypes.ERROR
        
        return symbol.type
    
    def visit_binary_op(self, node: BinaryOp) -> SAAMType:
        """Visit binary operation - including SYNAPSE OPERATORS! 🧠⚡"""
        left_type = self.visit_expression(node.left)
        right_type = self.visit_expression(node.right)
        
        # Handle neuroplastic morph operator ~>
        if node.operator == "~>":
            return self.handle_morph_operator(left_type, right_type, node)
        
        # Handle bidirectional bind operator <=>
        elif node.operator == "<=>":
            return self.handle_bind_operator(left_type, right_type, node)
        
        # Handle flow operator ->
        elif node.operator == "->":
            return self.handle_flow_operator(left_type, right_type, node)
        
        # Handle inject operator @>
        elif node.operator == "@>":
            return self.handle_inject_operator(left_type, right_type, node)
        
        # Regular arithmetic/logical operators
        elif node.operator in {"+", "-", "*", "/", "%", "**", "//"}:
            return self.handle_arithmetic_operator(left_type, right_type, node.operator, node.location)
        
        elif node.operator in {"==", "!=", "<", "<=", ">", ">="}:
            return self.handle_comparison_operator(left_type, right_type, node.operator, node.location)
        
        elif node.operator in {"&&", "||", "and", "or"}:
            return self.handle_logical_operator(left_type, right_type, node.operator, node.location)
        
        elif node.operator == "=":
            return self.handle_assignment_operator(left_type, right_type, node)
        
        else:
            self.error(f"Unknown binary operator: {node.operator}", node.location)
            return BuiltinTypes.ERROR
    
    def handle_morph_operator(self, left_type: SAAMType, right_type: SAAMType, node: BinaryOp) -> SAAMType:
        """Handle neuroplastic morph operator ~> 🧠"""
        if not left_type.is_neural:
            self.error(
                f"Cannot use morph operator on non-neural type {left_type}",
                node.location,
                ["Add 'neural' keyword to make the variable neuroplastic"]
            )
            return BuiltinTypes.ERROR
        
        if not left_type.can_morph_to(right_type):
            self.error(
                f"Type {left_type} cannot morph to {right_type}",
                node.location
            )
            return BuiltinTypes.ERROR
        
        # Perform the morph!
        morphed_type = left_type.morph_to(right_type)
        
        # Update the left side's type if it's an identifier
        if isinstance(node.left, Identifier):
            symbol = self.symbol_table.resolve(node.left.name)
            if symbol:
                symbol.type = morphed_type
        
        return morphed_type
    
    def handle_bind_operator(self, left_type: SAAMType, right_type: SAAMType, node: BinaryOp) -> SAAMType:
        """Handle bidirectional bind operator <=> for reactivity."""
        if not self.symbol_table.is_in_component():
            self.error(
                "Bidirectional binding <=> can only be used in components",
                node.location
            )
            return BuiltinTypes.ERROR
        
        # Both sides must be compatible for bidirectional flow
        if not left_type.is_compatible_with(right_type):
            self.error(
                f"Cannot bind incompatible types {left_type} and {right_type}",
                node.location
            )
            return BuiltinTypes.ERROR
        
        # Record reactive binding for component analysis
        if isinstance(node.left, Identifier) and isinstance(node.right, Identifier):
            left_name = node.left.name
            right_name = node.right.name
            
            if left_name not in self.reactive_bindings:
                self.reactive_bindings[left_name] = set()
            if right_name not in self.reactive_bindings:
                self.reactive_bindings[right_name] = set()
            
            self.reactive_bindings[left_name].add(right_name)
            self.reactive_bindings[right_name].add(left_name)
        
        return left_type
    
    def handle_flow_operator(self, left_type: SAAMType, right_type: SAAMType, node: BinaryOp) -> SAAMType:
        """Handle flow operator -> for pipelines."""
        # Right side must be callable (function type)
        if right_type.kind != TypeKind.FUNCTION:
            self.error(
                f"Flow operator -> requires function on right side, got {right_type}",
                node.location
            )
            return BuiltinTypes.ERROR
        
        # Check if left type is compatible with first parameter of function
        if right_type.args:
            param_type = right_type.args[0]
            if not left_type.is_compatible_with(param_type):
                self.error(
                    f"Flow input type {left_type} incompatible with function parameter {param_type}",
                    node.location
                )
                return BuiltinTypes.ERROR
        
        # Return type is the function's return type
        return right_type.args[-1] if right_type.args else BuiltinTypes.ANY
    
    def handle_inject_operator(self, left_type: SAAMType, right_type: SAAMType, node: BinaryOp) -> SAAMType:
        """Handle inject operator @> for dependency injection."""
        # This would integrate with a DI container system
        # For now, just check basic compatibility
        return left_type
    
    def handle_arithmetic_operator(self, left: SAAMType, right: SAAMType, op: str, loc: SourceLocation) -> SAAMType:
        """Handle arithmetic operations."""
        # Numeric types
        numeric_types = {"Int", "Float", "Int8", "Int16", "Int32", "Int64", 
                        "UInt", "UInt8", "UInt16", "UInt32", "UInt64",
                        "Float32", "Float64"}
        
        if left.name in numeric_types and right.name in numeric_types:
            # Float promotion
            if left.name.startswith("Float") or right.name.startswith("Float"):
                return BuiltinTypes.FLOAT
            return BuiltinTypes.INT
        
        # String concatenation
        if op == "+" and (left.name == "String" or right.name == "String"):
            return BuiltinTypes.STRING
        
        self.error(f"Cannot apply {op} to {left} and {right}", loc)
        return BuiltinTypes.ERROR
    
    def handle_comparison_operator(self, left: SAAMType, right: SAAMType, op: str, loc: SourceLocation) -> SAAMType:
        """Handle comparison operations."""
        if left.is_compatible_with(right) or right.is_compatible_with(left):
            return BuiltinTypes.BOOL
        
        self.error(f"Cannot compare {left} and {right}", loc)
        return BuiltinTypes.ERROR
    
    def handle_logical_operator(self, left: SAAMType, right: SAAMType, op: str, loc: SourceLocation) -> SAAMType:
        """Handle logical operations."""
        if left.name == "Bool" and right.name == "Bool":
            return BuiltinTypes.BOOL
        
        self.error(f"Logical operator {op} requires Bool types, got {left} and {right}", loc)
        return BuiltinTypes.ERROR
    
    def handle_assignment_operator(self, left_type: SAAMType, right_type: SAAMType, node: BinaryOp) -> SAAMType:
        """Handle assignment operator ="""
        if not isinstance(node.left, (Identifier, Member, Index)):
            self.error("Invalid assignment target", node.location)
            return BuiltinTypes.ERROR
        
        # Check mutability
        if isinstance(node.left, Identifier):
            symbol = self.symbol_table.resolve(node.left.name)
            if symbol and not symbol.is_mutable:
                self.error(f"Cannot assign to immutable variable '{node.left.name}'", node.location)
                return BuiltinTypes.ERROR
        
        # Check type compatibility
        if not self.check_assignment_compatibility(left_type, right_type, node.location):
            return BuiltinTypes.ERROR
        
        return left_type
    
    def visit_call(self, node: Call) -> SAAMType:
        """Visit function call."""
        callee_type = self.visit_expression(node.callee)
        
        if callee_type.kind != TypeKind.FUNCTION:
            self.error(f"Cannot call non-function type {callee_type}", node.location)
            return BuiltinTypes.ERROR
        
        # Check argument compatibility
        param_types = callee_type.args[:-1]  # Last is return type
        arg_types = [self.visit_expression(arg) for arg in node.arguments]
        
        if len(arg_types) != len(param_types):
            self.error(
                f"Function expects {len(param_types)} arguments, got {len(arg_types)}",
                node.location
            )
            return BuiltinTypes.ERROR
        
        for i, (param_type, arg_type) in enumerate(zip(param_types, arg_types)):
            if not self.check_assignment_compatibility(param_type, arg_type, node.location):
                self.error(
                    f"Argument {i+1} type {arg_type} incompatible with parameter type {param_type}",
                    node.location
                )
                return BuiltinTypes.ERROR
        
        return callee_type.args[-1]  # Return type
    
    def visit_jsx_element(self, node: JSXElement) -> SAAMType:
        """Visit JSX element for component render functions."""
        if not self.symbol_table.is_in_component():
            self.error("JSX can only be used in component render functions", node.location)
            return BuiltinTypes.ERROR
        
        # Analyze attributes
        for attr_name, attr_expr in node.attributes:
            self.visit_expression(attr_expr)
        
        # Analyze children
        for child in node.children:
            if isinstance(child, JSXElement):
                self.visit_jsx_element(child)
            else:
                self.visit_expression(child)
        
        # JSX elements return a virtual DOM node type
        return SAAMType(TypeKind.PRIMITIVE, "JSXElement")
    
    def visit_block(self, node: Block) -> SAAMType:
        """Visit block expression."""
        last_type = BuiltinTypes.NONE
        
        for stmt in node.statements:
            if isinstance(stmt, ExprStmt):
                last_type = self.visit_expression(stmt.expr)
            else:
                self.visit_statement(stmt)
        
        if node.final_expr:
            last_type = self.visit_expression(node.final_expr)
        
        return last_type
    
    def visit_if_expr(self, node: IfExpr) -> SAAMType:
        """Visit if expression."""
        cond_type = self.visit_expression(node.condition)
        
        if cond_type.name != "Bool":
            self.error(f"If condition must be Bool, got {cond_type}", node.location)
        
        then_type = self.visit_expression(node.then_branch)
        
        if node.else_branch:
            else_type = self.visit_expression(node.else_branch)
            
            # Both branches must return compatible types
            if not then_type.is_compatible_with(else_type):
                self.error(
                    f"If branches have incompatible types: {then_type} vs {else_type}",
                    node.location
                )
                return BuiltinTypes.ERROR
        
        return then_type
    
    def visit_match_expr(self, node: MatchExpr) -> SAAMType:
        """Visit match expression."""
        subject_type = self.visit_expression(node.subject)
        
        if not node.arms:
            self.error("Match expression must have at least one arm", node.location)
            return BuiltinTypes.ERROR
        
        # Analyze each arm
        arm_types = []
        for arm in node.arms:
            # Check pattern compatibility
            if not self.check_pattern_compatibility(arm.pattern, subject_type):
                self.error(
                    f"Pattern incompatible with match subject type {subject_type}",
                    arm.pattern.location if arm.pattern else node.location
                )
            
            # Analyze guard
            if arm.guard:
                guard_type = self.visit_expression(arm.guard)
                if guard_type.name != "Bool":
                    self.error("Match guard must be Bool", arm.guard.location)
            
            # Analyze body
            arm_type = self.visit_expression(arm.body)
            arm_types.append(arm_type)
        
        # Check exhaustiveness
        self.check_match_exhaustiveness(subject_type, [arm.pattern for arm in node.arms], node.location)
        
        # All arms must have compatible types
        result_type = arm_types[0]
        for i, arm_type in enumerate(arm_types[1:], 1):
            if not result_type.is_compatible_with(arm_type):
                self.error(
                    f"Match arm {i+1} type {arm_type} incompatible with first arm type {result_type}",
                    node.location
                )
                return BuiltinTypes.ERROR
        
        return result_type
    
    def visit_array_literal(self, node: ArrayLiteral) -> SAAMType:
        """Visit array literal."""
        if not node.elements:
            return BuiltinTypes.array(BuiltinTypes.UNKNOWN)
        
        element_types = [self.visit_expression(elem) for elem in node.elements]
        
        # All elements must be compatible
        elem_type = element_types[0]
        for i, etype in enumerate(element_types[1:], 1):
            if not elem_type.is_compatible_with(etype):
                # Try the other direction
                if etype.is_compatible_with(elem_type):
                    elem_type = etype
                else:
                    self.error(
                        f"Array element {i+1} type {etype} incompatible with first element type {elem_type}",
                        node.location
                    )
                    return BuiltinTypes.ERROR
        
        return BuiltinTypes.array(elem_type)
    
    def visit_map_literal(self, node: MapLiteral) -> SAAMType:
        """Visit map literal."""
        if not node.pairs:
            return BuiltinTypes.map(BuiltinTypes.UNKNOWN, BuiltinTypes.UNKNOWN)
        
        key_types = []
        value_types = []
        
        for key_expr, value_expr in node.pairs:
            key_types.append(self.visit_expression(key_expr))
            value_types.append(self.visit_expression(value_expr))
        
        # All keys must be compatible, all values must be compatible
        key_type = key_types[0]
        value_type = value_types[0]
        
        for ktype in key_types[1:]:
            if not key_type.is_compatible_with(ktype):
                self.error("Map keys must have compatible types", node.location)
                return BuiltinTypes.ERROR
        
        for vtype in value_types[1:]:
            if not value_type.is_compatible_with(vtype):
                self.error("Map values must have compatible types", node.location)
                return BuiltinTypes.ERROR
        
        return BuiltinTypes.map(key_type, value_type)
    
    def visit_tuple_literal(self, node: TupleLiteral) -> SAAMType:
        """Visit tuple literal."""
        element_types = [self.visit_expression(elem) for elem in node.elements]
        tuple_type = SAAMType(TypeKind.TUPLE, "Tuple", element_types)
        return tuple_type
    
    def visit_lambda(self, node: Lambda) -> SAAMType:
        """Visit lambda expression."""
        self.symbol_table.push_scope(is_function=True)
        
        try:
            param_types = []
            for param in node.params:
                param_type = BuiltinTypes.ANY
                if param.param_type:
                    param_type = self.resolve_type_annotation(param.param_type)
                
                param_symbol = Symbol(
                    name=param.name,
                    type=param_type,
                    location=param.location or node.location,
                    is_mutable=param.is_mutable
                )
                self.symbol_table.define(param_symbol)
                param_types.append(param_type)
            
            # Analyze body
            body_type = self.visit_expression(node.body)
            
            # Create function type
            func_type = SAAMType(TypeKind.FUNCTION, "lambda", param_types + [body_type])
            return func_type
        
        finally:
            self.symbol_table.pop_scope()
    
    # === STATEMENT ANALYSIS ===
    
    def visit_statement(self, node: Statement):
        """Visit any statement."""
        if isinstance(node, VarDecl):
            self.visit_var_decl(node)
        elif isinstance(node, ExprStmt):
            self.visit_expression(node.expr)
        elif isinstance(node, IfStmt):
            self.visit_if_stmt(node)
        elif isinstance(node, WhileStmt):
            self.visit_while_stmt(node)
        elif isinstance(node, ForStmt):
            self.visit_for_stmt(node)
        elif isinstance(node, ReturnStmt):
            self.visit_return_stmt(node)
        elif isinstance(node, BreakStmt):
            self.visit_break_stmt(node)
        elif isinstance(node, ContinueStmt):
            self.visit_continue_stmt(node)
        else:
            self.error(f"Unknown statement type: {type(node)}", node.location)
    
    def visit_if_stmt(self, node: IfStmt):
        """Visit if statement."""
        cond_type = self.visit_expression(node.condition)
        
        if cond_type.name != "Bool":
            self.error(f"If condition must be Bool, got {cond_type}", node.location)
        
        self.visit_block(node.then_branch)
        
        if node.else_branch:
            if isinstance(node.else_branch, Block):
                self.visit_block(node.else_branch)
            else:
                self.visit_if_stmt(node.else_branch)
    
    def visit_while_stmt(self, node: WhileStmt):
        """Visit while statement."""
        cond_type = self.visit_expression(node.condition)
        
        if cond_type.name != "Bool":
            self.error(f"While condition must be Bool, got {cond_type}", node.location)
        
        self.visit_block(node.body)
    
    def visit_for_stmt(self, node: ForStmt):
        """Visit for statement."""
        iterable_type = self.visit_expression(node.iterable)
        
        # Check if iterable implements iterator trait
        # For now, just check for array types
        if iterable_type.kind != TypeKind.ARRAY:
            self.error(f"Cannot iterate over non-iterable type {iterable_type}", node.location)
            return
        
        # Get element type
        elem_type = iterable_type.args[0] if iterable_type.args else BuiltinTypes.ANY
        
        # Define loop variable
        if node.variable:
            var_symbol = Symbol(
                name=node.variable,
                type=elem_type,
                location=node.location
            )
            self.symbol_table.define(var_symbol)
        
        self.visit_block(node.body)
    
    def visit_return_stmt(self, node: ReturnStmt):
        """Visit return statement."""
        if not self.symbol_table.is_in_function():
            self.error("Return statement outside function", node.location)
            return
        
        if node.value:
            self.visit_expression(node.value)
    
    def visit_break_stmt(self, node: BreakStmt):
        """Visit break statement."""
        # Should check if we're in a loop, but that requires loop tracking
        if node.value:
            self.visit_expression(node.value)
    
    def visit_continue_stmt(self, node: ContinueStmt):
        """Visit continue statement."""
        # Should check if we're in a loop
        pass
    
    # === TYPE RESOLUTION ===
    
    def resolve_type_annotation(self, node: Type) -> SAAMType:
        """Resolve a type annotation to a concrete type."""
        if isinstance(node, PrimitiveType):
            builtin = BuiltinTypes.get_by_name(node.name)
            return builtin if builtin else BuiltinTypes.UNKNOWN
        
        elif isinstance(node, NamedType):
            # Look up user-defined type
            type_name = node.name.full_name() if hasattr(node.name, 'full_name') else str(node.name)
            symbol = self.symbol_table.resolve(type_name)
            
            if not symbol:
                self.error(f"Undefined type: {type_name}", node.location)
                return BuiltinTypes.ERROR
            
            result_type = symbol.type
            
            # Handle generic types
            if node.type_args:
                arg_types = [self.resolve_type_annotation(arg) for arg in node.type_args]
                result_type = copy.deepcopy(result_type)
                result_type.args = arg_types
            
            return result_type
        
        elif isinstance(node, ArrayType):
            elem_type = self.resolve_type_annotation(node.element_type)
            return BuiltinTypes.array(elem_type)
        
        elif isinstance(node, OptionType):
            inner_type = self.resolve_type_annotation(node.inner_type)
            return BuiltinTypes.option(inner_type)
        
        elif isinstance(node, ReferenceType):
            inner_type = self.resolve_type_annotation(node.inner_type)
            return BuiltinTypes.reference(inner_type, node.is_mutable)
        
        elif isinstance(node, FunctionType):
            param_types = [self.resolve_type_annotation(p) for p in node.param_types]
            return_type = self.resolve_type_annotation(node.return_type) if node.return_type else BuiltinTypes.NONE
            return SAAMType(TypeKind.FUNCTION, "fn", param_types + [return_type])
        
        else:
            self.error(f"Unknown type annotation: {type(node)}", node.location)
            return BuiltinTypes.ERROR
    
    # === TYPE CHECKING HELPERS ===
    
    def check_assignment_compatibility(self, target: SAAMType, source: SAAMType, location: SourceLocation) -> bool:
        """Check if source can be assigned to target."""
        if target.is_compatible_with(source):
            return True
        
        # Try neuroplastic morphing
        if source.is_neural and source.can_morph_to(target):
            return True
        
        self.error(f"Cannot assign {source} to {target}", location)
        return False
    
    def check_pattern_compatibility(self, pattern: Pattern, subject_type: SAAMType) -> bool:
        """Check if pattern is compatible with subject type."""
        # This would be much more complex in a real implementation
        # For now, just basic checks
        return True
    
    def check_match_exhaustiveness(self, subject_type: SAAMType, patterns: List[Pattern], location: SourceLocation):
        """Check if match patterns are exhaustive."""
        # This requires sophisticated analysis of pattern coverage
        # For now, just check for wildcard pattern
        has_wildcard = any(isinstance(p, WildcardPattern) for p in patterns)
        
        if not has_wildcard:
            self.warning("Match expression may not be exhaustive", location)
    
    def analyze_component_reactivity(self, component: ComponentDecl):
        """Analyze component for reactive patterns."""
        # This would analyze state dependencies, prop changes, etc.
        # Mark reactive variables
        for state_var in component.state:
            symbol = self.symbol_table.resolve(state_var.name)
            if symbol:
                symbol.is_reactive = True
    
    def suggest_similar_names(self, name: str) -> List[str]:
        """Suggest similar variable names for typos."""
        suggestions = []
        
        # Simple Levenshtein-like suggestions
        for symbol_name in self.symbol_table.current_scope.symbols:
            if abs(len(name) - len(symbol_name)) <= 2:
                # Count differing characters
                diff = sum(c1 != c2 for c1, c2 in zip(name, symbol_name))
                if diff <= 2:
                    suggestions.append(symbol_name)
        
        return suggestions[:3]  # Top 3 suggestions
    
    def check_unused_variables(self):
        """Check for unused variables and warn."""
        def check_scope(scope: Scope):
            for symbol in scope.symbols.values():
                if (symbol.name not in {"self"} and 
                    not symbol.name.startswith("_") and 
                    not symbol.is_used and
                    symbol.location.file != "<builtin>"):
                    self.warning(f"Unused variable: {symbol.name}", symbol.location)
        
        # Check all scopes (would need to track them)
        # For now, just check current
        check_scope(self.symbol_table.current_scope)
    
    def resolve_type_constraints(self):
        """Resolve type inference constraints."""
        # This would implement constraint solving for type inference
        # Complex algorithm involving unification, etc.
        pass


def analyze_program(program: Program) -> tuple[bool, Dict[str, List[str]]]:
    """
    Analyze a SAAAM program.
    
    Returns:
        (success, diagnostics) where:
        - success: True if no errors
        - diagnostics: {"errors": [...], "warnings": [...]}
    """
    analyzer = SemanticAnalyzer()
    success = analyzer.analyze(program)
    diagnostics = analyzer.get_diagnostics()
    
    return success, diagnostics


def analyze_file(filepath: str) -> tuple[bool, Dict[str, List[str]]]:
    """Analyze a .saaam file."""
    from .parser import parse_file
    
    try:
        program = parse_file(filepath)
        return analyze_program(program)
    except Exception as e:
        return False, {"errors": [f"Failed to parse {filepath}: {e}"], "warnings": []}


if __name__ == "__main__":
    # Test the semantic analyzer
    from .parser import parse
    
    test_code = """
    neural x = 42
    x ~> "hello world"
    
    component Counter {
        state count: Int = 0
        
        fn increment() {
            count += 1
        }
    }
    """
    
    try:
        program = parse(test_code)
        success, diagnostics = analyze_program(program)
        
        print(f"Analysis {'succeeded' if success else 'failed'}")
        
        if diagnostics['errors']:
            print("\nErrors:")
            for error in diagnostics['errors']:
                print(f"  {error}")
        
        if diagnostics['warnings']:
            print("\nWarnings:")
            for warning in diagnostics['warnings']:
                print(f"  {warning}")
                
    except Exception as e:
        print(f"Error: {e}")
