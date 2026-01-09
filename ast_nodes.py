"""
SAAAM Language - Abstract Syntax Tree Nodes
The Neural Architecture: Every node is a neuron, every tree is a thought.

This is where code becomes COGNITION.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Any
from enum import Enum, auto

try:
    from .tokens import SourceLocation, Token
except ImportError:
    from tokens import SourceLocation, Token


# === BASE NODES ===

@dataclass
class ASTNode(ABC):
    """Base class for all AST nodes - the fundamental neuron."""
    location: Optional[SourceLocation] = None
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Accept a visitor for traversal."""
        method_name = f'visit_{self.__class__.__name__}'
        method = getattr(visitor, method_name, visitor.generic_visit)
        return method(self)
    
    @abstractmethod
    def children(self) -> list[ASTNode]:
        """Get child nodes for traversal."""
        pass


@dataclass
class Expression(ASTNode):
    """Base class for all expressions - neurons that produce values."""
    inferred_type: Optional['Type'] = None
    
    def children(self) -> list[ASTNode]:
        return []


@dataclass
class Statement(ASTNode):
    """Base class for all statements - neurons that perform actions."""
    def children(self) -> list[ASTNode]:
        return []


@dataclass
class Declaration(Statement):
    """Base for declarations (variables, functions, types, etc.)."""
    name: str = ""
    is_public: bool = False
    doc_comment: Optional[str] = None


# === PROGRAM STRUCTURE ===

@dataclass
class Program(ASTNode):
    """Root node - the brain itself."""
    module_path: Optional[str] = None
    imports: list['Import'] = field(default_factory=list)
    declarations: list[Declaration] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.imports + self.declarations


@dataclass
class Import(ASTNode):
    """Use/import statement."""
    path: list[str] = field(default_factory=list)  # e.g., ['std', 'io']
    items: list[str] = field(default_factory=list)  # Specific items to import
    alias: Optional[str] = None
    is_wildcard: bool = False
    
    def children(self) -> list[ASTNode]:
        return []


# === LITERALS ===

@dataclass
class Literal(Expression):
    """Base for all literal values."""
    value: Any = None


@dataclass 
class IntegerLiteral(Literal):
    """Integer literal: 42, 0xFF, 0b1010."""
    value: int = 0
    base: int = 10  # 10, 16, 8, or 2


@dataclass
class FloatLiteral(Literal):
    """Float literal: 3.14, 1e-10."""
    value: float = 0.0


@dataclass
class StringLiteral(Literal):
    """String literal: "hello"."""
    value: str = ""
    is_template: bool = False  # Template literal with interpolation


@dataclass
class CharLiteral(Literal):
    """Character literal: 'x'."""
    value: str = ""


@dataclass
class BoolLiteral(Literal):
    """Boolean literal: true/false."""
    value: bool = False


@dataclass
class NoneLiteral(Literal):
    """None literal."""
    value: None = None


@dataclass
class ArrayLiteral(Expression):
    """Array literal: [1, 2, 3]."""
    elements: list[Expression] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.elements


@dataclass
class MapLiteral(Expression):
    """Map/dict literal: {key: value}."""
    pairs: list[tuple[Expression, Expression]] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return [k for k, v in self.pairs] + [v for k, v in self.pairs]


@dataclass
class TupleLiteral(Expression):
    """Tuple literal: (1, "hello", true)."""
    elements: list[Expression] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.elements


# === IDENTIFIERS & REFERENCES ===

@dataclass
class Identifier(Expression):
    """Simple identifier reference."""
    name: str = ""


@dataclass
class QualifiedName(Expression):
    """Qualified name: module::path::Name."""
    parts: list[str] = field(default_factory=list)
    
    def full_name(self) -> str:
        return '::'.join(self.parts)


# === TYPES ===

class TypeKind(Enum):
    """Categories of types in SAAAM."""
    PRIMITIVE = auto()
    STRUCT = auto()
    ENUM = auto()
    TRAIT = auto()
    FUNCTION = auto()
    ARRAY = auto()
    MAP = auto()
    TUPLE = auto()
    OPTION = auto()
    RESULT = auto()
    REFERENCE = auto()
    NEURAL = auto()
    UNKNOWN = auto()


@dataclass
class Type(ASTNode):
    """Base type node."""
    kind: TypeKind = TypeKind.UNKNOWN
    
    def children(self) -> list[ASTNode]:
        return []


@dataclass
class PrimitiveType(Type):
    """Built-in primitive type."""
    name: str = ""  # Int, Float, String, Bool, etc.
    kind: TypeKind = TypeKind.PRIMITIVE


@dataclass
class NamedType(Type):
    """User-defined type reference."""
    name: QualifiedName = None
    type_args: list[Type] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return [self.name] + self.type_args if self.name else self.type_args


@dataclass
class ArrayType(Type):
    """Array type: Array<T> or [T]."""
    element_type: Type = None
    size: Optional[int] = None  # Fixed size if known
    kind: TypeKind = TypeKind.ARRAY
    
    def children(self) -> list[ASTNode]:
        return [self.element_type] if self.element_type else []


@dataclass
class MapType(Type):
    """Map type: Map<K, V>."""
    key_type: Type = None
    value_type: Type = None
    kind: TypeKind = TypeKind.MAP
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.key_type:
            result.append(self.key_type)
        if self.value_type:
            result.append(self.value_type)
        return result


@dataclass
class TupleType(Type):
    """Tuple type: (A, B, C)."""
    element_types: list[Type] = field(default_factory=list)
    kind: TypeKind = TypeKind.TUPLE
    
    def children(self) -> list[ASTNode]:
        return self.element_types


@dataclass
class FunctionType(Type):
    """Function type: fn(A, B) -> C."""
    param_types: list[Type] = field(default_factory=list)
    return_type: Optional[Type] = None
    is_async: bool = False
    effects: list[str] = field(default_factory=list)
    kind: TypeKind = TypeKind.FUNCTION
    
    def children(self) -> list[ASTNode]:
        result = list(self.param_types)
        if self.return_type:
            result.append(self.return_type)
        return result


@dataclass
class ReferenceType(Type):
    """Reference type: &T or &mut T."""
    inner_type: Type = None
    is_mutable: bool = False
    kind: TypeKind = TypeKind.REFERENCE
    
    def children(self) -> list[ASTNode]:
        return [self.inner_type] if self.inner_type else []


@dataclass
class OptionType(Type):
    """Option type: Option<T> or T?."""
    inner_type: Type = None
    kind: TypeKind = TypeKind.OPTION
    
    def children(self) -> list[ASTNode]:
        return [self.inner_type] if self.inner_type else []


@dataclass
class ResultType(Type):
    """Result type: Result<T, E>."""
    ok_type: Type = None
    err_type: Type = None
    kind: TypeKind = TypeKind.RESULT
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.ok_type:
            result.append(self.ok_type)
        if self.err_type:
            result.append(self.err_type)
        return result


@dataclass
class NeuralType(Type):
    """Neuroplastic type that can morph."""
    current_type: Optional[Type] = None
    possible_types: list[Type] = field(default_factory=list)
    kind: TypeKind = TypeKind.NEURAL
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.current_type:
            result.append(self.current_type)
        result.extend(self.possible_types)
        return result


# === EXPRESSIONS ===

@dataclass
class BinaryOp(Expression):
    """Binary operation: a + b, x ~> y, etc."""
    left: Expression = None
    operator: str = ""
    right: Expression = None
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.left:
            result.append(self.left)
        if self.right:
            result.append(self.right)
        return result


@dataclass
class UnaryOp(Expression):
    """Unary operation: !x, -y, &z, *ptr."""
    operator: str = ""
    operand: Expression = None
    is_prefix: bool = True
    
    def children(self) -> list[ASTNode]:
        return [self.operand] if self.operand else []


@dataclass
class Call(Expression):
    """Function/method call: foo(a, b)."""
    callee: Expression = None
    arguments: list[Expression] = field(default_factory=list)
    type_args: list[Type] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.callee:
            result.append(self.callee)
        result.extend(self.arguments)
        result.extend(self.type_args)
        return result


@dataclass
class Index(Expression):
    """Index/subscript: arr[i]."""
    object: Expression = None
    index: Expression = None
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.object:
            result.append(self.object)
        if self.index:
            result.append(self.index)
        return result


@dataclass
class Member(Expression):
    """Member access: obj.field."""
    object: Expression = None
    member: str = ""
    
    def children(self) -> list[ASTNode]:
        return [self.object] if self.object else []


@dataclass
class TraitMember(Expression):
    """Trait/namespace access: Type::method."""
    type_expr: Expression = None
    member: str = ""
    
    def children(self) -> list[ASTNode]:
        return [self.type_expr] if self.type_expr else []


@dataclass
class IfExpr(Expression):
    """If expression (returns value)."""
    condition: Expression = None
    then_branch: Expression = None
    else_branch: Optional[Expression] = None
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.condition:
            result.append(self.condition)
        if self.then_branch:
            result.append(self.then_branch)
        if self.else_branch:
            result.append(self.else_branch)
        return result


@dataclass
class MatchArm:
    """Single arm in a match expression."""
    pattern: 'Pattern' = None
    guard: Optional[Expression] = None
    body: Expression = None


@dataclass
class MatchExpr(Expression):
    """Match expression."""
    subject: Expression = None
    arms: list[MatchArm] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.subject:
            result.append(self.subject)
        for arm in self.arms:
            if arm.pattern:
                result.append(arm.pattern)
            if arm.guard:
                result.append(arm.guard)
            if arm.body:
                result.append(arm.body)
        return result


@dataclass
class Lambda(Expression):
    """Lambda/closure: (x, y) => x + y."""
    params: list['Parameter'] = field(default_factory=list)
    body: Expression = None
    is_async: bool = False
    
    def children(self) -> list[ASTNode]:
        result = list(self.params)
        if self.body:
            result.append(self.body)
        return result


@dataclass
class Block(Expression):
    """Block expression: { statements... expr }."""
    statements: list[Statement] = field(default_factory=list)
    final_expr: Optional[Expression] = None
    
    def children(self) -> list[ASTNode]:
        result = list(self.statements)
        if self.final_expr:
            result.append(self.final_expr)
        return result


@dataclass
class RangeExpr(Expression):
    """Range expression: start..end or start..=end."""
    start: Optional[Expression] = None
    end: Optional[Expression] = None
    inclusive: bool = False
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.start:
            result.append(self.start)
        if self.end:
            result.append(self.end)
        return result


@dataclass
class AwaitExpr(Expression):
    """Await expression: expr <~ or await expr."""
    expr: Expression = None
    
    def children(self) -> list[ASTNode]:
        return [self.expr] if self.expr else []


@dataclass
class TryExpr(Expression):
    """Try expression with ? operator."""
    expr: Expression = None
    
    def children(self) -> list[ASTNode]:
        return [self.expr] if self.expr else []


@dataclass
class CastExpr(Expression):
    """Type cast: expr as Type."""
    expr: Expression = None
    target_type: Type = None
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.expr:
            result.append(self.expr)
        if self.target_type:
            result.append(self.target_type)
        return result


@dataclass
class Comprehension(Expression):
    """List/set/map comprehension."""
    element: Expression = None
    variable: str = ""
    iterable: Expression = None
    condition: Optional[Expression] = None
    is_map: bool = False
    key_expr: Optional[Expression] = None  # For map comprehensions
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.element:
            result.append(self.element)
        if self.iterable:
            result.append(self.iterable)
        if self.condition:
            result.append(self.condition)
        if self.key_expr:
            result.append(self.key_expr)
        return result


@dataclass
class ConstructExpr(Expression):
    """Struct/enum construction: Point { x: 1, y: 2 }."""
    type_name: Expression = None
    fields: list[tuple[str, Expression]] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.type_name:
            result.append(self.type_name)
        result.extend(v for _, v in self.fields)
        return result


# === PATTERNS ===

@dataclass
class Pattern(ASTNode):
    """Base for all patterns (used in match, let, etc.)."""
    def children(self) -> list[ASTNode]:
        return []


@dataclass
class WildcardPattern(Pattern):
    """Wildcard pattern: _."""
    pass


@dataclass
class IdentifierPattern(Pattern):
    """Identifier pattern: x."""
    name: str = ""
    is_mutable: bool = False


@dataclass
class LiteralPattern(Pattern):
    """Literal pattern: 42, "hello"."""
    value: Literal = None
    
    def children(self) -> list[ASTNode]:
        return [self.value] if self.value else []


@dataclass
class RangePattern(Pattern):
    """Range pattern: 1..10 or 'a'..='z'."""
    start: Literal = None
    end: Literal = None
    inclusive: bool = False
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.start:
            result.append(self.start)
        if self.end:
            result.append(self.end)
        return result


@dataclass
class TuplePattern(Pattern):
    """Tuple pattern: (a, b, c)."""
    elements: list[Pattern] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.elements


@dataclass
class ArrayPattern(Pattern):
    """Array pattern: [first, second, ...rest]."""
    elements: list[Pattern] = field(default_factory=list)
    rest: Optional[str] = None  # Name for rest elements
    
    def children(self) -> list[ASTNode]:
        return self.elements


@dataclass
class StructPattern(Pattern):
    """Struct pattern: Point { x, y }."""
    type_name: str = ""
    fields: list[tuple[str, Pattern]] = field(default_factory=list)
    has_rest: bool = False
    
    def children(self) -> list[ASTNode]:
        return [p for _, p in self.fields]


@dataclass
class EnumPattern(Pattern):
    """Enum variant pattern: Some(x), Ok(value)."""
    variant: str = ""
    fields: list[Pattern] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.fields


@dataclass
class OrPattern(Pattern):
    """Or pattern: A | B | C."""
    patterns: list[Pattern] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.patterns


# === STATEMENTS ===

@dataclass
class ExprStmt(Statement):
    """Expression statement."""
    expr: Expression = None
    
    def children(self) -> list[ASTNode]:
        return [self.expr] if self.expr else []


@dataclass
class VarDecl(Declaration):
    """Variable declaration: let/var/const/neural/gc."""
    name: str = ""
    var_type: Optional[Type] = None
    initializer: Optional[Expression] = None
    is_mutable: bool = False
    is_const: bool = False
    is_neural: bool = False
    is_gc: bool = False
    is_strict: bool = False
    is_heap: bool = False
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.var_type:
            result.append(self.var_type)
        if self.initializer:
            result.append(self.initializer)
        return result


@dataclass
class Parameter(ASTNode):
    """Function parameter."""
    name: str = ""
    param_type: Optional[Type] = None
    default_value: Optional[Expression] = None
    is_variadic: bool = False
    is_mutable: bool = False
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.param_type:
            result.append(self.param_type)
        if self.default_value:
            result.append(self.default_value)
        return result


@dataclass
class FunctionDecl(Declaration):
    """Function declaration."""
    name: str = ""
    params: list[Parameter] = field(default_factory=list)
    return_type: Optional[Type] = None
    body: Optional[Block] = None
    is_async: bool = False
    is_const: bool = False  # Compile-time function
    type_params: list[str] = field(default_factory=list)
    where_clause: list[tuple[str, list[str]]] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        result = list(self.params)
        if self.return_type:
            result.append(self.return_type)
        if self.body:
            result.append(self.body)
        return result


@dataclass
class StructField(ASTNode):
    """Field in a struct."""
    name: str = ""
    field_type: Type = None
    default_value: Optional[Expression] = None
    is_public: bool = False
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.field_type:
            result.append(self.field_type)
        if self.default_value:
            result.append(self.default_value)
        return result


@dataclass
class StructDecl(Declaration):
    """Struct declaration."""
    name: str = ""
    fields: list[StructField] = field(default_factory=list)
    type_params: list[str] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.fields


@dataclass
class EnumVariant(ASTNode):
    """Variant in an enum."""
    name: str = ""
    fields: list[tuple[Optional[str], Type]] = field(default_factory=list)
    value: Optional[Expression] = None
    
    def children(self) -> list[ASTNode]:
        result = [t for _, t in self.fields]
        if self.value:
            result.append(self.value)
        return result


@dataclass
class EnumDecl(Declaration):
    """Enum declaration."""
    name: str = ""
    variants: list[EnumVariant] = field(default_factory=list)
    type_params: list[str] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.variants


@dataclass
class TraitDecl(Declaration):
    """Trait declaration."""
    name: str = ""
    methods: list[FunctionDecl] = field(default_factory=list)
    type_params: list[str] = field(default_factory=list)
    super_traits: list[str] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return self.methods


@dataclass
class ImplDecl(Declaration):
    """Implementation block."""
    target_type: Type = None
    trait_name: Optional[str] = None
    methods: list[FunctionDecl] = field(default_factory=list)
    type_params: list[str] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.target_type:
            result.append(self.target_type)
        result.extend(self.methods)
        return result


@dataclass
class TypeAlias(Declaration):
    """Type alias: type ID = String."""
    name: str = ""
    aliased_type: Type = None
    type_params: list[str] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return [self.aliased_type] if self.aliased_type else []


# === CONTROL FLOW STATEMENTS ===

@dataclass
class IfStmt(Statement):
    """If statement."""
    condition: Expression = None
    then_branch: Block = None
    else_branch: Optional[Union[Block, 'IfStmt']] = None
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.condition:
            result.append(self.condition)
        if self.then_branch:
            result.append(self.then_branch)
        if self.else_branch:
            result.append(self.else_branch)
        return result


@dataclass
class WhileStmt(Statement):
    """While loop."""
    condition: Expression = None
    body: Block = None
    else_branch: Optional[Block] = None  # Executed if loop never runs
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.condition:
            result.append(self.condition)
        if self.body:
            result.append(self.body)
        if self.else_branch:
            result.append(self.else_branch)
        return result


@dataclass
class ForStmt(Statement):
    """For loop."""
    variable: str = ""
    pattern: Optional[Pattern] = None
    iterable: Expression = None
    body: Block = None
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.pattern:
            result.append(self.pattern)
        if self.iterable:
            result.append(self.iterable)
        if self.body:
            result.append(self.body)
        return result


@dataclass
class LoopStmt(Statement):
    """Infinite loop."""
    body: Block = None
    label: Optional[str] = None
    
    def children(self) -> list[ASTNode]:
        return [self.body] if self.body else []


@dataclass
class BreakStmt(Statement):
    """Break statement."""
    label: Optional[str] = None
    value: Optional[Expression] = None
    
    def children(self) -> list[ASTNode]:
        return [self.value] if self.value else []


@dataclass
class ContinueStmt(Statement):
    """Continue statement."""
    label: Optional[str] = None
    
    def children(self) -> list[ASTNode]:
        return []


@dataclass
class ReturnStmt(Statement):
    """Return statement."""
    value: Optional[Expression] = None
    
    def children(self) -> list[ASTNode]:
        return [self.value] if self.value else []


@dataclass
class YieldStmt(Statement):
    """Yield statement (for generators)."""
    value: Optional[Expression] = None
    
    def children(self) -> list[ASTNode]:
        return [self.value] if self.value else []


@dataclass
class TryStmt(Statement):
    """Try-catch-finally statement."""
    body: Block = None
    catches: list[tuple[str, Optional[Type], Block]] = field(default_factory=list)
    finally_block: Optional[Block] = None
    
    def children(self) -> list[ASTNode]:
        result = []
        if self.body:
            result.append(self.body)
        for name, typ, block in self.catches:
            if typ:
                result.append(typ)
            result.append(block)
        if self.finally_block:
            result.append(self.finally_block)
        return result


# === CONCURRENCY ===

@dataclass
class SpawnStmt(Statement):
    """Spawn a task/thread."""
    body: Block = None
    
    def children(self) -> list[ASTNode]:
        return [self.body] if self.body else []


@dataclass
class ParallelStmt(Statement):
    """Parallel execution block."""
    tasks: list[tuple[str, Expression]] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        return [expr for _, expr in self.tasks]


@dataclass
class ChannelDecl(Declaration):
    """Channel declaration."""
    name: str = ""
    element_type: Type = None
    capacity: Optional[int] = None
    
    def children(self) -> list[ASTNode]:
        return [self.element_type] if self.element_type else []


@dataclass
class ActorDecl(Declaration):
    """Actor declaration."""
    name: str = ""
    state: list[VarDecl] = field(default_factory=list)
    receives: list[tuple[str, Block]] = field(default_factory=list)
    
    def children(self) -> list[ASTNode]:
        result = list(self.state)
        result.extend(block for _, block in self.receives)
        return result


# === COMPONENTS (React-style) ===

@dataclass
class ComponentDecl(Declaration):
    """Component declaration."""
    name: str = ""
    state: list[VarDecl] = field(default_factory=list)
    props: list[VarDecl] = field(default_factory=list)
    methods: list[FunctionDecl] = field(default_factory=list)
    lifecycle: dict[str, Block] = field(default_factory=dict)
    render: Optional['JSXElement'] = None
    
    def children(self) -> list[ASTNode]:
        result = list(self.state) + list(self.props) + list(self.methods)
        result.extend(self.lifecycle.values())
        if self.render:
            result.append(self.render)
        return result


@dataclass
class JSXElement(Expression):
    """JSX element: <div>...</div>."""
    tag: str = ""
    attributes: list[tuple[str, Expression]] = field(default_factory=list)
    children: list[Union['JSXElement', Expression]] = field(default_factory=list)
    is_self_closing: bool = False
    
    def children(self) -> list[ASTNode]:
        result = [v for _, v in self.attributes]
        result.extend(self.children)
        return result


# === NEURAL BLOCKS ===

@dataclass
class NeuralBlockDecl(Declaration):
    """Neural block declaration."""
    name: str = ""
    weights: list[VarDecl] = field(default_factory=list)
    train_mode: Optional[Block] = None
    infer_mode: Optional[Block] = None
    
    def children(self) -> list[ASTNode]:
        result = list(self.weights)
        if self.train_mode:
            result.append(self.train_mode)
        if self.infer_mode:
            result.append(self.infer_mode)
        return result


# === VISITOR PATTERN ===

class ASTVisitor:
    """Visitor for traversing the AST."""
    
    def visit(self, node: ASTNode) -> Any:
        """Visit a node."""
        return node.accept(self)
    
    def generic_visit(self, node: ASTNode) -> Any:
        """Default visitor that visits all children."""
        for child in node.children():
            self.visit(child)
        return None


class ASTTransformer(ASTVisitor):
    """Transformer that can modify the AST."""
    
    def generic_visit(self, node: ASTNode) -> ASTNode:
        """Transform all children and return the node."""
        for i, child in enumerate(node.children()):
            transformed = self.visit(child)
            # Subclasses should handle replacing children
        return node
