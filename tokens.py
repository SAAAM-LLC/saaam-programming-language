"""
SAAAM Language - Token Definitions
The atomic building blocks of our language.
No tokenization in the traditional sense - these are SYNAPSE SIGNALS!
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional


class TokenType(Enum):
    """Every token type in the SAAAM language."""
    
    # === LITERALS ===
    INTEGER = auto()          # 42, 0xFF, 0b1010, 0o755
    FLOAT = auto()            # 3.14, 1e-10, .5
    STRING = auto()           # "hello", 'world', `template {x}`
    CHAR = auto()             # 'a', '\n'
    BOOL = auto()             # true, false
    NONE = auto()             # None
    
    # === IDENTIFIERS ===
    IDENTIFIER = auto()       # variable_name, TypeName
    TYPE_IDENTIFIER = auto()  # Capitalized identifiers treated as types
    
    # === KEYWORDS - VARIABLES ===
    LET = auto()              # let (immutable)
    VAR = auto()              # var (mutable)
    CONST = auto()            # const (compile-time constant)
    NEURAL = auto()           # neural (neuroplastic type)
    GC = auto()               # gc (garbage collected)
    STRICT = auto()           # strict (no type morphing)
    HEAP = auto()             # heap (heap allocation)
    
    # === KEYWORDS - FUNCTIONS ===
    FN = auto()               # fn
    ASYNC = auto()            # async
    AWAIT = auto()            # await
    RETURN = auto()           # return
    YIELD = auto()            # yield
    PUB = auto()              # pub (public)
    
    # === KEYWORDS - CONTROL FLOW ===
    IF = auto()               # if
    ELSE = auto()             # else
    ELIF = auto()             # elif
    MATCH = auto()            # match
    FOR = auto()              # for
    WHILE = auto()            # while
    LOOP = auto()             # loop
    BREAK = auto()            # break
    CONTINUE = auto()         # continue
    IN = auto()               # in
    WHERE = auto()            # where (pattern guard)
    
    # === KEYWORDS - TYPES ===
    STRUCT = auto()           # struct
    ENUM = auto()             # enum
    TRAIT = auto()            # trait
    IMPL = auto()             # impl
    TYPE = auto()             # type (alias)
    SELF = auto()             # self
    SELF_TYPE = auto()        # Self (the type)
    
    # === KEYWORDS - CONCURRENCY ===
    SPAWN = auto()            # spawn
    PARALLEL = auto()         # parallel
    CHAN = auto()             # chan (channel)
    ACTOR = auto()            # actor
    RECEIVE = auto()          # receive
    SEND = auto()             # send
    
    # === KEYWORDS - MEMORY ===
    MOVE = auto()             # move
    COPY = auto()             # copy
    DROP = auto()             # drop
    ARENA = auto()            # arena
    REGION = auto()           # region
    
    # === KEYWORDS - ERROR HANDLING ===
    TRY = auto()              # try
    CATCH = auto()            # catch
    FINALLY = auto()          # finally
    THROW = auto()            # throw
    OK = auto()               # Ok
    ERR = auto()              # Err
    SOME = auto()             # Some
    
    # === KEYWORDS - MODULES ===
    MODULE = auto()           # module
    USE = auto()              # use
    AS = auto()               # as
    FROM = auto()             # from
    EXPORT = auto()           # export
    
    # === KEYWORDS - COMPONENTS ===
    COMPONENT = auto()        # component
    STATE = auto()            # state
    PROPS = auto()            # props
    RENDER = auto()           # render
    ON = auto()               # on (lifecycle)
    MOUNT = auto()            # mount
    UNMOUNT = auto()          # unmount
    UPDATE = auto()           # update
    
    # === KEYWORDS - NEURAL ===
    BLOCK = auto()            # block (neural block)
    TRAIN = auto()            # train
    INFER = auto()            # infer
    MODE = auto()             # mode
    WEIGHTS = auto()          # weights
    FORWARD = auto()          # forward
    BACKWARD = auto()         # backward
    
    # === KEYWORDS - EFFECTS ===
    WITH = auto()             # with (effect annotation)
    PURE = auto()             # pure
    
    # === KEYWORDS - OTHER ===
    AND = auto()              # and
    OR = auto()               # or
    NOT = auto()              # not
    IS = auto()               # is
    TRUE = auto()             # true
    FALSE = auto()            # false
    
    # === STANDARD OPERATORS ===
    PLUS = auto()             # +
    MINUS = auto()            # -
    STAR = auto()             # *
    SLASH = auto()            # /
    PERCENT = auto()          # %
    POWER = auto()            # **
    FLOOR_DIV = auto()        # //
    
    # === COMPARISON OPERATORS ===
    EQ = auto()               # ==
    NE = auto()               # !=
    LT = auto()               # <
    LE = auto()               # <=
    GT = auto()               # >
    GE = auto()               # >=
    
    # === ASSIGNMENT OPERATORS ===
    ASSIGN = auto()           # =
    PLUS_ASSIGN = auto()      # +=
    MINUS_ASSIGN = auto()     # -=
    STAR_ASSIGN = auto()      # *=
    SLASH_ASSIGN = auto()     # /=
    PERCENT_ASSIGN = auto()   # %=
    
    # === SYNAPSE OPERATORS (THE MAGIC!) ===
    MORPH = auto()            # ~> (neuroplastic assignment)
    BIND = auto()             # <=> (bidirectional binding)
    FLOW = auto()             # -> (data flow / return type)
    ARROW = auto()            # => (lambda arrow)
    PARALLEL_PIPE = auto()    # |> (parallel pipe)
    REVERSE_FLOW = auto()     # <~ (reverse flow / await)
    INJECT = auto()           # @> (dependency injection)
    TRAIT_IMPL = auto()       # :: (namespace / trait impl)
    
    # === BITWISE OPERATORS ===
    BIT_AND = auto()          # &
    BIT_OR = auto()           # |
    BIT_XOR = auto()          # ^
    BIT_NOT = auto()          # ~
    LSHIFT = auto()           # <<
    RSHIFT = auto()           # >>
    
    # === LOGICAL OPERATORS ===
    LOGICAL_AND = auto()      # &&
    LOGICAL_OR = auto()       # ||
    LOGICAL_NOT = auto()      # !
    
    # === DELIMITERS ===
    LPAREN = auto()           # (
    RPAREN = auto()           # )
    LBRACKET = auto()         # [
    RBRACKET = auto()         # ]
    LBRACE = auto()           # {
    RBRACE = auto()           # }
    
    # === PUNCTUATION ===
    COMMA = auto()            # ,
    COLON = auto()            # :
    SEMICOLON = auto()        # ;
    DOT = auto()              # .
    RANGE = auto()            # ..
    RANGE_INCLUSIVE = auto()  # ..=
    ELLIPSIS = auto()         # ...
    QUESTION = auto()         # ?
    AT = auto()               # @
    HASH = auto()             # #
    DOLLAR = auto()           # $
    UNDERSCORE = auto()       # _
    BACKSLASH = auto()        # \
    PIPE = auto()             # | (also bitwise or, context-dependent)
    
    # === SPECIAL ===
    NEWLINE = auto()          # Significant in some contexts
    INDENT = auto()           # For Python-style blocks
    DEDENT = auto()           # End of indented block
    EOF = auto()              # End of file
    COMMENT = auto()          # // or # or /* */
    DOC_COMMENT = auto()      # /// or /** */
    
    # === JSX/TEMPLATE LITERALS ===
    JSX_OPEN = auto()         # <tag
    JSX_CLOSE = auto()        # </tag>
    JSX_SELF_CLOSE = auto()   # />
    TEMPLATE_START = auto()   # `
    TEMPLATE_EXPR_START = auto()  # {
    TEMPLATE_EXPR_END = auto()    # }
    
    # === ANNOTATIONS ===
    ANNOTATION = auto()       # #[...]


@dataclass
class SourceLocation:
    """Where in the source code this token lives."""
    file: str
    line: int
    column: int
    offset: int  # Byte offset from start of file
    
    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class Token:
    """
    A single token in the SAAAM language.
    These are the synaptic signals that flow through our compiler!
    """
    type: TokenType
    value: Any
    location: SourceLocation
    raw: str  # Original source text
    
    # Metadata for enhanced error messages
    leading_whitespace: str = ""
    trailing_whitespace: str = ""
    
    def __str__(self) -> str:
        return f"Token({self.type.name}, {repr(self.value)}, {self.location})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def is_operator(self) -> bool:
        """Check if this is any kind of operator."""
        return self.type in {
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.POWER, TokenType.FLOOR_DIV,
            TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE,
            TokenType.GT, TokenType.GE, TokenType.MORPH, TokenType.BIND,
            TokenType.FLOW, TokenType.ARROW, TokenType.PARALLEL_PIPE,
            TokenType.REVERSE_FLOW, TokenType.INJECT, TokenType.TRAIT_IMPL,
            TokenType.BIT_AND, TokenType.BIT_OR, TokenType.BIT_XOR,
            TokenType.LOGICAL_AND, TokenType.LOGICAL_OR,
        }
    
    def is_synapse_operator(self) -> bool:
        """Check if this is one of our special synapse operators."""
        return self.type in {
            TokenType.MORPH, TokenType.BIND, TokenType.FLOW,
            TokenType.PARALLEL_PIPE, TokenType.REVERSE_FLOW,
            TokenType.INJECT,
        }
    
    def is_keyword(self) -> bool:
        """Check if this is a keyword."""
        return self.type.name in KEYWORDS
    
    def is_literal(self) -> bool:
        """Check if this is a literal value."""
        return self.type in {
            TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING,
            TokenType.CHAR, TokenType.BOOL, TokenType.NONE,
        }


# Keyword mapping - string to TokenType
KEYWORDS: dict[str, TokenType] = {
    # Variables
    "let": TokenType.LET,
    "var": TokenType.VAR,
    "const": TokenType.CONST,
    "neural": TokenType.NEURAL,
    "gc": TokenType.GC,
    "strict": TokenType.STRICT,
    "heap": TokenType.HEAP,
    
    # Functions
    "fn": TokenType.FN,
    "async": TokenType.ASYNC,
    "await": TokenType.AWAIT,
    "return": TokenType.RETURN,
    "yield": TokenType.YIELD,
    "pub": TokenType.PUB,
    
    # Control flow
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "elif": TokenType.ELIF,
    "match": TokenType.MATCH,
    "for": TokenType.FOR,
    "while": TokenType.WHILE,
    "loop": TokenType.LOOP,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "in": TokenType.IN,
    "where": TokenType.WHERE,
    
    # Types
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "trait": TokenType.TRAIT,
    "impl": TokenType.IMPL,
    "type": TokenType.TYPE,
    "self": TokenType.SELF,
    "Self": TokenType.SELF_TYPE,
    
    # Concurrency
    "spawn": TokenType.SPAWN,
    "parallel": TokenType.PARALLEL,
    "chan": TokenType.CHAN,
    "actor": TokenType.ACTOR,
    "receive": TokenType.RECEIVE,
    "send": TokenType.SEND,
    
    # Memory
    "move": TokenType.MOVE,
    "copy": TokenType.COPY,
    "drop": TokenType.DROP,
    "arena": TokenType.ARENA,
    "region": TokenType.REGION,
    
    # Error handling
    "try": TokenType.TRY,
    "catch": TokenType.CATCH,
    "finally": TokenType.FINALLY,
    "throw": TokenType.THROW,
    "Ok": TokenType.OK,
    "Err": TokenType.ERR,
    "Some": TokenType.SOME,
    "None": TokenType.NONE,
    
    # Modules
    "module": TokenType.MODULE,
    "use": TokenType.USE,
    "as": TokenType.AS,
    "from": TokenType.FROM,
    "export": TokenType.EXPORT,
    
    # Components
    "component": TokenType.COMPONENT,
    "state": TokenType.STATE,
    "props": TokenType.PROPS,
    "render": TokenType.RENDER,
    "on": TokenType.ON,
    "mount": TokenType.MOUNT,
    "unmount": TokenType.UNMOUNT,
    "update": TokenType.UPDATE,
    
    # Neural
    "block": TokenType.BLOCK,
    "train": TokenType.TRAIN,
    "infer": TokenType.INFER,
    "mode": TokenType.MODE,
    "weights": TokenType.WEIGHTS,
    "forward": TokenType.FORWARD,
    "backward": TokenType.BACKWARD,
    
    # Effects
    "with": TokenType.WITH,
    "pure": TokenType.PURE,
    
    # Logical/Boolean
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "is": TokenType.IS,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
}


# Operator precedence (higher = binds tighter)
PRECEDENCE: dict[TokenType, int] = {
    # Assignment (lowest)
    TokenType.ASSIGN: 1,
    TokenType.PLUS_ASSIGN: 1,
    TokenType.MINUS_ASSIGN: 1,
    TokenType.STAR_ASSIGN: 1,
    TokenType.SLASH_ASSIGN: 1,
    TokenType.PERCENT_ASSIGN: 1,
    
    # Synapse operators
    TokenType.MORPH: 2,
    TokenType.BIND: 2,
    TokenType.INJECT: 2,
    
    # Logical OR
    TokenType.LOGICAL_OR: 3,
    TokenType.OR: 3,
    
    # Logical AND
    TokenType.LOGICAL_AND: 4,
    TokenType.AND: 4,
    
    # Bitwise OR
    TokenType.BIT_OR: 5,
    TokenType.PIPE: 5,
    
    # Bitwise XOR
    TokenType.BIT_XOR: 6,
    
    # Bitwise AND
    TokenType.BIT_AND: 7,
    
    # Equality
    TokenType.EQ: 8,
    TokenType.NE: 8,
    
    # Comparison
    TokenType.LT: 9,
    TokenType.LE: 9,
    TokenType.GT: 9,
    TokenType.GE: 9,
    
    # Bit shift
    TokenType.LSHIFT: 10,
    TokenType.RSHIFT: 10,
    
    # Flow operators
    TokenType.FLOW: 11,
    TokenType.PARALLEL_PIPE: 11,
    TokenType.REVERSE_FLOW: 11,
    
    # Range
    TokenType.RANGE: 12,
    TokenType.RANGE_INCLUSIVE: 12,
    
    # Addition/Subtraction
    TokenType.PLUS: 13,
    TokenType.MINUS: 13,
    
    # Multiplication/Division
    TokenType.STAR: 14,
    TokenType.SLASH: 14,
    TokenType.PERCENT: 14,
    TokenType.FLOOR_DIV: 14,
    
    # Power (right-associative)
    TokenType.POWER: 15,
    
    # Unary (highest for operators)
    TokenType.LOGICAL_NOT: 16,
    TokenType.BIT_NOT: 16,
    
    # Member access
    TokenType.DOT: 17,
    TokenType.TRAIT_IMPL: 17,
    
    # Call/Index
    TokenType.LPAREN: 18,
    TokenType.LBRACKET: 18,
}


# Right-associative operators
RIGHT_ASSOCIATIVE: set[TokenType] = {
    TokenType.POWER,
    TokenType.ASSIGN,
    TokenType.PLUS_ASSIGN,
    TokenType.MINUS_ASSIGN,
    TokenType.STAR_ASSIGN,
    TokenType.SLASH_ASSIGN,
    TokenType.PERCENT_ASSIGN,
    TokenType.ARROW,
}


def get_precedence(token_type: TokenType) -> int:
    """Get the precedence of an operator. Returns 0 if not an operator."""
    return PRECEDENCE.get(token_type, 0)


def is_right_associative(token_type: TokenType) -> bool:
    """Check if an operator is right-associative."""
    return token_type in RIGHT_ASSOCIATIVE
