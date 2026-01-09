"""
SAAAM Language - Lexer
The Neural Gateway: Transforms raw source code into token streams.
This isn't just tokenization - it's SYNAPTOGENESIS!
Every character becomes a signal, every token a neuron firing.
"""

from typing import Iterator, Optional
from dataclasses import dataclass
import re

try:
    from .tokens import Token, TokenType, SourceLocation, KEYWORDS
except ImportError:
    from tokens import Token, TokenType, SourceLocation, KEYWORDS


class LexerError(Exception):
    """When the neural signals go haywire."""
    def __init__(self, message: str, location: SourceLocation):
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}")


@dataclass
class LexerState:
    """Current state of the lexer - our neural membrane potential."""
    source: str
    filename: str
    pos: int = 0
    line: int = 1
    column: int = 1
    
    # For template literal parsing
    in_template: bool = False
    template_depth: int = 0
    brace_depth: int = 0
    
    # For JSX parsing
    in_jsx: bool = False
    jsx_tag_stack: list[str] = None
    
    # For significant whitespace (Python-style optional mode)
    indent_stack: list[int] = None
    at_line_start: bool = True
    
    def __post_init__(self):
        if self.jsx_tag_stack is None:
            self.jsx_tag_stack = []
        if self.indent_stack is None:
            self.indent_stack = [0]


class Lexer:
    """
    The SAAAM Lexer - Neural Signal Transducer
    
    Transforms source code into a stream of tokens,
    handling all our synapse operators and neuroplastic syntax.
    """
    
    def __init__(self, source: str, filename: str = "<input>"):
        self.state = LexerState(source=source, filename=filename)
        self._peeked: list[Token] = []
    
    @property
    def source(self) -> str:
        return self.state.source
    
    @property
    def pos(self) -> int:
        return self.state.pos
    
    @pos.setter
    def pos(self, value: int):
        self.state.pos = value
    
    def current_location(self) -> SourceLocation:
        """Get current position in source."""
        return SourceLocation(
            file=self.state.filename,
            line=self.state.line,
            column=self.state.column,
            offset=self.state.pos
        )
    
    def peek_char(self, offset: int = 0) -> str:
        """Look at a character without consuming it."""
        idx = self.state.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return '\0'  # EOF sentinel
    
    def advance(self) -> str:
        """Consume and return the current character."""
        if self.state.pos >= len(self.source):
            return '\0'
        
        char = self.source[self.state.pos]
        self.state.pos += 1
        
        if char == '\n':
            self.state.line += 1
            self.state.column = 1
            self.state.at_line_start = True
        else:
            self.state.column += 1
            if not char.isspace():
                self.state.at_line_start = False
        
        return char
    
    def skip_whitespace(self) -> str:
        """Skip whitespace, return what was skipped."""
        skipped = []
        while self.peek_char().isspace() and self.peek_char() != '\n':
            skipped.append(self.advance())
        return ''.join(skipped)
    
    def skip_to_eol(self):
        """Skip to end of line (for comments)."""
        while self.peek_char() not in ('\n', '\0'):
            self.advance()
    
    def error(self, message: str) -> LexerError:
        """Create a lexer error at current position."""
        return LexerError(message, self.current_location())
    
    def make_token(self, token_type: TokenType, value: any, 
                   start_loc: SourceLocation, raw: str) -> Token:
        """Create a token with all the metadata."""
        return Token(
            type=token_type,
            value=value,
            location=start_loc,
            raw=raw
        )
    
    # === NUMBER PARSING ===
    
    def scan_number(self) -> Token:
        """Parse integer or float literal."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        # Check for different bases
        if self.peek_char() == '0':
            next_char = self.peek_char(1).lower()
            if next_char == 'x':
                return self.scan_hex()
            elif next_char == 'b':
                return self.scan_binary()
            elif next_char == 'o':
                return self.scan_octal()
        
        # Regular decimal number
        has_dot = False
        has_exp = False
        
        while True:
            c = self.peek_char()
            if c.isdigit() or c == '_':
                self.advance()
            elif c == '.' and not has_dot and not has_exp:
                # Check if it's a range operator
                if self.peek_char(1) == '.':
                    break
                has_dot = True
                self.advance()
            elif c in 'eE' and not has_exp:
                has_exp = True
                self.advance()
                if self.peek_char() in '+-':
                    self.advance()
            else:
                break
        
        raw = self.source[start_pos:self.state.pos]
        clean = raw.replace('_', '')
        
        if has_dot or has_exp:
            value = float(clean)
            return self.make_token(TokenType.FLOAT, value, start_loc, raw)
        else:
            value = int(clean)
            return self.make_token(TokenType.INTEGER, value, start_loc, raw)
    
    def scan_hex(self) -> Token:
        """Parse hexadecimal literal."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        self.advance()  # 0
        self.advance()  # x
        
        if not self.peek_char().isalnum():
            raise self.error("Expected hexadecimal digit after '0x'")
        
        while self.peek_char().isalnum() or self.peek_char() == '_':
            self.advance()
        
        raw = self.source[start_pos:self.state.pos]
        clean = raw.replace('_', '')
        value = int(clean, 16)
        return self.make_token(TokenType.INTEGER, value, start_loc, raw)
    
    def scan_binary(self) -> Token:
        """Parse binary literal."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        self.advance()  # 0
        self.advance()  # b
        
        if self.peek_char() not in '01':
            raise self.error("Expected binary digit after '0b'")
        
        while self.peek_char() in '01_':
            self.advance()
        
        raw = self.source[start_pos:self.state.pos]
        clean = raw.replace('_', '')
        value = int(clean, 2)
        return self.make_token(TokenType.INTEGER, value, start_loc, raw)
    
    def scan_octal(self) -> Token:
        """Parse octal literal."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        self.advance()  # 0
        self.advance()  # o
        
        if self.peek_char() not in '01234567':
            raise self.error("Expected octal digit after '0o'")
        
        while self.peek_char() in '01234567_':
            self.advance()
        
        raw = self.source[start_pos:self.state.pos]
        clean = raw.replace('_', '')
        value = int(clean, 8)
        return self.make_token(TokenType.INTEGER, value, start_loc, raw)
    
    # === STRING PARSING ===
    
    def scan_string(self, quote: str) -> Token:
        """Parse string literal (single, double, or backtick)."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        # Check for triple-quoted strings
        if self.peek_char(1) == quote and self.peek_char(2) == quote:
            return self.scan_triple_string(quote)
        
        self.advance()  # Opening quote
        
        chars = []
        while True:
            c = self.peek_char()
            
            if c == '\0':
                raise self.error("Unterminated string literal")
            
            if c == '\n' and quote != '`':
                raise self.error("Newline in string literal")
            
            if c == quote:
                self.advance()
                break
            
            if c == '\\':
                chars.append(self.scan_escape())
            elif c == '{' and quote == '`':
                # Template expression - for now, just include literally
                # Full template parsing happens at a higher level
                chars.append(self.advance())
            else:
                chars.append(self.advance())
        
        raw = self.source[start_pos:self.state.pos]
        value = ''.join(chars)
        return self.make_token(TokenType.STRING, value, start_loc, raw)
    
    def scan_triple_string(self, quote: str) -> Token:
        """Parse triple-quoted string (multiline)."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        # Consume opening quotes
        self.advance()
        self.advance()
        self.advance()
        
        chars = []
        while True:
            if self.peek_char() == '\0':
                raise self.error("Unterminated triple-quoted string")
            
            if (self.peek_char() == quote and 
                self.peek_char(1) == quote and 
                self.peek_char(2) == quote):
                self.advance()
                self.advance()
                self.advance()
                break
            
            if self.peek_char() == '\\':
                chars.append(self.scan_escape())
            else:
                chars.append(self.advance())
        
        raw = self.source[start_pos:self.state.pos]
        value = ''.join(chars)
        return self.make_token(TokenType.STRING, value, start_loc, raw)
    
    def scan_escape(self) -> str:
        """Parse escape sequence in string."""
        self.advance()  # backslash
        
        c = self.advance()
        escapes = {
            'n': '\n',
            't': '\t',
            'r': '\r',
            '\\': '\\',
            "'": "'",
            '"': '"',
            '`': '`',
            '0': '\0',
            'a': '\a',
            'b': '\b',
            'f': '\f',
            'v': '\v',
        }
        
        if c in escapes:
            return escapes[c]
        elif c == 'x':
            # Hex escape \xNN
            hex_chars = self.advance() + self.advance()
            return chr(int(hex_chars, 16))
        elif c == 'u':
            # Unicode escape \uNNNN or \u{NNNNNN}
            if self.peek_char() == '{':
                self.advance()
                hex_chars = []
                while self.peek_char() != '}':
                    hex_chars.append(self.advance())
                self.advance()  # }
                return chr(int(''.join(hex_chars), 16))
            else:
                hex_chars = ''.join(self.advance() for _ in range(4))
                return chr(int(hex_chars, 16))
        else:
            raise self.error(f"Unknown escape sequence: \\{c}")
    
    def scan_char(self) -> Token:
        """Parse character literal 'c'."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        self.advance()  # Opening '
        
        if self.peek_char() == '\\':
            value = self.scan_escape()
        elif self.peek_char() == "'":
            raise self.error("Empty character literal")
        else:
            value = self.advance()
        
        if self.peek_char() != "'":
            raise self.error("Character literal must contain exactly one character")
        
        self.advance()  # Closing '
        
        raw = self.source[start_pos:self.state.pos]
        return self.make_token(TokenType.CHAR, value, start_loc, raw)
    
    # === IDENTIFIER/KEYWORD PARSING ===
    
    def scan_identifier(self) -> Token:
        """Parse identifier or keyword."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        while self.peek_char().isalnum() or self.peek_char() == '_':
            self.advance()
        
        raw = self.source[start_pos:self.state.pos]
        
        # Check if it's a keyword
        if raw in KEYWORDS:
            return self.make_token(KEYWORDS[raw], raw, start_loc, raw)
        
        # Check if it looks like a type (starts with uppercase)
        if raw[0].isupper():
            return self.make_token(TokenType.TYPE_IDENTIFIER, raw, start_loc, raw)
        
        return self.make_token(TokenType.IDENTIFIER, raw, start_loc, raw)
    
    # === COMMENT PARSING ===
    
    def scan_comment(self) -> Optional[Token]:
        """Parse comment (returns None to skip, or Token for doc comments)."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        if self.peek_char(1) == '/':
            # Single-line comment //
            is_doc = self.peek_char(2) == '/'
            self.skip_to_eol()
            
            if is_doc:
                raw = self.source[start_pos:self.state.pos]
                return self.make_token(TokenType.DOC_COMMENT, raw[3:].strip(), start_loc, raw)
            return None
        
        elif self.peek_char(1) == '*':
            # Multi-line comment /* */
            is_doc = self.peek_char(2) == '*' and self.peek_char(3) != '/'
            self.advance()  # /
            self.advance()  # *
            
            depth = 1
            while depth > 0:
                if self.peek_char() == '\0':
                    raise self.error("Unterminated block comment")
                if self.peek_char() == '/' and self.peek_char(1) == '*':
                    depth += 1
                    self.advance()
                elif self.peek_char() == '*' and self.peek_char(1) == '/':
                    depth -= 1
                    self.advance()
                self.advance()
            
            if is_doc:
                raw = self.source[start_pos:self.state.pos]
                # Strip /** and */ and clean up
                content = raw[3:-2].strip()
                return self.make_token(TokenType.DOC_COMMENT, content, start_loc, raw)
            return None
        
        # Just a slash
        self.advance()
        return self.make_token(TokenType.SLASH, '/', start_loc, '/')
    
    def scan_hash_comment(self) -> Optional[Token]:
        """Parse # style comment."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        # Check for annotation #[
        if self.peek_char(1) == '[':
            return self.scan_annotation()
        
        # Regular comment
        self.skip_to_eol()
        return None
    
    def scan_annotation(self) -> Token:
        """Parse #[annotation]."""
        start_loc = self.current_location()
        start_pos = self.state.pos
        
        self.advance()  # #
        self.advance()  # [
        
        depth = 1
        while depth > 0:
            c = self.peek_char()
            if c == '\0':
                raise self.error("Unterminated annotation")
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
            self.advance()
        
        raw = self.source[start_pos:self.state.pos]
        # Extract content between #[ and ]
        content = raw[2:-1]
        return self.make_token(TokenType.ANNOTATION, content, start_loc, raw)
    
    # === OPERATOR SCANNING ===
    
    def scan_operator(self) -> Token:
        """Parse operators and punctuation."""
        start_loc = self.current_location()
        c = self.peek_char()
        
        # Multi-character operators (order matters - longest first!)
        multi_ops = [
            # 3-char
            ('..=', TokenType.RANGE_INCLUSIVE),
            ('...', TokenType.ELLIPSIS),
            ('<=>',TokenType.BIND),
            
            # 2-char
            ('~>', TokenType.MORPH),
            ('<~', TokenType.REVERSE_FLOW),
            ('->', TokenType.FLOW),
            ('=>', TokenType.ARROW),
            ('|>', TokenType.PARALLEL_PIPE),
            ('@>', TokenType.INJECT),
            ('::', TokenType.TRAIT_IMPL),
            ('==', TokenType.EQ),
            ('!=', TokenType.NE),
            ('<=', TokenType.LE),
            ('>=', TokenType.GE),
            ('&&', TokenType.LOGICAL_AND),
            ('||', TokenType.LOGICAL_OR),
            ('<<', TokenType.LSHIFT),
            ('>>', TokenType.RSHIFT),
            ('**', TokenType.POWER),
            ('//', TokenType.FLOOR_DIV),
            ('..', TokenType.RANGE),
            ('+=', TokenType.PLUS_ASSIGN),
            ('-=', TokenType.MINUS_ASSIGN),
            ('*=', TokenType.STAR_ASSIGN),
            ('/=', TokenType.SLASH_ASSIGN),
            ('%=', TokenType.PERCENT_ASSIGN),
        ]
        
        for op, token_type in multi_ops:
            if self.source[self.state.pos:self.state.pos + len(op)] == op:
                for _ in op:
                    self.advance()
                return self.make_token(token_type, op, start_loc, op)
        
        # Single-character operators
        single_ops = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '%': TokenType.PERCENT,
            '<': TokenType.LT,
            '>': TokenType.GT,
            '=': TokenType.ASSIGN,
            '!': TokenType.LOGICAL_NOT,
            '&': TokenType.BIT_AND,
            '|': TokenType.BIT_OR,
            '^': TokenType.BIT_XOR,
            '~': TokenType.BIT_NOT,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ',': TokenType.COMMA,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON,
            '.': TokenType.DOT,
            '?': TokenType.QUESTION,
            '@': TokenType.AT,
            '$': TokenType.DOLLAR,
            '_': TokenType.UNDERSCORE,
            '\\': TokenType.BACKSLASH,
        }
        
        if c in single_ops:
            self.advance()
            return self.make_token(single_ops[c], c, start_loc, c)
        
        raise self.error(f"Unexpected character: {c!r}")
    
    # === MAIN TOKENIZATION ===
    
    def next_token(self) -> Token:
        """Get the next token from the source."""
        # Skip whitespace (but not newlines in some modes)
        leading_ws = self.skip_whitespace()
        
        if self.state.pos >= len(self.source):
            return self.make_token(TokenType.EOF, None, self.current_location(), '')
        
        c = self.peek_char()
        
        # Newline (significant in some contexts)
        if c == '\n':
            loc = self.current_location()
            self.advance()
            return self.make_token(TokenType.NEWLINE, '\n', loc, '\n')
        
        # Numbers
        if c.isdigit():
            return self.scan_number()
        
        # Handle .5 style floats
        if c == '.' and self.peek_char(1).isdigit():
            return self.scan_number()
        
        # Strings
        if c in '"\'':
            # Check if it could be a char literal
            if c == "'" and self.peek_char(2) == "'":
                # Probably a char 'x'
                if self.peek_char(1) != '\\' or self.peek_char(3) == "'":
                    return self.scan_char()
            return self.scan_string(c)
        
        # Template literals
        if c == '`':
            return self.scan_string('`')
        
        # Identifiers and keywords
        if c.isalpha() or c == '_':
            return self.scan_identifier()
        
        # Comments
        if c == '/':
            result = self.scan_comment()
            if result is None:
                return self.next_token()  # Skip comment
            return result
        
        if c == '#':
            result = self.scan_hash_comment()
            if result is None:
                return self.next_token()  # Skip comment
            return result
        
        # Operators and punctuation
        return self.scan_operator()
    
    def tokenize(self) -> Iterator[Token]:
        """Generate all tokens from the source."""
        while True:
            token = self.next_token()
            yield token
            if token.type == TokenType.EOF:
                break
    
    def peek(self, offset: int = 0) -> Token:
        """Look ahead at tokens without consuming them."""
        while len(self._peeked) <= offset:
            self._peeked.append(self.next_token())
        return self._peeked[offset]
    
    def consume(self) -> Token:
        """Consume and return the next token."""
        if self._peeked:
            return self._peeked.pop(0)
        return self.next_token()
    
    def expect(self, token_type: TokenType) -> Token:
        """Consume a token of the expected type, or raise error."""
        token = self.consume()
        if token.type != token_type:
            raise self.error(
                f"Expected {token_type.name}, got {token.type.name}"
            )
        return token
    
    def match(self, *types: TokenType) -> Optional[Token]:
        """If next token matches any type, consume and return it."""
        if self.peek().type in types:
            return self.consume()
        return None


def tokenize(source: str, filename: str = "<input>") -> list[Token]:
    """Convenience function to tokenize a string into a list of tokens."""
    lexer = Lexer(source, filename)
    return list(lexer.tokenize())


def tokenize_file(filepath: str) -> list[Token]:
    """Tokenize a .saaam file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    return tokenize(source, filepath)
