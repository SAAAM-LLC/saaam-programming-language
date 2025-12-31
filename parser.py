"""
SAAAM Language - Parser
The Neural Network: Transforms token streams into living AST structures.

This is where syntax becomes MEANING.
Pratt parsing meets recursive descent meets PURE INNOVATION.
"""

from typing import Optional, Callable, Any, List, Union
from dataclasses import dataclass
from io import StringIO

try:
    from .tokens import Token, TokenType, PRECEDENCE, RIGHT_ASSOCIATIVE, get_precedence
    from .lexer import Lexer, LexerError
    from .ast_nodes import *
except ImportError:
    from tokens import Token, TokenType, PRECEDENCE, RIGHT_ASSOCIATIVE, get_precedence
    from lexer import Lexer, LexerError
    from ast_nodes import *


class ParseError(Exception):
    """When the neural pathway can't be formed."""
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"{token.location}: {message}")


class ParseErrorCollection(Exception):
    """Collection of parse errors for batch reporting."""
    def __init__(self, errors: List[ParseError]):
        self.errors = errors
        messages = [str(e) for e in errors]
        super().__init__(f"Found {len(errors)} parse errors:\n" + "\n".join(messages))


class Parser:
    """
    The SAAAM Parser - Neural Pattern Recognizer
    
    Uses a hybrid approach:
    - Pratt parsing for expressions (handles precedence elegantly)
    - Recursive descent for statements and declarations
    - Contextual parsing for JSX, templates, and components
    """
    
    def __init__(self, source: str, filename: str = "<input>"):
        self.lexer = Lexer(source, filename)
        self.current: Optional[Token] = None
        self.previous: Optional[Token] = None
        self.errors: List[ParseError] = []
        self.panic_mode: bool = False
        self._advance()
        
        # Prefix parsers (for literals, identifiers, unary ops, etc.)
        self.prefix_parsers: dict[TokenType, Callable[[], Expression]] = {
            TokenType.INTEGER: self.parse_integer,
            TokenType.FLOAT: self.parse_float,
            TokenType.STRING: self.parse_string,
            TokenType.CHAR: self.parse_char,
            TokenType.TRUE: self.parse_bool,
            TokenType.FALSE: self.parse_bool,
            TokenType.NONE: self.parse_none,
            TokenType.IDENTIFIER: self.parse_identifier,
            TokenType.TYPE_IDENTIFIER: self.parse_identifier,
            TokenType.LPAREN: self.parse_grouped_or_tuple,
            TokenType.LBRACKET: self.parse_array_or_comprehension,
            TokenType.LBRACE: self.parse_block_or_map,
            TokenType.MINUS: self.parse_unary,
            TokenType.LOGICAL_NOT: self.parse_unary,
            TokenType.BIT_NOT: self.parse_unary,
            TokenType.BIT_AND: self.parse_reference,
            TokenType.STAR: self.parse_deref,
            TokenType.IF: self.parse_if_expr,
            TokenType.MATCH: self.parse_match_expr,
            TokenType.FN: self.parse_lambda,
            TokenType.ASYNC: self.parse_async_lambda,
            TokenType.SOME: self.parse_some,
            TokenType.OK: self.parse_ok,
            TokenType.ERR: self.parse_err,
            TokenType.LT: self.parse_jsx,
            TokenType.SELF: self.parse_self,
            TokenType.SELF_TYPE: self.parse_self_type,
        }
        
        # Infix parsers (for binary ops)
        self.infix_parsers: dict[TokenType, Callable[[Expression], Expression]] = {
            TokenType.PLUS: self.parse_binary,
            TokenType.MINUS: self.parse_binary,
            TokenType.STAR: self.parse_binary,
            TokenType.SLASH: self.parse_binary,
            TokenType.PERCENT: self.parse_binary,
            TokenType.POWER: self.parse_binary,
            TokenType.FLOOR_DIV: self.parse_binary,
            TokenType.EQ: self.parse_binary,
            TokenType.NE: self.parse_binary,
            TokenType.LT: self.parse_binary,
            TokenType.LE: self.parse_binary,
            TokenType.GT: self.parse_binary,
            TokenType.GE: self.parse_binary,
            TokenType.AND: self.parse_binary,
            TokenType.OR: self.parse_binary,
            TokenType.LOGICAL_AND: self.parse_binary,
            TokenType.LOGICAL_OR: self.parse_binary,
            TokenType.BIT_AND: self.parse_binary,
            TokenType.BIT_OR: self.parse_binary,
            TokenType.BIT_XOR: self.parse_binary,
            TokenType.LSHIFT: self.parse_binary,
            TokenType.RSHIFT: self.parse_binary,
            TokenType.MORPH: self.parse_binary,
            TokenType.BIND: self.parse_binary,
            TokenType.FLOW: self.parse_flow,
            TokenType.PARALLEL_PIPE: self.parse_binary,
            TokenType.REVERSE_FLOW: self.parse_await,
            TokenType.INJECT: self.parse_binary,
            TokenType.RANGE: self.parse_range,
            TokenType.RANGE_INCLUSIVE: self.parse_range,
            TokenType.DOT: self.parse_member,
            TokenType.TRAIT_IMPL: self.parse_trait_member,
            TokenType.LPAREN: self.parse_call,
            TokenType.LBRACKET: self.parse_index,
            TokenType.QUESTION: self.parse_try_expr,
            TokenType.AS: self.parse_cast,
            TokenType.ASSIGN: self.parse_assign,
            TokenType.PLUS_ASSIGN: self.parse_compound_assign,
            TokenType.MINUS_ASSIGN: self.parse_compound_assign,
            TokenType.STAR_ASSIGN: self.parse_compound_assign,
            TokenType.SLASH_ASSIGN: self.parse_compound_assign,
            TokenType.PERCENT_ASSIGN: self.parse_compound_assign,
            TokenType.IS: self.parse_is_expr,
        }
    
    # === UTILITY METHODS ===
    
    def _advance(self) -> Token:
        """Advance to the next non-newline token."""
        self.previous = self.current
        while True:
            self.current = self.lexer.consume()
            if self.current.type != TokenType.NEWLINE:
                break
        return self.previous
    
    def _check(self, *types: TokenType) -> bool:
        """Check if current token is one of the given types."""
        return self.current.type in types
    
    def _match(self, *types: TokenType) -> Optional[Token]:
        """If current token matches, consume and return it."""
        if self._check(*types):
            return self._advance()
        return None
    
    def _expect(self, *token_types: TokenType, message: str = None) -> Token:
        """
        Expect one of the given token types, raise error if not found.
        
        FIXED: Now properly accepts multiple token types!
        """
        if not self._check(*token_types):
            if len(token_types) == 1:
                expected = token_types[0].name
            else:
                expected = " or ".join(t.name for t in token_types)
            msg = message or f"Expected {expected}, got {self.current.type.name}"
            raise ParseError(msg, self.current)
        return self._advance()
    
    def _error(self, message: str) -> ParseError:
        """Create a parse error at current position."""
        return ParseError(message, self.current)
    
    def _report_error(self, error: ParseError):
        """Report an error without immediately raising it."""
        if not self.panic_mode:
            self.errors.append(error)
            self.panic_mode = True
    
    def _sync(self):
        """Synchronize after error to continue parsing."""
        self.panic_mode = False
        self._advance()
        while not self._check(TokenType.EOF):
            if self.previous.type == TokenType.SEMICOLON:
                return
            if self._check(TokenType.FN, TokenType.LET, TokenType.VAR,
                          TokenType.STRUCT, TokenType.ENUM, TokenType.TRAIT,
                          TokenType.IMPL, TokenType.IF, TokenType.FOR,
                          TokenType.WHILE, TokenType.RETURN, TokenType.COMPONENT,
                          TokenType.MODULE, TokenType.USE, TokenType.PUB):
                return
            self._advance()
    
    def _skip_newlines(self):
        """Skip any newline tokens."""
        while self._check(TokenType.NEWLINE):
            self._advance()
    
    def _peek_is(self, *types: TokenType) -> bool:
        """Check if the NEXT token (after current) is one of the given types."""
        return self.lexer.peek().type in types
    
    # === EXPRESSION PARSING (Pratt Parser) ===
    
    def parse_expression(self, min_precedence: int = 0) -> Expression:
        """Parse an expression using Pratt parsing."""
        # Get prefix parser
        prefix_parser = self.prefix_parsers.get(self.current.type)
        if prefix_parser is None:
            raise self._error(f"Unexpected token: {self.current.type.name}")
        
        left = prefix_parser()
        
        # Parse infix operations
        while True:
            token_type = self.current.type
            prec = get_precedence(token_type)
            
            if prec < min_precedence:
                break
            
            infix_parser = self.infix_parsers.get(token_type)
            if infix_parser is None:
                break
            
            # Handle right-associativity
            if token_type in RIGHT_ASSOCIATIVE:
                left = infix_parser(left)
            else:
                left = infix_parser(left)
        
        return left
    
    # === PREFIX PARSERS ===
    
    def parse_integer(self) -> IntegerLiteral:
        """Parse integer literal."""
        token = self._advance()
        return IntegerLiteral(
            value=token.value,
            location=token.location
        )
    
    def parse_float(self) -> FloatLiteral:
        """Parse float literal."""
        token = self._advance()
        return FloatLiteral(
            value=token.value,
            location=token.location
        )
    
    def parse_string(self) -> StringLiteral:
        """Parse string literal."""
        token = self._advance()
        return StringLiteral(
            value=token.value,
            is_template=token.raw.startswith('`'),
            location=token.location
        )
    
    def parse_char(self) -> CharLiteral:
        """Parse character literal."""
        token = self._advance()
        return CharLiteral(
            value=token.value,
            location=token.location
        )
    
    def parse_bool(self) -> BoolLiteral:
        """Parse boolean literal."""
        token = self._advance()
        return BoolLiteral(
            value=(token.type == TokenType.TRUE),
            location=token.location
        )
    
    def parse_none(self) -> NoneLiteral:
        """Parse None literal."""
        token = self._advance()
        return NoneLiteral(location=token.location)
    
    def parse_self(self) -> Identifier:
        """Parse self keyword."""
        token = self._advance()
        return Identifier(name="self", location=token.location)
    
    def parse_self_type(self) -> Identifier:
        """Parse Self type keyword."""
        token = self._advance()
        return Identifier(name="Self", location=token.location)
    
    def parse_identifier(self) -> Expression:
        """Parse identifier or qualified name."""
        token = self._advance()
        name = Identifier(name=token.value, location=token.location)
        
        # Check for qualified name (but not after a call/index)
        if self._check(TokenType.TRAIT_IMPL) and not self._peek_is(TokenType.LPAREN, TokenType.LBRACKET):
            parts = [token.value]
            while self._match(TokenType.TRAIT_IMPL):
                next_tok = self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER)
                parts.append(next_tok.value)
            return QualifiedName(parts=parts, location=token.location)
        
        return name
    
    def parse_grouped_or_tuple(self) -> Expression:
        """Parse (expr) or (a, b, c)."""
        start = self._advance()  # (
        
        if self._check(TokenType.RPAREN):
            # Empty tuple ()
            self._advance()
            return TupleLiteral(elements=[], location=start.location)
        
        first = self.parse_expression()
        
        if self._check(TokenType.COMMA):
            # Tuple
            elements = [first]
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RPAREN):
                    break
                elements.append(self.parse_expression())
            self._expect(TokenType.RPAREN)
            return TupleLiteral(elements=elements, location=start.location)
        
        self._expect(TokenType.RPAREN)
        return first  # Grouped expression
    
    def parse_array_or_comprehension(self) -> Expression:
        """Parse [1, 2, 3] or [x for x in items]."""
        start = self._advance()  # [
        
        if self._check(TokenType.RBRACKET):
            self._advance()
            return ArrayLiteral(elements=[], location=start.location)
        
        first = self.parse_expression()
        
        # Check for comprehension
        if self._match(TokenType.FOR):
            var_tok = self._expect(TokenType.IDENTIFIER)
            self._expect(TokenType.IN)
            iterable = self.parse_expression()
            
            condition = None
            if self._match(TokenType.IF):
                condition = self.parse_expression()
            
            self._expect(TokenType.RBRACKET)
            return Comprehension(
                element=first,
                variable=var_tok.value,
                iterable=iterable,
                condition=condition,
                location=start.location
            )
        
        # Regular array
        elements = [first]
        while self._match(TokenType.COMMA):
            if self._check(TokenType.RBRACKET):
                break
            elements.append(self.parse_expression())
        
        self._expect(TokenType.RBRACKET)
        return ArrayLiteral(elements=elements, location=start.location)
    
    def parse_block_or_map(self) -> Expression:
        """Parse { stmts } or { key: value }."""
        start = self._advance()  # {
        
        if self._check(TokenType.RBRACE):
            # Empty block
            self._advance()
            return Block(statements=[], location=start.location)
        
        # Try to detect if it's a map literal
        # Maps start with key: value
        if (self._check(TokenType.IDENTIFIER, TokenType.STRING, TokenType.INTEGER) and
            self.lexer.peek().type == TokenType.COLON):
            return self.parse_map_literal(start)
        
        # Parse as block
        return self.parse_block_body(start)
    
    def parse_map_literal(self, start: Token) -> MapLiteral:
        """Parse { key: value, ... }."""
        pairs = []
        
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            key = self.parse_expression()
            self._expect(TokenType.COLON)
            value = self.parse_expression()
            pairs.append((key, value))
            
            if not self._match(TokenType.COMMA):
                break
        
        self._expect(TokenType.RBRACE)
        return MapLiteral(pairs=pairs, location=start.location)
    
    def parse_block_body(self, start: Token) -> Block:
        """Parse the body of a block."""
        statements = []
        final_expr = None
        
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            stmt = self.parse_statement()
            
            # Check if this could be the final expression
            if isinstance(stmt, ExprStmt) and self._check(TokenType.RBRACE):
                final_expr = stmt.expr
            else:
                statements.append(stmt)
        
        self._expect(TokenType.RBRACE)
        return Block(
            statements=statements,
            final_expr=final_expr,
            location=start.location
        )
    
    def parse_unary(self) -> UnaryOp:
        """Parse unary expression: -x, !y."""
        op = self._advance()
        operand = self.parse_expression(get_precedence(TokenType.LOGICAL_NOT))
        return UnaryOp(
            operator=op.raw,
            operand=operand,
            location=op.location
        )
    
    def parse_reference(self) -> UnaryOp:
        """Parse reference: &x or &mut x."""
        op = self._advance()  # &
        is_mutable = self._match(TokenType.VAR) is not None  # &mut
        operand = self.parse_expression(get_precedence(TokenType.LOGICAL_NOT))
        return UnaryOp(
            operator='&mut' if is_mutable else '&',
            operand=operand,
            location=op.location
        )
    
    def parse_deref(self) -> UnaryOp:
        """Parse dereference: *ptr."""
        op = self._advance()
        operand = self.parse_expression(get_precedence(TokenType.LOGICAL_NOT))
        return UnaryOp(
            operator='*',
            operand=operand,
            location=op.location
        )
    
    def parse_if_expr(self) -> IfExpr:
        """Parse if expression."""
        start = self._advance()  # if
        condition = self.parse_expression()
        self._expect(TokenType.LBRACE)
        then_branch = self.parse_block_body(self.previous)
        
        else_branch = None
        if self._match(TokenType.ELSE):
            if self._check(TokenType.IF):
                else_branch = self.parse_if_expr()
            else:
                self._expect(TokenType.LBRACE)
                else_branch = self.parse_block_body(self.previous)
        
        return IfExpr(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
            location=start.location
        )
    
    def parse_match_expr(self) -> MatchExpr:
        """Parse match expression."""
        start = self._advance()  # match
        subject = self.parse_expression()
        self._expect(TokenType.LBRACE)
        
        arms = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            pattern = self.parse_pattern()
            
            guard = None
            if self._match(TokenType.IF):
                guard = self.parse_expression()
            
            self._expect(TokenType.ARROW)
            body = self.parse_expression()
            
            arms.append(MatchArm(pattern=pattern, guard=guard, body=body))
            
            # Comma or newline between arms
            self._match(TokenType.COMMA)
        
        self._expect(TokenType.RBRACE)
        return MatchExpr(
            subject=subject,
            arms=arms,
            location=start.location
        )
    
    def parse_lambda(self) -> Lambda:
        """Parse lambda: fn(x) => x * 2 or fn(x) { body }."""
        start = self._advance()  # fn
        
        params = self.parse_parameter_list()
        
        # Return type annotation (optional)
        return_type = None
        if self._match(TokenType.FLOW):
            # Check if it's a return type or expression body
            if self._check(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER, 
                          TokenType.LPAREN, TokenType.LBRACKET, TokenType.FN):
                return_type = self.parse_type()
        
        # Body can be => expr or { block }
        if self._match(TokenType.ARROW):
            body = self.parse_expression()
        elif self._check(TokenType.LBRACE):
            self._advance()
            body = self.parse_block_body(self.previous)
        else:
            # Just => expr syntax
            self._expect(TokenType.ARROW)
            body = self.parse_expression()
        
        return Lambda(
            params=params,
            body=body,
            location=start.location
        )
    
    def parse_async_lambda(self) -> Lambda:
        """Parse async lambda."""
        start = self._advance()  # async
        self._expect(TokenType.FN)
        
        params = self.parse_parameter_list()
        
        if self._match(TokenType.ARROW):
            body = self.parse_expression()
        elif self._check(TokenType.LBRACE):
            self._advance()
            body = self.parse_block_body(self.previous)
        else:
            self._expect(TokenType.ARROW)
            body = self.parse_expression()
        
        return Lambda(
            params=params,
            body=body,
            is_async=True,
            location=start.location
        )
    
    def parse_some(self) -> ConstructExpr:
        """Parse Some(value)."""
        start = self._advance()  # Some
        self._expect(TokenType.LPAREN)
        value = self.parse_expression()
        self._expect(TokenType.RPAREN)
        
        return ConstructExpr(
            type_name=Identifier(name="Some", location=start.location),
            fields=[("0", value)],
            location=start.location
        )
    
    def parse_ok(self) -> ConstructExpr:
        """Parse Ok(value)."""
        start = self._advance()
        self._expect(TokenType.LPAREN)
        value = self.parse_expression()
        self._expect(TokenType.RPAREN)
        
        return ConstructExpr(
            type_name=Identifier(name="Ok", location=start.location),
            fields=[("0", value)],
            location=start.location
        )
    
    def parse_err(self) -> ConstructExpr:
        """Parse Err(error)."""
        start = self._advance()
        self._expect(TokenType.LPAREN)
        value = self.parse_expression()
        self._expect(TokenType.RPAREN)
        
        return ConstructExpr(
            type_name=Identifier(name="Err", location=start.location),
            fields=[("0", value)],
            location=start.location
        )
    
    def parse_jsx(self) -> JSXElement:
        """Parse JSX element: <tag>...</tag>."""
        start = self._advance()  # <
        
        tag = self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value
        
        # Parse attributes
        attributes = []
        while not self._check(TokenType.GT, TokenType.SLASH, TokenType.EOF):
            # Allow keywords as attribute names (e.g., on, class, for)
            if self._check(TokenType.IDENTIFIER):
                attr_name = self._advance().value
            elif self.current.type.name in ('ON', 'CLASS', 'FOR', 'TYPE', 'IN', 'IF', 'ELSE'):
                # Allow certain keywords as attribute names in JSX
                attr_name = self._advance().raw
            else:
                attr_name = self._expect(TokenType.IDENTIFIER).value
            
            attr_value = None
            if self._match(TokenType.ASSIGN):
                if self._match(TokenType.LBRACE):
                    attr_value = self.parse_expression()
                    self._expect(TokenType.RBRACE)
                else:
                    attr_value = self.parse_expression()
            else:
                attr_value = BoolLiteral(value=True, location=self.current.location)
            
            attributes.append((attr_name, attr_value))
        
        # Self-closing tag
        if self._match(TokenType.SLASH):
            self._expect(TokenType.GT)
            return JSXElement(
                tag=tag,
                attributes=attributes,
                is_self_closing=True,
                location=start.location
            )
        
        self._expect(TokenType.GT)
        
        # Parse children
        children = []
        while not (self._check(TokenType.LT) and self.lexer.peek().type == TokenType.SLASH):
            if self._check(TokenType.LBRACE):
                self._advance()
                children.append(self.parse_expression())
                self._expect(TokenType.RBRACE)
            elif self._check(TokenType.LT):
                children.append(self.parse_jsx())
            elif self._check(TokenType.STRING):
                children.append(self.parse_string())
            elif self._check(TokenType.EOF):
                raise self._error("Unterminated JSX element")
            elif self._check(TokenType.RBRACE, TokenType.SEMICOLON):
                # These shouldn't appear in JSX content
                break
            else:
                # JSX text content - collect a single token as text
                # This handles identifiers, operators like +/-, etc.
                text = self.current.raw
                text_loc = self.current.location
                self._advance()
                children.append(StringLiteral(
                    value=text.strip(),
                    is_template=False,
                    location=text_loc
                ))
        
        # Closing tag
        self._expect(TokenType.LT)
        self._expect(TokenType.SLASH)
        closing_tag = self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value
        if closing_tag != tag:
            raise self._error(f"Mismatched JSX tags: <{tag}> and </{closing_tag}>")
        self._expect(TokenType.GT)
        
        return JSXElement(
            tag=tag,
            attributes=attributes,
            children=children,
            location=start.location
        )
    
    # === INFIX PARSERS ===
    
    def parse_binary(self, left: Expression) -> BinaryOp:
        """Parse binary operation."""
        op = self._advance()
        prec = get_precedence(op.type)
        
        if op.type in RIGHT_ASSOCIATIVE:
            right = self.parse_expression(prec)
        else:
            right = self.parse_expression(prec + 1)
        
        return BinaryOp(
            left=left,
            operator=op.raw,
            right=right,
            location=op.location
        )
    
    def parse_flow(self, left: Expression) -> Expression:
        """Parse flow operator: data -> transform."""
        op = self._advance()  # ->
        
        # Could be a pipeline or return type annotation in different contexts
        right = self.parse_expression(get_precedence(TokenType.FLOW) + 1)
        
        # If right is a call, it's a pipeline
        if isinstance(right, Call):
            # Insert left as first argument
            right.arguments.insert(0, left)
            return right
        
        return BinaryOp(
            left=left,
            operator='->',
            right=right,
            location=op.location
        )
    
    def parse_await(self, left: Expression) -> AwaitExpr:
        """Parse reverse flow (await): result <~ async_call()."""
        self._advance()  # <~
        return AwaitExpr(expr=left, location=left.location)
    
    def parse_range(self, left: Expression) -> RangeExpr:
        """Parse range expression."""
        op = self._advance()
        inclusive = op.type == TokenType.RANGE_INCLUSIVE
        
        end = None
        if not self._check(TokenType.RBRACKET, TokenType.RPAREN, TokenType.COMMA,
                          TokenType.LBRACE, TokenType.SEMICOLON):
            end = self.parse_expression(get_precedence(TokenType.RANGE) + 1)
        
        return RangeExpr(
            start=left,
            end=end,
            inclusive=inclusive,
            location=op.location
        )
    
    def parse_member(self, left: Expression) -> Expression:
        """Parse member access: obj.field or obj.0 (tuple)."""
        self._advance()  # .
        
        # Handle tuple index (obj.0, obj.1, etc.)
        if self._check(TokenType.INTEGER):
            index_token = self._advance()
            return Index(
                object=left,
                index=IntegerLiteral(value=index_token.value, location=index_token.location),
                location=left.location
            )
        
        member = self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value
        return Member(
            object=left,
            member=member,
            location=left.location
        )
    
    def parse_trait_member(self, left: Expression) -> TraitMember:
        """Parse trait member: Type::method."""
        self._advance()  # ::
        member = self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value
        return TraitMember(
            type_expr=left,
            member=member,
            location=left.location
        )
    
    def parse_call(self, left: Expression) -> Call:
        """Parse function call: func(args) or func<T>(args)."""
        # Check for type arguments func<T, U>(...)
        type_args = []
        
        self._advance()  # (
        
        arguments = []
        if not self._check(TokenType.RPAREN):
            # Handle named arguments: foo(name: value)
            arguments.append(self._parse_call_argument())
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RPAREN):
                    break
                arguments.append(self._parse_call_argument())
        
        self._expect(TokenType.RPAREN)
        return Call(
            callee=left,
            arguments=arguments,
            type_args=type_args,
            location=left.location
        )
    
    def _parse_call_argument(self) -> Expression:
        """Parse a single call argument, potentially named."""
        # For now, just parse as expression
        # TODO: Handle named arguments like foo(name: value)
        return self.parse_expression()
    
    def parse_index(self, left: Expression) -> Index:
        """Parse index access: arr[i]."""
        self._advance()  # [
        index = self.parse_expression()
        self._expect(TokenType.RBRACKET)
        return Index(
            object=left,
            index=index,
            location=left.location
        )
    
    def parse_try_expr(self, left: Expression) -> TryExpr:
        """Parse try expression: expr?"""
        self._advance()  # ?
        return TryExpr(expr=left, location=left.location)
    
    def parse_cast(self, left: Expression) -> CastExpr:
        """Parse cast: expr as Type."""
        self._advance()  # as
        target_type = self.parse_type()
        return CastExpr(
            expr=left,
            target_type=target_type,
            location=left.location
        )
    
    def parse_is_expr(self, left: Expression) -> BinaryOp:
        """Parse is expression: expr is Type."""
        op = self._advance()  # is
        target_type = self.parse_type()
        return BinaryOp(
            left=left,
            operator='is',
            right=Identifier(name=str(target_type), location=op.location),
            location=op.location
        )
    
    def parse_assign(self, left: Expression) -> BinaryOp:
        """Parse assignment: x = value."""
        op = self._advance()
        right = self.parse_expression(get_precedence(TokenType.ASSIGN))
        return BinaryOp(
            left=left,
            operator='=',
            right=right,
            location=op.location
        )
    
    def parse_compound_assign(self, left: Expression) -> BinaryOp:
        """Parse compound assignment: x += 1."""
        op = self._advance()
        right = self.parse_expression(get_precedence(op.type))
        return BinaryOp(
            left=left,
            operator=op.raw,
            right=right,
            location=op.location
        )
    
    # === TYPE PARSING ===
    
    def parse_type(self) -> Type:
        """Parse a type annotation."""
        type_node = self.parse_primary_type()
        
        # Array type shorthand: [T]
        # Already handled in primary
        
        # Check for ? (Option shorthand)
        if self._match(TokenType.QUESTION):
            type_node = OptionType(inner_type=type_node)
        
        return type_node
    
    def parse_primary_type(self) -> Type:
        """Parse primary type."""
        if self._match(TokenType.LPAREN):
            # Tuple type or function type
            types = []
            if not self._check(TokenType.RPAREN):
                types.append(self.parse_type())
                while self._match(TokenType.COMMA):
                    types.append(self.parse_type())
            self._expect(TokenType.RPAREN)
            
            if self._match(TokenType.FLOW):
                # Function type
                return_type = self.parse_type()
                return FunctionType(param_types=types, return_type=return_type)
            
            if len(types) == 1:
                return types[0]  # Grouped type
            return TupleType(element_types=types)
        
        if self._match(TokenType.LBRACKET):
            # Array type: [T] or [T; N]
            elem_type = self.parse_type()
            size = None
            if self._match(TokenType.SEMICOLON):
                size_tok = self._expect(TokenType.INTEGER)
                size = size_tok.value
            self._expect(TokenType.RBRACKET)
            return ArrayType(element_type=elem_type, size=size)
        
        if self._match(TokenType.BIT_AND):
            # Reference type
            is_mutable = self._match(TokenType.VAR) is not None
            inner = self.parse_type()
            return ReferenceType(inner_type=inner, is_mutable=is_mutable)
        
        if self._match(TokenType.FN):
            # Function type: fn(A, B) -> C
            self._expect(TokenType.LPAREN)
            param_types = []
            if not self._check(TokenType.RPAREN):
                param_types.append(self.parse_type())
                while self._match(TokenType.COMMA):
                    param_types.append(self.parse_type())
            self._expect(TokenType.RPAREN)
            
            return_type = None
            if self._match(TokenType.FLOW):
                return_type = self.parse_type()
            
            return FunctionType(param_types=param_types, return_type=return_type)
        
        # Named type
        name_tok = self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER)
        name = name_tok.value
        
        # Check for qualified name
        parts = [name]
        while self._match(TokenType.TRAIT_IMPL):
            next_tok = self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER)
            parts.append(next_tok.value)
        
        # Type arguments
        type_args = []
        if self._match(TokenType.LT):
            type_args.append(self.parse_type())
            while self._match(TokenType.COMMA):
                type_args.append(self.parse_type())
            self._expect(TokenType.GT)
        
        # Check for built-in types
        builtins = {'Int', 'Float', 'String', 'Bool', 'Char', 'Any',
                    'Int8', 'Int16', 'Int32', 'Int64',
                    'UInt', 'UInt8', 'UInt16', 'UInt32', 'UInt64',
                    'Float32', 'Float64', 'Void'}
        
        if len(parts) == 1 and name in builtins and not type_args:
            return PrimitiveType(name=name)
        
        qualified = QualifiedName(parts=parts, location=name_tok.location)
        return NamedType(name=qualified, type_args=type_args)
    
    # === PATTERN PARSING ===
    
    def parse_pattern(self) -> Pattern:
        """Parse a pattern (for match, let, etc.)."""
        if self._match(TokenType.UNDERSCORE):
            return WildcardPattern(location=self.previous.location)
        
        if self._check(TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING, TokenType.CHAR):
            loc = self.current.location
            return LiteralPattern(value=self.parse_expression(), location=loc)
        
        if self._match(TokenType.TRUE):
            return LiteralPattern(
                value=BoolLiteral(value=True, location=self.previous.location),
                location=self.previous.location
            )
        
        if self._match(TokenType.FALSE):
            return LiteralPattern(
                value=BoolLiteral(value=False, location=self.previous.location),
                location=self.previous.location
            )
        
        if self._match(TokenType.NONE):
            return EnumPattern(variant="None", fields=[], location=self.previous.location)
        
        if self._match(TokenType.LPAREN):
            # Tuple pattern
            start = self.previous
            elements = []
            if not self._check(TokenType.RPAREN):
                elements.append(self.parse_pattern())
                while self._match(TokenType.COMMA):
                    if self._check(TokenType.RPAREN):
                        break
                    elements.append(self.parse_pattern())
            self._expect(TokenType.RPAREN)
            return TuplePattern(elements=elements, location=start.location)
        
        if self._match(TokenType.LBRACKET):
            # Array pattern
            start = self.previous
            elements = []
            rest = None
            while not self._check(TokenType.RBRACKET, TokenType.EOF):
                if self._match(TokenType.ELLIPSIS):
                    if self._check(TokenType.IDENTIFIER):
                        rest = self._advance().value
                    else:
                        rest = "_"
                    break
                elements.append(self.parse_pattern())
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.RBRACKET)
            return ArrayPattern(elements=elements, rest=rest, location=start.location)
        
        if self._check(TokenType.IDENTIFIER):
            name_tok = self._advance()
            
            # Check for struct pattern: Name { fields }
            if self._match(TokenType.LBRACE):
                fields = []
                has_rest = False
                while not self._check(TokenType.RBRACE, TokenType.EOF):
                    if self._match(TokenType.ELLIPSIS):
                        has_rest = True
                        break
                    field_name = self._expect(TokenType.IDENTIFIER).value
                    if self._match(TokenType.COLON):
                        field_pattern = self.parse_pattern()
                    else:
                        field_pattern = IdentifierPattern(name=field_name)
                    fields.append((field_name, field_pattern))
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RBRACE)
                return StructPattern(
                    type_name=name_tok.value,
                    fields=fields,
                    has_rest=has_rest,
                    location=name_tok.location
                )
            
            # Check for enum pattern: Variant(fields)
            if self._match(TokenType.LPAREN):
                fields = []
                if not self._check(TokenType.RPAREN):
                    fields.append(self.parse_pattern())
                    while self._match(TokenType.COMMA):
                        if self._check(TokenType.RPAREN):
                            break
                        fields.append(self.parse_pattern())
                self._expect(TokenType.RPAREN)
                return EnumPattern(
                    variant=name_tok.value,
                    fields=fields,
                    location=name_tok.location
                )
            
            # Check for range pattern: 1..10
            if self._check(TokenType.RANGE, TokenType.RANGE_INCLUSIVE):
                is_inclusive = self._match(TokenType.RANGE_INCLUSIVE) is not None
                if not is_inclusive:
                    self._match(TokenType.RANGE)
                end = self.parse_pattern()
                return RangePattern(
                    start=LiteralPattern(value=Identifier(name=name_tok.value, location=name_tok.location), location=name_tok.location),
                    end=end.value if isinstance(end, LiteralPattern) else None,
                    inclusive=is_inclusive,
                    location=name_tok.location
                )
            
            return IdentifierPattern(name=name_tok.value, location=name_tok.location)
        
        if self._check(TokenType.TYPE_IDENTIFIER):
            # Capitalized = enum variant, struct pattern, or type
            name_tok = self._advance()
            
            # Check for struct pattern: Point { x, y }
            if self._match(TokenType.LBRACE):
                fields = []
                has_rest = False
                while not self._check(TokenType.RBRACE, TokenType.EOF):
                    if self._match(TokenType.ELLIPSIS):
                        has_rest = True
                        break
                    field_name = self._expect(TokenType.IDENTIFIER).value
                    if self._match(TokenType.COLON):
                        field_pattern = self.parse_pattern()
                    else:
                        field_pattern = IdentifierPattern(name=field_name)
                    fields.append((field_name, field_pattern))
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RBRACE)
                return StructPattern(
                    type_name=name_tok.value,
                    fields=fields,
                    has_rest=has_rest,
                    location=name_tok.location
                )
            
            # Check for tuple enum pattern: Variant(x, y)
            if self._match(TokenType.LPAREN):
                fields = []
                if not self._check(TokenType.RPAREN):
                    fields.append(self.parse_pattern())
                    while self._match(TokenType.COMMA):
                        fields.append(self.parse_pattern())
                self._expect(TokenType.RPAREN)
                return EnumPattern(variant=name_tok.value, fields=fields, location=name_tok.location)
            
            # Unit enum variant
            return EnumPattern(variant=name_tok.value, fields=[], location=name_tok.location)
        
        # Check for Some/Ok/Err patterns
        if self._match(TokenType.SOME):
            start = self.previous
            self._expect(TokenType.LPAREN)
            inner = self.parse_pattern()
            self._expect(TokenType.RPAREN)
            return EnumPattern(variant="Some", fields=[inner], location=start.location)
        
        if self._match(TokenType.OK):
            start = self.previous
            self._expect(TokenType.LPAREN)
            inner = self.parse_pattern()
            self._expect(TokenType.RPAREN)
            return EnumPattern(variant="Ok", fields=[inner], location=start.location)
        
        if self._match(TokenType.ERR):
            start = self.previous
            self._expect(TokenType.LPAREN)
            inner = self.parse_pattern()
            self._expect(TokenType.RPAREN)
            return EnumPattern(variant="Err", fields=[inner], location=start.location)
        
        raise self._error(f"Expected pattern, got {self.current.type.name}")
    
    # === STATEMENT PARSING ===
    
    def parse_statement(self) -> Statement:
        """Parse a statement."""
        # Skip any doc comments and annotations
        doc_comment = None
        annotations = []
        
        while self._check(TokenType.DOC_COMMENT, TokenType.ANNOTATION):
            if self._match(TokenType.DOC_COMMENT):
                doc_comment = self.previous.value
            elif self._match(TokenType.ANNOTATION):
                annotations.append(self.previous.value)
        
        # Public modifier
        is_public = self._match(TokenType.PUB) is not None
        
        # Declarations
        if self._check(TokenType.LET, TokenType.VAR, TokenType.CONST, 
                      TokenType.NEURAL, TokenType.GC):
            decl = self.parse_var_decl()
            decl.is_public = is_public
            decl.doc_comment = doc_comment
            return decl
        
        if self._check(TokenType.FN):
            decl = self.parse_function_decl(annotations)
            decl.is_public = is_public
            decl.doc_comment = doc_comment
            return decl
        
        if self._check(TokenType.ASYNC) and self._peek_is(TokenType.FN):
            self._advance()  # async
            decl = self.parse_function_decl(annotations)
            decl.is_async = True
            decl.is_public = is_public
            decl.doc_comment = doc_comment
            return decl
        
        if self._check(TokenType.STRUCT):
            decl = self.parse_struct_decl()
            decl.is_public = is_public
            return decl
        
        if self._check(TokenType.ENUM):
            decl = self.parse_enum_decl()
            decl.is_public = is_public
            return decl
        
        if self._check(TokenType.TRAIT):
            decl = self.parse_trait_decl()
            decl.is_public = is_public
            return decl
        
        if self._check(TokenType.IMPL):
            return self.parse_impl_decl()
        
        if self._check(TokenType.TYPE):
            decl = self.parse_type_alias()
            decl.is_public = is_public
            return decl
        
        if self._check(TokenType.COMPONENT):
            decl = self.parse_component_decl()
            decl.is_public = is_public
            return decl
        
        if self._check(TokenType.ACTOR):
            return self.parse_actor_decl()
        
        if self._check(TokenType.CHAN):
            return self.parse_channel_decl()
        
        # Control flow
        if self._check(TokenType.IF):
            return self.parse_if_stmt()
        
        if self._check(TokenType.WHILE):
            return self.parse_while_stmt()
        
        if self._check(TokenType.FOR):
            return self.parse_for_stmt()
        
        if self._check(TokenType.LOOP):
            return self.parse_loop_stmt()
        
        if self._match(TokenType.BREAK):
            value = None
            label = None
            # Check for label: break 'label
            if not self._check(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.NEWLINE, TokenType.EOF):
                value = self.parse_expression()
            return BreakStmt(value=value, label=label, location=self.previous.location)
        
        if self._match(TokenType.CONTINUE):
            label = None
            return ContinueStmt(label=label, location=self.previous.location)
        
        if self._match(TokenType.RETURN):
            value = None
            if not self._check(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.NEWLINE, TokenType.EOF):
                value = self.parse_expression()
            return ReturnStmt(value=value, location=self.previous.location)
        
        if self._match(TokenType.YIELD):
            value = None
            if not self._check(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.NEWLINE, TokenType.EOF):
                value = self.parse_expression()
            return YieldStmt(value=value, location=self.previous.location)
        
        if self._check(TokenType.TRY):
            return self.parse_try_stmt()
        
        if self._check(TokenType.SPAWN):
            return self.parse_spawn_stmt()
        
        if self._check(TokenType.PARALLEL):
            return self.parse_parallel_stmt()
        
        # Expression statement
        expr = self.parse_expression()
        self._match(TokenType.SEMICOLON)
        return ExprStmt(expr=expr, location=expr.location)
    
    def parse_var_decl(self) -> VarDecl:
        """Parse variable declaration."""
        start = self._advance()
        
        is_mutable = start.type == TokenType.VAR
        is_const = start.type == TokenType.CONST
        is_neural = start.type == TokenType.NEURAL
        is_gc = start.type == TokenType.GC
        is_strict = False
        is_heap = False
        
        # Handle modifiers like: gc var x, strict let x, heap let x
        if is_gc and self._match(TokenType.VAR):
            is_mutable = True
        if is_gc and self._match(TokenType.LET):
            is_mutable = False
        
        # Handle strict and heap keywords before let/var
        if start.type == TokenType.LET and self._match(TokenType.STRICT):
            is_strict = True
        
        name = self._expect(TokenType.IDENTIFIER).value
        
        var_type = None
        if self._match(TokenType.COLON):
            var_type = self.parse_type()
        
        initializer = None
        is_await = False
        
        # Handle both = and <~ (async assignment)
        if self._match(TokenType.ASSIGN):
            initializer = self.parse_expression()
        elif self._match(TokenType.REVERSE_FLOW):
            # let x <~ async_call() is sugar for let x = await async_call()
            is_await = True
            initializer = AwaitExpr(
                expr=self.parse_expression(),
                location=self.previous.location
            )
        
        return VarDecl(
            name=name,
            var_type=var_type,
            initializer=initializer,
            is_mutable=is_mutable,
            is_const=is_const,
            is_neural=is_neural,
            is_gc=is_gc,
            is_strict=is_strict,
            is_heap=is_heap,
            location=start.location
        )
    
    def parse_parameter_list(self) -> list[Parameter]:
        """Parse function parameter list."""
        self._expect(TokenType.LPAREN)
        
        params = []
        while not self._check(TokenType.RPAREN, TokenType.EOF):
            is_variadic = self._match(TokenType.ELLIPSIS) is not None
            is_mutable = False
            is_ref = False
            
            # Handle reference parameters: &self, &mut self, &var self
            if self._match(TokenType.BIT_AND):
                is_ref = True
                if self._match(TokenType.VAR):
                    is_mutable = True
            elif self._match(TokenType.VAR):
                is_mutable = True
            
            # Handle self parameter
            if self._check(TokenType.SELF):
                self._advance()
                name = "self"
                param_type = ReferenceType(
                    inner_type=NamedType(
                        name=QualifiedName(parts=["Self"], location=self.previous.location),
                        type_args=[]
                    ),
                    is_mutable=is_mutable
                ) if is_ref else NamedType(
                    name=QualifiedName(parts=["Self"], location=self.previous.location),
                    type_args=[]
                )
            else:
                name = self._expect(TokenType.IDENTIFIER).value
                param_type = None
                if self._match(TokenType.COLON):
                    param_type = self.parse_type()
                # If we got a ref but no explicit type, wrap in reference
                if is_ref and param_type is not None:
                    param_type = ReferenceType(inner_type=param_type, is_mutable=is_mutable)
            
            default_value = None
            if self._match(TokenType.ASSIGN):
                default_value = self.parse_expression()
            
            params.append(Parameter(
                name=name,
                param_type=param_type,
                default_value=default_value,
                is_variadic=is_variadic,
                is_mutable=is_mutable
            ))
            
            if not self._match(TokenType.COMMA):
                break
        
        self._expect(TokenType.RPAREN)
        return params
    
    def parse_function_decl(self, annotations: list[str] = None) -> FunctionDecl:
        """Parse function declaration."""
        start = self._advance()  # fn
        
        is_async = False
        is_const = False
        
        # Check for const fn
        if start.type == TokenType.CONST:
            is_const = True
            self._expect(TokenType.FN)
        
        name = self._expect(TokenType.IDENTIFIER).value
        
        # Type parameters
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        params = self.parse_parameter_list()
        
        return_type = None
        if self._match(TokenType.FLOW):
            return_type = self.parse_type()
        
        # Where clause for bounds
        where_clause = []
        if self._match(TokenType.WHERE):
            while True:
                type_name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
                self._expect(TokenType.COLON)
                bounds = [self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value]
                while self._match(TokenType.PLUS):
                    bounds.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
                where_clause.append((type_name, bounds))
                if not self._match(TokenType.COMMA):
                    break
        
        # Effects
        effects = []
        if self._match(TokenType.WITH):
            effects.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                effects.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
        
        body = None
        if self._match(TokenType.LBRACE):
            body = self.parse_block_body(self.previous)
        
        return FunctionDecl(
            name=name,
            params=params,
            return_type=return_type,
            body=body,
            is_async=is_async,
            is_const=is_const,
            type_params=type_params,
            where_clause=where_clause,
            effects=effects,
            annotations=annotations or [],
            location=start.location
        )
    
    def parse_struct_decl(self) -> StructDecl:
        """Parse struct declaration."""
        start = self._advance()  # struct
        name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        self._expect(TokenType.LBRACE)
        
        fields = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            is_public = self._match(TokenType.PUB) is not None
            field_name = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.COLON)
            field_type = self.parse_type()
            
            default_value = None
            if self._match(TokenType.ASSIGN):
                default_value = self.parse_expression()
            
            fields.append(StructField(
                name=field_name,
                field_type=field_type,
                default_value=default_value,
                is_public=is_public
            ))
            
            self._match(TokenType.COMMA)
        
        self._expect(TokenType.RBRACE)
        
        return StructDecl(
            name=name,
            fields=fields,
            type_params=type_params,
            location=start.location
        )
    
    def parse_enum_decl(self) -> EnumDecl:
        """Parse enum declaration."""
        start = self._advance()  # enum
        name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        self._expect(TokenType.LBRACE)
        
        variants = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            variant_name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
            
            fields = []
            if self._match(TokenType.LPAREN):
                while not self._check(TokenType.RPAREN, TokenType.EOF):
                    field_name = None
                    if self._check(TokenType.IDENTIFIER) and self.lexer.peek().type == TokenType.COLON:
                        field_name = self._advance().value
                        self._advance()  # :
                    field_type = self.parse_type()
                    fields.append((field_name, field_type))
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
            
            value = None
            if self._match(TokenType.ASSIGN):
                value = self.parse_expression()
            
            variants.append(EnumVariant(
                name=variant_name,
                fields=fields,
                value=value
            ))
            
            self._match(TokenType.COMMA)
        
        self._expect(TokenType.RBRACE)
        
        return EnumDecl(
            name=name,
            variants=variants,
            type_params=type_params,
            location=start.location
        )
    
    def parse_trait_decl(self) -> TraitDecl:
        """Parse trait declaration."""
        start = self._advance()  # trait
        name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        super_traits = []
        if self._match(TokenType.COLON):
            super_traits.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.PLUS):
                super_traits.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
        
        self._expect(TokenType.LBRACE)
        
        methods = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            methods.append(self.parse_function_decl())
        
        self._expect(TokenType.RBRACE)
        
        return TraitDecl(
            name=name,
            methods=methods,
            type_params=type_params,
            super_traits=super_traits,
            location=start.location
        )
    
    def parse_impl_decl(self) -> ImplDecl:
        """Parse impl block."""
        start = self._advance()  # impl
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        # Could be "impl Type" or "impl Trait for Type"
        first_type = self.parse_type()
        
        trait_name = None
        target_type = first_type
        
        if self._match(TokenType.FOR):
            if isinstance(first_type, NamedType):
                trait_name = first_type.name.full_name()
            elif isinstance(first_type, PrimitiveType):
                trait_name = first_type.name
            else:
                trait_name = str(first_type)
            target_type = self.parse_type()
        
        self._expect(TokenType.LBRACE)
        
        methods = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            methods.append(self.parse_function_decl())
        
        self._expect(TokenType.RBRACE)
        
        return ImplDecl(
            target_type=target_type,
            trait_name=trait_name,
            methods=methods,
            type_params=type_params,
            location=start.location
        )
    
    def parse_type_alias(self) -> TypeAlias:
        """Parse type alias."""
        start = self._advance()  # type
        name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        self._expect(TokenType.ASSIGN)
        aliased_type = self.parse_type()
        
        return TypeAlias(
            name=name,
            aliased_type=aliased_type,
            type_params=type_params,
            location=start.location
        )
    
    def parse_component_decl(self) -> ComponentDecl:
        """Parse component declaration."""
        start = self._advance()  # component
        name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
        
        self._expect(TokenType.LBRACE)
        
        state = []
        props = []
        methods = []
        lifecycle = {}
        render = None
        
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            if self._match(TokenType.STATE):
                # state count: Int = 0 OR state { ... }
                if self._check(TokenType.LBRACE):
                    self._advance()
                    while not self._check(TokenType.RBRACE, TokenType.EOF):
                        state.append(self._parse_field_decl())
                    self._expect(TokenType.RBRACE)
                else:
                    state.append(self._parse_field_decl())
            elif self._match(TokenType.PROPS):
                self._expect(TokenType.LBRACE)
                while not self._check(TokenType.RBRACE, TokenType.EOF):
                    # Props can have 'let' keyword or just field: Type = value
                    if self._check(TokenType.LET, TokenType.VAR, TokenType.CONST):
                        props.append(self.parse_var_decl())
                    else:
                        props.append(self._parse_field_decl())
                self._expect(TokenType.RBRACE)
            elif self._match(TokenType.ON):
                # Lifecycle hook - name could be a keyword (mount, unmount, update) or identifier
                if self._check(TokenType.MOUNT, TokenType.UNMOUNT, TokenType.UPDATE):
                    hook = self._advance().raw
                else:
                    hook = self._expect(TokenType.IDENTIFIER).value
                self._expect(TokenType.LBRACE)
                lifecycle[hook] = self.parse_block_body(self.previous)
            elif self._match(TokenType.RENDER):
                self._expect(TokenType.LBRACE)
                render = self.parse_jsx()
                self._expect(TokenType.RBRACE)
            elif self._check(TokenType.FN):
                methods.append(self.parse_function_decl())
            else:
                raise self._error(f"Unexpected token in component: {self.current.type.name}")
        
        self._expect(TokenType.RBRACE)
        
        return ComponentDecl(
            name=name,
            state=state,
            props=props,
            methods=methods,
            lifecycle=lifecycle,
            render=render,
            location=start.location
        )
    
    def _parse_field_decl(self) -> VarDecl:
        """Parse a field declaration like: name: Type = value."""
        start_loc = self.current.location
        name = self._expect(TokenType.IDENTIFIER).value
        
        var_type = None
        if self._match(TokenType.COLON):
            var_type = self.parse_type()
        
        initializer = None
        if self._match(TokenType.ASSIGN):
            initializer = self.parse_expression()
        
        return VarDecl(
            name=name,
            var_type=var_type,
            initializer=initializer,
            is_mutable=True,  # Component state is mutable
            location=start_loc
        )
    
    def parse_actor_decl(self) -> ActorDecl:
        """Parse actor declaration."""
        start = self._advance()  # actor
        name = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
        
        self._expect(TokenType.LBRACE)
        
        state = []
        receives = []
        
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            if self._match(TokenType.STATE):
                state.append(self.parse_var_decl())
            elif self._match(TokenType.RECEIVE):
                msg_type = self._expect(TokenType.TYPE_IDENTIFIER, TokenType.IDENTIFIER).value
                self._expect(TokenType.ARROW)
                if self._check(TokenType.LBRACE):
                    self._advance()
                    body = self.parse_block_body(self.previous)
                else:
                    expr = self.parse_expression()
                    body = Block(statements=[], final_expr=expr, location=expr.location)
                receives.append((msg_type, body))
            else:
                raise self._error(f"Unexpected token in actor: {self.current.type.name}")
        
        self._expect(TokenType.RBRACE)
        
        return ActorDecl(
            name=name,
            state=state,
            receives=receives,
            location=start.location
        )
    
    def parse_channel_decl(self) -> ChannelDecl:
        """Parse channel declaration."""
        start = self._advance()  # chan
        name = self._expect(TokenType.IDENTIFIER).value
        
        self._expect(TokenType.COLON)
        element_type = self.parse_type()
        
        capacity = None
        if self._match(TokenType.LPAREN):
            cap_tok = self._expect(TokenType.INTEGER)
            capacity = cap_tok.value
            self._expect(TokenType.RPAREN)
        
        return ChannelDecl(
            name=name,
            element_type=element_type,
            capacity=capacity,
            location=start.location
        )
    
    def parse_if_stmt(self) -> IfStmt:
        """Parse if statement."""
        start = self._advance()  # if
        condition = self.parse_expression()
        self._expect(TokenType.LBRACE)
        then_branch = self.parse_block_body(self.previous)
        
        else_branch = None
        if self._match(TokenType.ELSE):
            if self._check(TokenType.IF):
                else_branch = self.parse_if_stmt()
            else:
                self._expect(TokenType.LBRACE)
                else_branch = self.parse_block_body(self.previous)
        
        return IfStmt(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
            location=start.location
        )
    
    def parse_while_stmt(self) -> WhileStmt:
        """Parse while statement."""
        start = self._advance()  # while
        condition = self.parse_expression()
        self._expect(TokenType.LBRACE)
        body = self.parse_block_body(self.previous)
        
        # Optional else branch (executed if loop never runs)
        else_branch = None
        if self._match(TokenType.ELSE):
            self._expect(TokenType.LBRACE)
            else_branch = self.parse_block_body(self.previous)
        
        return WhileStmt(
            condition=condition,
            body=body,
            else_branch=else_branch,
            location=start.location
        )
    
    def parse_for_stmt(self) -> ForStmt:
        """Parse for statement."""
        start = self._advance()  # for
        
        # Pattern or simple variable
        if self._check(TokenType.IDENTIFIER) and self.lexer.peek().type == TokenType.IN:
            variable = self._advance().value
            pattern = None
        else:
            pattern = self.parse_pattern()
            variable = ""
        
        self._expect(TokenType.IN)
        iterable = self.parse_expression()
        self._expect(TokenType.LBRACE)
        body = self.parse_block_body(self.previous)
        
        return ForStmt(
            variable=variable,
            pattern=pattern,
            iterable=iterable,
            body=body,
            location=start.location
        )
    
    def parse_loop_stmt(self) -> LoopStmt:
        """Parse loop statement."""
        start = self._advance()  # loop
        
        # Optional label: 'label: loop { ... }
        label = None
        
        self._expect(TokenType.LBRACE)
        body = self.parse_block_body(self.previous)
        
        return LoopStmt(body=body, label=label, location=start.location)
    
    def parse_try_stmt(self) -> TryStmt:
        """Parse try statement."""
        start = self._advance()  # try
        self._expect(TokenType.LBRACE)
        body = self.parse_block_body(self.previous)
        
        catches = []
        while self._match(TokenType.CATCH):
            var_name = self._expect(TokenType.IDENTIFIER).value
            
            error_type = None
            if self._match(TokenType.COLON):
                error_type = self.parse_type()
            
            self._expect(TokenType.LBRACE)
            catch_body = self.parse_block_body(self.previous)
            catches.append((var_name, error_type, catch_body))
        
        finally_block = None
        if self._match(TokenType.FINALLY):
            self._expect(TokenType.LBRACE)
            finally_block = self.parse_block_body(self.previous)
        
        return TryStmt(
            body=body,
            catches=catches,
            finally_block=finally_block,
            location=start.location
        )
    
    def parse_spawn_stmt(self) -> SpawnStmt:
        """Parse spawn statement."""
        start = self._advance()  # spawn
        self._expect(TokenType.LBRACE)
        body = self.parse_block_body(self.previous)
        
        return SpawnStmt(body=body, location=start.location)
    
    def parse_parallel_stmt(self) -> ParallelStmt:
        """Parse parallel statement."""
        start = self._advance()  # parallel
        self._expect(TokenType.LBRACE)
        
        tasks = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            name = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.ASSIGN)
            expr = self.parse_expression()
            tasks.append((name, expr))
            self._match(TokenType.SEMICOLON)
            self._match(TokenType.COMMA)
        
        self._expect(TokenType.RBRACE)
        
        return ParallelStmt(tasks=tasks, location=start.location)
    
    # === PROGRAM PARSING ===
    
    def parse_import(self) -> Import:
        """Parse use/import statement."""
        start = self._advance()  # use
        
        path = [self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value]
        while self._match(TokenType.TRAIT_IMPL):
            if self._check(TokenType.LBRACE, TokenType.STAR):
                break
            path.append(self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value)
        
        items = []
        alias = None
        is_wildcard = False
        
        if self._match(TokenType.TRAIT_IMPL) or self._check(TokenType.LBRACE, TokenType.STAR):
            if self._match(TokenType.LBRACE):
                # use path::{A, B, C}
                items.append(self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value)
                while self._match(TokenType.COMMA):
                    if self._check(TokenType.RBRACE):
                        break
                    items.append(self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value)
                self._expect(TokenType.RBRACE)
            elif self._match(TokenType.STAR):
                # use path::*
                is_wildcard = True
            else:
                # use path::item
                items.append(self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value)
        
        if self._match(TokenType.AS):
            alias = self._expect(TokenType.IDENTIFIER).value
        
        return Import(
            path=path,
            items=items,
            alias=alias,
            is_wildcard=is_wildcard,
            location=start.location
        )
    
    def parse_program(self) -> Program:
        """Parse entire program."""
        imports = []
        declarations = []
        module_path = None
        
        # Optional module declaration
        if self._match(TokenType.MODULE):
            path_parts = [self._expect(TokenType.IDENTIFIER).value]
            while self._match(TokenType.TRAIT_IMPL):
                path_parts.append(self._expect(TokenType.IDENTIFIER).value)
            module_path = '::'.join(path_parts)
        
        # Parse imports and declarations
        while not self._check(TokenType.EOF):
            if self._check(TokenType.USE):
                imports.append(self.parse_import())
            else:
                try:
                    declarations.append(self.parse_statement())
                except ParseError as e:
                    self._report_error(e)
                    self._sync()
        
        # If we accumulated errors, report them all
        if self.errors:
            raise ParseErrorCollection(self.errors)
        
        return Program(
            module_path=module_path,
            imports=imports,
            declarations=declarations
        )


# === AST UTILITIES ===

class ASTPrinter(ASTVisitor):
    """Pretty-print an AST for debugging."""
    
    def __init__(self):
        self.indent_level = 0
        self.output = StringIO()
    
    def indent(self) -> str:
        return "  " * self.indent_level
    
    def print(self, text: str):
        self.output.write(f"{self.indent()}{text}\n")
    
    def generic_visit(self, node: ASTNode) -> str:
        node_name = node.__class__.__name__
        self.print(f"{node_name}")
        self.indent_level += 1
        for child in node.children():
            self.visit(child)
        self.indent_level -= 1
        return self.output.getvalue()
    
    def visit_IntegerLiteral(self, node: IntegerLiteral) -> str:
        self.print(f"IntegerLiteral({node.value})")
        return self.output.getvalue()
    
    def visit_FloatLiteral(self, node: FloatLiteral) -> str:
        self.print(f"FloatLiteral({node.value})")
        return self.output.getvalue()
    
    def visit_StringLiteral(self, node: StringLiteral) -> str:
        self.print(f"StringLiteral({repr(node.value)})")
        return self.output.getvalue()
    
    def visit_BoolLiteral(self, node: BoolLiteral) -> str:
        self.print(f"BoolLiteral({node.value})")
        return self.output.getvalue()
    
    def visit_Identifier(self, node: Identifier) -> str:
        self.print(f"Identifier({node.name})")
        return self.output.getvalue()
    
    def visit_BinaryOp(self, node: BinaryOp) -> str:
        self.print(f"BinaryOp({node.operator})")
        self.indent_level += 1
        self.visit(node.left)
        self.visit(node.right)
        self.indent_level -= 1
        return self.output.getvalue()
    
    def visit_UnaryOp(self, node: UnaryOp) -> str:
        self.print(f"UnaryOp({node.operator})")
        self.indent_level += 1
        self.visit(node.operand)
        self.indent_level -= 1
        return self.output.getvalue()
    
    def visit_Call(self, node: Call) -> str:
        self.print("Call")
        self.indent_level += 1
        self.print("callee:")
        self.indent_level += 1
        self.visit(node.callee)
        self.indent_level -= 1
        self.print("arguments:")
        self.indent_level += 1
        for arg in node.arguments:
            self.visit(arg)
        self.indent_level -= 2
        return self.output.getvalue()
    
    def visit_FunctionDecl(self, node: FunctionDecl) -> str:
        self.print(f"FunctionDecl({node.name})")
        self.indent_level += 1
        self.print("params:")
        self.indent_level += 1
        for param in node.params:
            self.print(f"Parameter({param.name}: {param.param_type})")
        self.indent_level -= 1
        if node.return_type:
            self.print(f"return_type: {node.return_type}")
        if node.body:
            self.print("body:")
            self.indent_level += 1
            self.visit(node.body)
            self.indent_level -= 1
        self.indent_level -= 1
        return self.output.getvalue()


def print_ast(ast: ASTNode) -> str:
    """Pretty-print an AST."""
    printer = ASTPrinter()
    printer.visit(ast)
    return printer.output.getvalue()


def format_expression(expr: Expression) -> str:
    """Format an expression back to source code."""
    if isinstance(expr, IntegerLiteral):
        return str(expr.value)
    elif isinstance(expr, FloatLiteral):
        return str(expr.value)
    elif isinstance(expr, StringLiteral):
        return f'"{expr.value}"'
    elif isinstance(expr, BoolLiteral):
        return "true" if expr.value else "false"
    elif isinstance(expr, NoneLiteral):
        return "None"
    elif isinstance(expr, Identifier):
        return expr.name
    elif isinstance(expr, QualifiedName):
        return expr.full_name()
    elif isinstance(expr, BinaryOp):
        return f"({format_expression(expr.left)} {expr.operator} {format_expression(expr.right)})"
    elif isinstance(expr, UnaryOp):
        return f"{expr.operator}{format_expression(expr.operand)}"
    elif isinstance(expr, Call):
        args = ", ".join(format_expression(a) for a in expr.arguments)
        return f"{format_expression(expr.callee)}({args})"
    elif isinstance(expr, Member):
        return f"{format_expression(expr.object)}.{expr.member}"
    elif isinstance(expr, Index):
        return f"{format_expression(expr.object)}[{format_expression(expr.index)}]"
    elif isinstance(expr, ArrayLiteral):
        elems = ", ".join(format_expression(e) for e in expr.elements)
        return f"[{elems}]"
    elif isinstance(expr, TupleLiteral):
        elems = ", ".join(format_expression(e) for e in expr.elements)
        return f"({elems})"
    elif isinstance(expr, Lambda):
        params = ", ".join(p.name for p in expr.params)
        return f"fn({params}) => {format_expression(expr.body)}"
    else:
        return f"<{expr.__class__.__name__}>"


# === PUBLIC API ===

def parse(source: str, filename: str = "<input>") -> Program:
    """Parse source code into an AST."""
    parser = Parser(source, filename)
    return parser.parse_program()


def parse_expression_only(source: str) -> Expression:
    """Parse just an expression (for REPL/eval)."""
    parser = Parser(source, "<expr>")
    return parser.parse_expression()


def parse_file(filepath: str) -> Program:
    """Parse a .saaam file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    return parse(source, filepath)


def validate_syntax(source: str, filename: str = "<input>") -> list[ParseError]:
    """Validate syntax without raising exceptions, return list of errors."""
    parser = Parser(source, filename)
    try:
        parser.parse_program()
        return []
    except ParseErrorCollection as e:
        return e.errors
    except ParseError as e:
        return [e]


# === REPL HELPERS ===

def is_complete_input(source: str) -> bool:
    """Check if input is syntactically complete (for REPL)."""
    # Count braces, parens, brackets
    depth = {'(': 0, '[': 0, '{': 0}
    in_string = False
    string_char = None
    
    i = 0
    while i < len(source):
        c = source[i]
        
        # Handle escape sequences
        if c == '\\' and in_string:
            i += 2
            continue
        
        # Handle string boundaries
        if c in '"\'`' and not in_string:
            in_string = True
            string_char = c
        elif c == string_char and in_string:
            in_string = False
            string_char = None
        
        # Count delimiters outside strings
        if not in_string:
            if c == '(':
                depth['('] += 1
            elif c == ')':
                depth['('] -= 1
            elif c == '[':
                depth['['] += 1
            elif c == ']':
                depth['['] -= 1
            elif c == '{':
                depth['{'] += 1
            elif c == '}':
                depth['{'] -= 1
        
        i += 1
    
    # Complete if all delimiters balanced and not in string
    return (all(d == 0 for d in depth.values()) and 
            not in_string)
