"""
SAAAM Language - Parser
The Neural Network: Transforms token streams into living AST structures.

This is where syntax becomes MEANING.
Pratt parsing meets recursive descent meets PURE INNOVATION.
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass

from .tokens import Token, TokenType, PRECEDENCE, RIGHT_ASSOCIATIVE, get_precedence
from .lexer import Lexer, LexerError
from .ast_nodes import *


class ParseError(Exception):
    """When the neural pathway can't be formed."""
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"{token.location}: {message}")


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
    
    def _expect(self, token_type: TokenType, message: str = None) -> Token:
        """Expect a specific token type, raise error if not found."""
        if not self._check(token_type):
            msg = message or f"Expected {token_type.name}, got {self.current.type.name}"
            raise ParseError(msg, self.current)
        return self._advance()
    
    def _error(self, message: str) -> ParseError:
        """Create a parse error at current position."""
        return ParseError(message, self.current)
    
    def _sync(self):
        """Synchronize after error to continue parsing."""
        self._advance()
        while not self._check(TokenType.EOF):
            if self.previous.type == TokenType.SEMICOLON:
                return
            if self._check(TokenType.FN, TokenType.LET, TokenType.VAR,
                          TokenType.STRUCT, TokenType.ENUM, TokenType.TRAIT,
                          TokenType.IMPL, TokenType.IF, TokenType.FOR,
                          TokenType.WHILE, TokenType.RETURN):
                return
            self._advance()
    
    def _skip_newlines(self):
        """Skip any newline tokens."""
        while self._check(TokenType.NEWLINE):
            self._advance()
    
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
    
    def parse_identifier(self) -> Expression:
        """Parse identifier or qualified name."""
        token = self._advance()
        name = Identifier(name=token.value, location=token.location)
        
        # Check for qualified name
        if self._check(TokenType.TRAIT_IMPL):
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
        """Parse lambda: fn(x) => x * 2 or |x| x * 2."""
        start = self._advance()  # fn
        
        params = self.parse_parameter_list()
        
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
        
        tag = self._expect(TokenType.IDENTIFIER).value
        
        # Parse attributes
        attributes = []
        while not self._check(TokenType.GT, TokenType.SLASH, TokenType.EOF):
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
            else:
                break
        
        # Closing tag
        self._expect(TokenType.LT)
        self._expect(TokenType.SLASH)
        closing_tag = self._expect(TokenType.IDENTIFIER).value
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
        if not self._check(TokenType.RBRACKET, TokenType.RPAREN, TokenType.COMMA):
            end = self.parse_expression(get_precedence(TokenType.RANGE) + 1)
        
        return RangeExpr(
            start=left,
            end=end,
            inclusive=inclusive,
            location=op.location
        )
    
    def parse_member(self, left: Expression) -> Member:
        """Parse member access: obj.field."""
        self._advance()  # .
        member = self._expect(TokenType.IDENTIFIER).value
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
        """Parse function call: func(args)."""
        self._advance()  # (
        
        arguments = []
        if not self._check(TokenType.RPAREN):
            arguments.append(self.parse_expression())
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RPAREN):
                    break
                arguments.append(self.parse_expression())
        
        self._expect(TokenType.RPAREN)
        return Call(
            callee=left,
            arguments=arguments,
            location=left.location
        )
    
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
                    'Float32', 'Float64'}
        
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
            return LiteralPattern(value=self.parse_expression(), location=self.current.location)
        
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
        
        if self._match(TokenType.LPAREN):
            # Tuple pattern
            elements = []
            if not self._check(TokenType.RPAREN):
                elements.append(self.parse_pattern())
                while self._match(TokenType.COMMA):
                    if self._check(TokenType.RPAREN):
                        break
                    elements.append(self.parse_pattern())
            self._expect(TokenType.RPAREN)
            return TuplePattern(elements=elements, location=self.previous.location)
        
        if self._match(TokenType.LBRACKET):
            # Array pattern
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
            return ArrayPattern(elements=elements, rest=rest, location=self.previous.location)
        
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
            
            return IdentifierPattern(name=name_tok.value, location=name_tok.location)
        
        if self._check(TokenType.TYPE_IDENTIFIER):
            # Capitalized = enum variant or type
            name_tok = self._advance()
            
            if self._match(TokenType.LPAREN):
                fields = []
                if not self._check(TokenType.RPAREN):
                    fields.append(self.parse_pattern())
                    while self._match(TokenType.COMMA):
                        fields.append(self.parse_pattern())
                self._expect(TokenType.RPAREN)
                return EnumPattern(variant=name_tok.value, fields=fields, location=name_tok.location)
            
            return EnumPattern(variant=name_tok.value, fields=[], location=name_tok.location)
        
        raise self._error(f"Expected pattern, got {self.current.type.name}")
    
    # === STATEMENT PARSING ===
    
    def parse_statement(self) -> Statement:
        """Parse a statement."""
        # Skip any annotations
        annotations = []
        while self._match(TokenType.ANNOTATION):
            annotations.append(self.previous.value)
        
        # Declarations
        if self._check(TokenType.LET, TokenType.VAR, TokenType.CONST, 
                      TokenType.NEURAL, TokenType.GC):
            return self.parse_var_decl()
        
        if self._check(TokenType.FN):
            return self.parse_function_decl(annotations)
        
        if self._check(TokenType.STRUCT):
            return self.parse_struct_decl()
        
        if self._check(TokenType.ENUM):
            return self.parse_enum_decl()
        
        if self._check(TokenType.TRAIT):
            return self.parse_trait_decl()
        
        if self._check(TokenType.IMPL):
            return self.parse_impl_decl()
        
        if self._check(TokenType.TYPE):
            return self.parse_type_alias()
        
        if self._check(TokenType.COMPONENT):
            return self.parse_component_decl()
        
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
            if not self._check(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.NEWLINE):
                value = self.parse_expression()
            return BreakStmt(value=value, location=self.previous.location)
        
        if self._match(TokenType.CONTINUE):
            return ContinueStmt(location=self.previous.location)
        
        if self._match(TokenType.RETURN):
            value = None
            if not self._check(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.NEWLINE):
                value = self.parse_expression()
            return ReturnStmt(value=value, location=self.previous.location)
        
        if self._match(TokenType.YIELD):
            value = None
            if not self._check(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.NEWLINE):
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
        
        # Handle modifiers like: gc var x
        if is_gc and self._match(TokenType.VAR):
            is_mutable = True
        
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
            is_mutable=is_mutable,
            is_const=is_const,
            is_neural=is_neural,
            is_gc=is_gc,
            location=start.location
        )
    
    def parse_parameter_list(self) -> list[Parameter]:
        """Parse function parameter list."""
        self._expect(TokenType.LPAREN)
        
        params = []
        while not self._check(TokenType.RPAREN, TokenType.EOF):
            is_variadic = self._match(TokenType.ELLIPSIS) is not None
            is_mutable = self._match(TokenType.VAR) is not None
            
            name = self._expect(TokenType.IDENTIFIER).value
            
            param_type = None
            if self._match(TokenType.COLON):
                param_type = self.parse_type()
            
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
        if start.type == TokenType.ASYNC:
            is_async = True
            self._expect(TokenType.FN)
        
        name = self._expect(TokenType.IDENTIFIER).value
        
        # Type parameters
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        params = self.parse_parameter_list()
        
        return_type = None
        if self._match(TokenType.FLOW):
            return_type = self.parse_type()
        
        # Effects
        effects = []
        if self._match(TokenType.WITH):
            effects.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                effects.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
        
        body = None
        if self._match(TokenType.LBRACE):
            body = self.parse_block_body(self.previous)
        
        return FunctionDecl(
            name=name,
            params=params,
            return_type=return_type,
            body=body,
            is_async=is_async,
            type_params=type_params,
            effects=effects,
            annotations=annotations or [],
            location=start.location
        )
    
    def parse_struct_decl(self) -> StructDecl:
        """Parse struct declaration."""
        start = self._advance()  # struct
        name = self._expect(TokenType.TYPE_IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
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
        name = self._expect(TokenType.TYPE_IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
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
        name = self._expect(TokenType.TYPE_IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        super_traits = []
        if self._match(TokenType.COLON):
            super_traits.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.PLUS):
                super_traits.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
        
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
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            self._expect(TokenType.GT)
        
        # Could be "impl Type" or "impl Trait for Type"
        first_type = self.parse_type()
        
        trait_name = None
        target_type = first_type
        
        if self._match(TokenType.FOR):
            trait_name = first_type.name.full_name() if isinstance(first_type, NamedType) else str(first_type)
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
        name = self._expect(TokenType.TYPE_IDENTIFIER).value
        
        type_params = []
        if self._match(TokenType.LT):
            type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                type_params.append(self._expect(TokenType.TYPE_IDENTIFIER).value)
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
        name = self._expect(TokenType.TYPE_IDENTIFIER).value
        
        self._expect(TokenType.LBRACE)
        
        state = []
        props = []
        methods = []
        lifecycle = {}
        render = None
        
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            if self._match(TokenType.STATE):
                state.append(self.parse_var_decl())
            elif self._match(TokenType.PROPS):
                self._expect(TokenType.LBRACE)
                while not self._check(TokenType.RBRACE):
                    props.append(self.parse_var_decl())
                self._expect(TokenType.RBRACE)
            elif self._match(TokenType.ON):
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
        
        return WhileStmt(
            condition=condition,
            body=body,
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
        self._expect(TokenType.LBRACE)
        body = self.parse_block_body(self.previous)
        
        return LoopStmt(body=body, location=start.location)
    
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
        
        self._expect(TokenType.RBRACE)
        
        return ParallelStmt(tasks=tasks, location=start.location)
    
    # === PROGRAM PARSING ===
    
    def parse_import(self) -> Import:
        """Parse use/import statement."""
        start = self._advance()  # use
        
        path = [self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value]
        while self._match(TokenType.TRAIT_IMPL):
            path.append(self._expect(TokenType.IDENTIFIER, TokenType.TYPE_IDENTIFIER).value)
        
        items = []
        alias = None
        is_wildcard = False
        
        if self._match(TokenType.TRAIT_IMPL):
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
                    print(f"Parse error: {e}")
                    self._sync()
        
        return Program(
            module_path=module_path,
            imports=imports,
            declarations=declarations
        )


def parse(source: str, filename: str = "<input>") -> Program:
    """Parse source code into an AST."""
    parser = Parser(source, filename)
    return parser.parse_program()


def parse_file(filepath: str) -> Program:
    """Parse a .saaam file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    return parse(source, filepath)
