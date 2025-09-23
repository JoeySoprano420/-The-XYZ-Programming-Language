# xyzc.py
# XYZ Compiler (prototype v2)
# Lexer + Parser + Dodecagram AST Builder with blocks + arithmetic
# Author: XYZ Project

import re
from typing import List, Union

# --------------------------
# Token Types
# --------------------------
KEYWORDS = {
    "start", "Start", "run", "main", "print",
    "try", "catch", "throw", "except", "isolate", "delete",
    "collect", "store", "flush", "sweep"
}

SYMBOLS = {"{", "}", "[", "]", "(", ")", ";", "=", "+", "-", "*", "/", "%"}

class Token:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"<{self.type}:{self.value}>"

# --------------------------
# Lexer
# --------------------------
def tokenize(source: str) -> List[Token]:
    tokens = []
    i = 0
    while i < len(source):
        ch = source[i]

        # Whitespace
        if ch.isspace():
            i += 1
            continue

        # Comments
        if ch == ';':  # single line
            while i < len(source) and source[i] != '\n':
                i += 1
            continue
        if ch == '-' and i + 1 < len(source) and source[i+1] == '-':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Symbols
        if ch in SYMBOLS:
            tokens.append(Token("SYMBOL", ch))
            i += 1
            continue

        # Strings
        if ch in {'"', '“', '”'}:
            quote = ch
            i += 1
            val = ""
            while i < len(source) and source[i] not in {'"', '“', '”'}:
                val += source[i]
                i += 1
            i += 1
            tokens.append(Token("STRING", val))
            continue

        # Numbers
        if ch.isdigit():
            val = ""
            while i < len(source) and source[i].isdigit():
                val += source[i]
                i += 1
            tokens.append(Token("NUMBER", val))
            continue

        # Identifiers / keywords
        if ch.isalpha():
            val = ""
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                val += source[i]
                i += 1
            if val in KEYWORDS:
                tokens.append(Token("KEYWORD", val))
            else:
                tokens.append(Token("IDENT", val))
            continue

        # Unknown
        tokens.append(Token("UNKNOWN", ch))
        i += 1

    return tokens

# --------------------------
# AST
# --------------------------
class ASTNode:
    def __init__(self, nodetype: str, value: str = "", children=None):
        self.nodetype = nodetype
        self.value = value
        self.children = children or []

    def __repr__(self, level=0):
        indent = "  " * level
        res = f"{indent}{self.nodetype}({self.value})\n"
        for child in self.children:
            res += child.__repr__(level+1)
        return res

# --------------------------
# Parser
# --------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        tok = self.peek()
        if tok: self.pos += 1
        return tok

    def expect(self, value):
        tok = self.advance()
        if not tok or tok.value != value:
            raise SyntaxError(f"Expected {value}, got {tok}")
        return tok

    # ---- Grammar ----
    def parse_program(self) -> ASTNode:
        root = ASTNode("Program")
        while self.peek():
            root.children.append(self.parse_statement())
        return root

    def parse_statement(self) -> ASTNode:
        tok = self.peek()
        if not tok: return None

        if tok.value in {"main", "Start"}:   # function entry
            return self.parse_function()
        if tok.value == "print":
            return self.parse_print()
        if tok.type == "IDENT":              # assignment or expr
            return self.parse_assignment()
        return self.parse_expression()

    def parse_function(self) -> ASTNode:
        name = self.advance().value
        self.expect("(")
        self.expect(")")
        block = self.parse_block()
        return ASTNode("Function", name, [block])

    def parse_block(self) -> ASTNode:
        tok = self.peek()
        if tok and tok.value in {"{", "["}:
            open_brace = self.advance().value
            close_brace = "}" if open_brace == "{" else "]"
            block = ASTNode("Block")
            while self.peek() and self.peek().value != close_brace:
                block.children.append(self.parse_statement())
            self.expect(close_brace)
            return block
        else:
            # Single-statement implicit block
            stmt = self.parse_statement()
            return ASTNode("Block", children=[stmt])

    def parse_print(self) -> ASTNode:
        self.advance()  # 'print'
        self.expect("[")
        expr = self.parse_expression()
        self.expect("]")
        return ASTNode("Print", children=[expr])

    def parse_assignment(self) -> ASTNode:
        ident = self.advance().value
        self.expect("=")
        expr = self.parse_expression()
        return ASTNode("Assign", ident, [expr])

    def parse_expression(self) -> ASTNode:
        node = self.parse_term()
        while self.peek() and self.peek().value in {"+", "-"}:
            op = self.advance().value
            right = self.parse_term()
            node = ASTNode("BinaryOp", op, [node, right])
        return node

    def parse_term(self) -> ASTNode:
        node = self.parse_factor()
        while self.peek() and self.peek().value in {"*", "/", "%"}:
            op = self.advance().value
            right = self.parse_factor()
            node = ASTNode("BinaryOp", op, [node, right])
        return node

    def parse_factor(self) -> ASTNode:
        tok = self.advance()
        if tok.type == "NUMBER":
            return ASTNode("Number", tok.value)
        if tok.type == "STRING":
            return ASTNode("String", tok.value)
        if tok.type == "IDENT":
            return ASTNode("Var", tok.value)
        if tok.value == "(":
            expr = self.parse_expression()
            self.expect(")")
            return expr
        raise SyntaxError(f"Unexpected token {tok}")

# --------------------------
# Dodecagram Encoder (0–9,a,b)
# --------------------------
def encode_dodecagram(n: int) -> str:
    symbols = "0123456789ab"
    if n == 0: return "0"
    digits = ""
    while n > 0:
        digits = symbols[n % 12] + digits
        n //= 12
    return digits

def build_dodecagram_ast(ast: ASTNode, counter=[0]) -> dict:
    node_id = encode_dodecagram(counter[0])
    counter[0] += 1
    return {
        "id": node_id,
        "type": ast.nodetype,
        "value": ast.value,
        "children": [build_dodecagram_ast(c, counter) for c in ast.children]
    }

# --------------------------
# Demo
# --------------------------
if __name__ == "__main__":
    code = """
    main() {
        x = 5 + 3 * 2
        print [x]
    }
    """
    toks = tokenize(code)
    print("TOKENS:", toks)

    parser = Parser(toks)
    ast = parser.parse_program()
    print("\nAST:\n", ast)

    dag = build_dodecagram_ast(ast)
    print("\nDODECAGRAM AST:", dag)
