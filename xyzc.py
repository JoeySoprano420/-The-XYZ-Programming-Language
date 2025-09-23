# xyzc.py
# XYZ Compiler (prototype)
# Lexer + Parser + Dodecagram AST Builder
# Author: XYZ Project

import re
from typing import List, Tuple, Union

# --------------------------
# Token Types
# --------------------------
KEYWORDS = {
    "start", "Start", "run", "main", "print",
    "try", "catch", "throw", "except", "isolate", "delete",
    "collect", "store", "flush", "sweep"
}

SYMBOLS = {"{", "}", "[", "]", "(", ")", ";", "--"}

# --------------------------
# Lexer
# --------------------------
class Token:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"<{self.type}:{self.value}>"

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
        if ch in "{}[]();":
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
# Parser → AST
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

# Very simple parser for `main() print ["Hello"] start`
def parse(tokens: List[Token]) -> ASTNode:
    root = ASTNode("Program")
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.value in {"Start", "main"}:
            node = ASTNode("Function", tok.value)
            root.children.append(node)
        elif tok.value == "print":
            val_tok = tokens[i+2]  # grab string after bracket
            node = ASTNode("Print", val_tok.value)
            root.children.append(node)
            i += 3
        elif tok.value in {"start", "run"}:
            node = ASTNode("Exec", tok.value)
            root.children.append(node)
        i += 1
    return root

# --------------------------
# Dodecagram Encoder (0–9,a,b)
# --------------------------
def encode_dodecagram(n: int) -> str:
    symbols = "0123456789ab"
    if n == 0:
        return "0"
    digits = ""
    while n > 0:
        digits = symbols[n % 12] + digits
        n //= 12
    return digits

def build_dodecagram_ast(ast: ASTNode, counter=[0]) -> dict:
    """Return AST with Dodecagram IDs"""
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
    main() print ["Hello, world"] start
    """
    toks = tokenize(code)
    print("TOKENS:", toks)

    ast = parse(toks)
    print("\nAST:\n", ast)

    dag = build_dodecagram_ast(ast)
    print("\nDODECAGRAM AST:", dag)
