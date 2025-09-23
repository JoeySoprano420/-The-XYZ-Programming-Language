# xyzc.py
# XYZ Compiler (prototype v6)
# Adds: typed functions, overloading by type, pragmas, lambdas, primitives,
# recursion, parallelism, deltas, decimals, negatives, div-by-zero safe ops
# Author: XYZ Project

import re
from typing import List
import hashlib
from typing import List

# --------------------------
# Token Types
# --------------------------
KEYWORDS = {
    "main", "Start", "print", "return",
    "if", "else", "while", "for", "parallel", "lambda",
    "int", "float", "string", "bool",
    "try", "catch", "throw", "except", "isolate", "delete",
    "collect", "store", "flush", "sweep", "delta"
}

SYMBOLS = {
    "{", "}", "[", "]", "(", ")", ";", "=", "+", "-", "*", "/", "%",
    "<", ">", "==", ",", ":", "->"
}

PRAGMAS = {"#pragma"}

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

        # Pragmas
        if source.startswith("#pragma", i):
            val = "#pragma"
            tokens.append(Token("PRAGMA", val))
            i += len(val)
            continue

        # Comments
        if ch == ';':  # single line
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Multi-char symbols
        if source.startswith("==", i):
            tokens.append(Token("SYMBOL", "=="))
            i += 2
            continue
        if source.startswith("->", i):
            tokens.append(Token("SYMBOL", "->"))
            i += 2
            continue

        # Symbols
        if ch in SYMBOLS:
            tokens.append(Token("SYMBOL", ch))
            i += 1
            continue

        # Strings
        if ch == '"' or ch in {'“','”'}:
            i += 1
            val = ""
            while i < len(source) and source[i] not in {'"', '“', '”'}:
                val += source[i]
                i += 1
            i += 1
            tokens.append(Token("STRING", val))
            continue

        # Numbers (support decimals, negatives)
        if ch.isdigit() or (ch == "-" and i+1 < len(source) and source[i+1].isdigit()):
            val = ch
            i += 1
            dot_seen = (ch == ".")
            while i < len(source) and (source[i].isdigit() or (source[i] == "." and not dot_seen)):
                if source[i] == ".":
                    dot_seen = True
                val += source[i]
                i += 1
            tokens.append(Token("NUMBER", val))
            continue

        # Identifiers / keywords
        if ch.isalpha():
            val = ""
            while i < len(source) and (source[i].isalnum() or source[i] in {"_", "-"}):
                val += source[i]
                i += 1
            if val in KEYWORDS:
                tokens.append(Token("KEYWORD", val))
            else:
                tokens.append(Token("IDENT", val))
            continue

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
        self.functions = {}  # { name: { (arity, types): ASTNode } }

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

    # ---- Program ----
    def parse_program(self) -> ASTNode:
        root = ASTNode("Program")
        while self.peek():
            stmt = self.parse_statement()
            if stmt: root.children.append(stmt)
        return root

    def parse_statement(self) -> ASTNode:
        tok = self.peek()
        if not tok: return None

        if tok.type == "PRAGMA":
            return self.parse_pragma()
        if tok.value in {"main","Start"} or (tok.type=="IDENT" and self.lookahead_is_func_def()):
            return self.parse_function()
        if tok.type == "KEYWORD" and tok.value == "return":
            return self.parse_return()
        if tok.value == "print":
            return self.parse_print()
        if tok.value == "if":
            return self.parse_if()
        if tok.value == "while":
            return self.parse_while()
        if tok.value == "for":
            return self.parse_for()
        if tok.value == "parallel":
            return self.parse_parallel()
        if tok.value == "lambda":
            return self.parse_lambda()
        if tok.type == "IDENT":
            return self.parse_assignment_or_call()
        return self.parse_expression()

    def lookahead_is_func_def(self) -> bool:
        # ident "(" paramlist ")"
        return (self.pos+1 < len(self.tokens) and self.tokens[self.pos+1].value == "(")

    # ---- Pragmas ----
    def parse_pragma(self) -> ASTNode:
        self.advance() # consume #pragma
        name = self.advance().value
        return ASTNode("Pragma", name)

    # ---- Functions ----
    def parse_function(self) -> ASTNode:
        name = self.advance().value
        self.expect("(")
        params, types = [], []
        while self.peek() and self.peek().value != ")":
            p = self.advance().value  # param name
            self.expect(":")
            t = self.advance().value  # type
            params.append(p)
            types.append(t)
            if self.peek() and self.peek().value == ",":
                self.advance()
        self.expect(")")
        block = self.parse_block()
        sig = (len(params), tuple(types))
        func_node = ASTNode("Function", f"{name}/{sig}", [
            ASTNode("Params", children=[ASTNode("Param", f"{p}:{t}") for p,t in zip(params,types)]),
            block
        ])
        if name not in self.functions: self.functions[name] = {}
        if sig in self.functions[name]:
            raise SyntaxError(f"Function {name} with signature {sig} already defined")
        self.functions[name][sig] = func_node
        return func_node

    # ---- Blocks ----
    def parse_block(self) -> ASTNode:
        tok = self.peek()
        if tok and tok.value in {"{","["}:
            close = "}" if tok.value == "{" else "]"
            self.advance()
            block = ASTNode("Block")
            while self.peek() and self.peek().value != close:
                block.children.append(self.parse_statement())
            self.expect(close)
            return block
        else:
            stmt = self.parse_statement()
            return ASTNode("Block", children=[stmt])

    # ---- Constructs ----
    def parse_print(self): 
        self.advance(); self.expect("["); expr=self.parse_expression(); self.expect("]")
        return ASTNode("Print", children=[expr])

    def parse_return(self): 
        self.advance(); expr=self.parse_expression()
        return ASTNode("Return", children=[expr])

    def parse_if(self):
        self.advance(); self.expect("("); cond=self.parse_expression(); self.expect(")")
        thenb=self.parse_block(); elseb=None
        if self.peek() and self.peek().value=="else":
            self.advance(); elseb=self.parse_block()
        return ASTNode("If", children=[cond,thenb]+([elseb] if elseb else []))

    def parse_while(self):
        self.advance(); self.expect("("); cond=self.parse_expression(); self.expect(")")
        body=self.parse_block(); return ASTNode("While",children=[cond,body])

    def parse_for(self):
        self.advance(); self.expect("(")
        init=self.parse_assignment_or_call(); cond=self.parse_expression()
        self.expect(";"); step=self.parse_assignment_or_call(); self.expect(")")
        body=self.parse_block()
        return ASTNode("For",children=[init,cond,step,body])

    def parse_parallel(self):
        self.advance(); body=self.parse_block()
        return ASTNode("Parallel",children=[body])

    def parse_lambda(self):
        self.advance(); self.expect("(")
        params=[]; 
        while self.peek() and self.peek().value!=")":
            p=self.advance().value
            if self.peek() and self.peek().value==":":
                self.advance(); t=self.advance().value; p=f"{p}:{t}"
            params.append(p)
            if self.peek() and self.peek().value==",":
                self.advance()
        self.expect(")")
        self.expect("->")
        expr=self.parse_expression()
        return ASTNode("Lambda",children=[ASTNode("Params",children=[ASTNode("Param",p) for p in params]),expr])

    # ---- Assignment / Call ----
    def parse_assignment_or_call(self):
        ident=self.advance().value
        if self.peek() and self.peek().value=="=":
            self.advance(); expr=self.parse_expression()
            return ASTNode("Assign",ident,[expr])
        elif self.peek() and self.peek().value=="(":
            self.expect("(")
            args=[]
            while self.peek() and self.peek().value!=")":
                args.append(self.parse_expression())
                if self.peek() and self.peek().value==",":
                    self.advance()
            self.expect(")")
            sig=(len(args),"*") # simplified: type resolution could go here
            return ASTNode("Call",f"{ident}/{sig}",args)
        return ASTNode("Var",ident)

    # ---- Expressions ----
    def parse_expression(self):
        node=self.parse_term()
        while self.peek() and self.peek().value in {"+","-","==","<",">"}:
            op=self.advance().value; right=self.parse_term()
            node=ASTNode("BinaryOp",op,[node,right])
        return node

    def parse_term(self):
        node=self.parse_factor()
        while self.peek() and self.peek().value in {"*","/","%"}:
            op=self.advance().value; right=self.parse_factor()
            if op=="/":  # safe div node
                node=ASTNode("SafeDiv",op,[node,right])
            else:
                node=ASTNode("BinaryOp",op,[node,right])
        return node

    def parse_factor(self):
        tok=self.advance()
        if tok.type=="NUMBER": return ASTNode("Number",tok.value)
        if tok.type=="STRING": return ASTNode("String",tok.value)
        if tok.type=="IDENT":
            if self.peek() and self.peek().value=="(":
                self.expect("("); args=[]
                while self.peek() and self.peek().value!=")":
                    args.append(self.parse_expression())
                    if self.peek() and self.peek().value==",":
                        self.advance()
                self.expect(")")
                return ASTNode("Call",f"{tok.value}/{len(args)}",args)
            return ASTNode("Var",tok.value)
        if tok.value=="(":
            expr=self.parse_expression(); self.expect(")"); return expr
        if tok.value=="-": # negative numbers
            val=self.parse_factor(); return ASTNode("Neg",children=[val])
        if tok.value=="delta": # delta operator
            var=self.advance().value; expr=self.parse_expression()
            return ASTNode("Delta",var,[expr])
        raise SyntaxError(f"Unexpected token {tok}")

# --------------------------
# Dodecagram Encoder (0–9,a,b)
# --------------------------
def encode_dodecagram(n:int)->str:
    symbols="0123456789ab"
    if n==0:return "0"
    out=""
    while n>0: out=symbols[n%12]+out; n//=12
    return out

def build_dodecagram_ast(ast:ASTNode,counter=[0])->dict:
    node_id=encode_dodecagram(counter[0]); counter[0]+=1
    return {
        "id":node_id,
        "type":ast.nodetype,
        "value":ast.value,
        "children":[build_dodecagram_ast(c,counter) for c in ast.children]
    }

# --------------------------
# Demo
# --------------------------
if __name__=="__main__":
    code="""
    #pragma parallel

    Factorial(n:int) {
        if(n == 0) { return 1 }
        else { return n * Factorial(n - 1) }
    }

    Add(x:int, y:int) { return x + y }
    Add(x:string, y:string) { return x + y }

    main() {
        a = Factorial(5)
        b = Add(3,4)
        c = Add("Hi, ","there")
        f = lambda(x:int,y:int) -> x + y
        print [a]
        print [b]
        print [c]
        print [f(10,20)]
        parallel { print ["running in parallel"] }
        delta dx x - 1
        z = -3.14 / 0  ; safe div by zero
    }
    """
    toks=tokenize(code)
    print("TOKENS:",toks)

    parser=Parser(toks)
    ast=parser.parse_program()
    print("\nAST:\n",ast)

    dag=build_dodecagram_ast(ast)
    print("\nDODECAGRAM AST:",dag)

# --------------------------
# Token Types
# --------------------------
KEYWORDS = {
    "main", "Start", "print", "return", "if", "else",
    "while", "for", "do", "loop", "switch", "case",
    "parallel", "lambda", "enum", "eval", "poly", "derivative",
    "true", "false", "null",
    "int", "float", "string", "bool", "nulltype"
}

SYMBOLS = {
    "{", "}", "[", "]", "(", ")", ";", "=", "+", "-", "*", "/", "%",
    "<", ">", "==", ",", ":", "->", "^"
}

PRAGMAS = {"#pragma"}

class Token:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value
    def __repr__(self):
        return f"<{self.type}:{self.value}>"

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
# Lexer
# --------------------------
def tokenize(src: str) -> List[Token]:
    i, tokens = 0, []
    while i < len(src):
        ch = src[i]

        if src.startswith("#pragma", i):
            tokens.append(Token("PRAGMA", "#pragma"))
            i += 7; continue

        if ch.isspace():
            i += 1; continue

        if src.startswith("==", i):
            tokens.append(Token("SYMBOL","==")); i+=2; continue
        if src.startswith("->", i):
            tokens.append(Token("SYMBOL","->")); i+=2; continue

        if ch in SYMBOLS:
            tokens.append(Token("SYMBOL",ch)); i+=1; continue

        if ch=='"':
            i+=1; val=""
            while i<len(src) and src[i]!='"':
                val+=src[i]; i+=1
            i+=1; tokens.append(Token("STRING",val)); continue

        if ch.isdigit() or (ch=="-" and i+1<len(src) and src[i+1].isdigit()):
            val=ch; i+=1
            while i<len(src) and (src[i].isdigit() or src[i]=="."):
                val+=src[i]; i+=1
            tokens.append(Token("NUMBER",val)); continue

        if ch.isalpha():
            val=""
            while i<len(src) and (src[i].isalnum() or src[i]=="_"):
                val+=src[i]; i+=1
            if val in KEYWORDS: tokens.append(Token("KEYWORD",val))
            else: tokens.append(Token("IDENT",val))
            continue

        i+=1
    return tokens

# --------------------------
# Type Checker
# --------------------------
class TypeEnv:
    def __init__(self):
        self.symbols={}  # var → type
    def set(self,name,typ): self.symbols[name]=typ
    def get(self,name): return self.symbols.get(name,"unknown")

def type_check(node:ASTNode, env:TypeEnv):
    if node.nodetype=="Number":
        return "float" if "." in node.value else "int"
    if node.nodetype=="String": return "string"
    if node.nodetype=="Bool": return "bool"
    if node.nodetype=="Null": return "nulltype"
    if node.nodetype=="Var":
        return env.get(node.value)
    if node.nodetype=="Assign":
        typ=type_check(node.children[0],env)
        env.set(node.value,typ); return typ
    if node.nodetype=="BinaryOp":
        lt=type_check(node.children[0],env)
        rt=type_check(node.children[1],env)
        if lt!=rt: raise TypeError(f"Type mismatch {lt} vs {rt}")
        return lt
    if node.nodetype=="Call":
        return "int" # simplified placeholder
    if node.nodetype=="Derivative": return "poly"
    if node.nodetype=="Eval": return "int"
    return "unknown"

# --------------------------
# Parser (partial demo subset)
# --------------------------
class Parser:
    def __init__(self,tokens:List[Token]):
        self.tokens=tokens; self.pos=0
    def peek(self): return self.tokens[self.pos] if self.pos<len(self.tokens) else None
    def advance(self): tok=self.peek(); self.pos+=1; return tok
    def expect(self,v): tok=self.advance(); 
        # skipping real check for brevity
        return tok

    def parse_program(self)->ASTNode:
        root=ASTNode("Program")
        while self.peek(): root.children.append(self.parse_statement())
        return root

    def parse_statement(self)->ASTNode:
        tok=self.peek()
        if not tok: return None
        if tok.value=="enum": return self.parse_enum()
        if tok.value=="eval": return self.parse_eval()
        if tok.value=="derivative": return self.parse_derivative()
        if tok.value in {"true","false"}: self.advance(); return ASTNode("Bool",tok.value)
        if tok.value=="null": self.advance(); return ASTNode("Null","null")
        if tok.type=="NUMBER": self.advance(); return ASTNode("Number",tok.value)
        if tok.type=="STRING": self.advance(); return ASTNode("String",tok.value)
        if tok.type=="IDENT": return self.parse_assignment()
        self.advance(); return ASTNode("Unknown",tok.value)

    def parse_enum(self)->ASTNode:
        self.advance() # enum
        name=self.advance().value
        self.expect("{")
        values=[]
        while self.peek() and self.peek().value!="}":
            v=self.advance().value
            if v!="}": values.append(ASTNode("EnumVal",v))
            if self.peek() and self.peek().value==",": self.advance()
        self.expect("}")
        return ASTNode("Enum",name,values)

    def parse_eval(self)->ASTNode:
        self.advance(); self.expect("(")
        expr=self.advance().value
        self.expect(")")
        return ASTNode("Eval",expr)

    def parse_derivative(self)->ASTNode:
        self.advance(); self.expect("(")
        var=self.advance().value
        expr=self.advance().value
        self.expect(")")
        return ASTNode("Derivative",var,[ASTNode("Expr",expr)])

    def parse_assignment(self)->ASTNode:
        name=self.advance().value
        if self.peek() and self.peek().value=="=":
            self.advance()
            expr=self.advance().value
            return ASTNode("Assign",name,[ASTNode("Var",expr)])
        return ASTNode("Var",name)

# --------------------------
# Demo
# --------------------------
if __name__=="__main__":
    code="""
    enum Colors { Red, Green, Blue }
    x = Colors
    eval("3+4")
    derivative(x x^2+3x)
    y = true
    z = null
    """
    toks=tokenize(code)
    print("TOKENS:",toks)

    parser=Parser(toks)
    ast=parser.parse_program()
    print("\nAST:\n",ast)

    env=TypeEnv()
    for child in ast.children:
        try: print(f"{child.nodetype} -> {type_check(child,env)}")
        except Exception as e: print(f"Type error: {e}")

# xyzc.py
# XYZ Compiler (prototype v8)
# Adds: switch/case, do/while, loop, polynomial expressions,
# obfuscation + polymorphism engines
# Author: XYZ Project

import hashlib
from typing import List

# --------------------------
# Token + Lexer
# --------------------------
KEYWORDS = {
    "main","Start","print","return",
    "if","else","while","for","do","loop",
    "switch","case","default","break",
    "parallel","lambda","enum","eval","poly","derivative",
    "true","false","null",
    "int","float","string","bool","nulltype"
}

SYMBOLS = {
    "{","}","[","]","(",")",";","=", "+","-","*","/","%","<",">","==",",",":","->","^"
}

PRAGMAS = {"#pragma"}

class Token:
    def __init__(self,type_:str,value:str):
        self.type=type_; self.value=value
    def __repr__(self): return f"<{self.type}:{self.value}>"

def tokenize(src:str)->List[Token]:
    i,toks=0,[]
    while i<len(src):
        ch=src[i]
        if src.startswith("#pragma",i):
            toks.append(Token("PRAGMA","#pragma")); i+=7; continue
        if ch.isspace(): i+=1; continue
        if src.startswith("==",i): toks.append(Token("SYMBOL","==")); i+=2; continue
        if src.startswith("->",i): toks.append(Token("SYMBOL","->")); i+=2; continue
        if ch in SYMBOLS: toks.append(Token("SYMBOL",ch)); i+=1; continue
        if ch=='"': i+=1; val=""
        # string
            while i<len(src) and src[i]!='"':
                val+=src[i]; i+=1
            i+=1; toks.append(Token("STRING",val)); continue
        if ch.isdigit() or (ch=="-" and i+1<len(src) and src[i+1].isdigit()):
            val=ch; i+=1
            while i<len(src) and (src[i].isdigit() or src[i]=="."):
                val+=src[i]; i+=1
            toks.append(Token("NUMBER",val)); continue
        if ch.isalpha():
            val=""
            while i<len(src) and (src[i].isalnum() or src[i]=="_"):
                val+=src[i]; i+=1
            if val in KEYWORDS: toks.append(Token("KEYWORD",val))
            else: toks.append(Token("IDENT",val))
            continue
        i+=1
    return toks

# --------------------------
# AST
# --------------------------
class ASTNode:
    def __init__(self,nodetype:str,value:str="",children=None):
        self.nodetype=nodetype; self.value=value; self.children=children or []
    def __repr__(self,level=0):
        ind="  "*level; s=f"{ind}{self.nodetype}({self.value})\n"
        for c in self.children: s+=c.__repr__(level+1)
        return s

# --------------------------
# Parser
# --------------------------
class Parser:
    def __init__(self,toks:List[Token]):
        self.toks=toks; self.pos=0
    def peek(self): return self.toks[self.pos] if self.pos<len(self.toks) else None
    def advance(self): t=self.peek(); self.pos+=1; return t
    def expect(self,v): t=self.advance(); 
        if not t or t.value!=v: raise SyntaxError(f"Expected {v}, got {t}")
        return t

    def parse_program(self)->ASTNode:
        root=ASTNode("Program")
        while self.peek(): root.children.append(self.parse_statement())
        return root

    def parse_statement(self)->ASTNode:
        tok=self.peek()
        if not tok: return None
        if tok.type=="PRAGMA": return self.parse_pragma()
        if tok.value=="switch": return self.parse_switch()
        if tok.value=="do": return self.parse_do_while()
        if tok.value=="loop": return self.parse_loop()
        if tok.value=="poly": return self.parse_polynomial()
        if tok.type=="NUMBER": self.advance(); return ASTNode("Number",tok.value)
        if tok.type=="STRING": self.advance(); return ASTNode("String",tok.value)
        if tok.value in {"true","false"}: self.advance(); return ASTNode("Bool",tok.value)
        if tok.value=="null": self.advance(); return ASTNode("Null","null")
        self.advance(); return ASTNode("Unknown",tok.value)

    # --- Pragmas ---
    def parse_pragma(self)->ASTNode:
        self.advance()
        name=self.advance().value
        return ASTNode("Pragma",name)

    # --- Switch/Case ---
    def parse_switch(self)->ASTNode:
        self.advance(); self.expect("(")
        expr=self.advance().value
        self.expect(")")
        self.expect("{")
        cases=[]
        while self.peek() and self.peek().value!="}":
            if self.peek().value=="case":
                self.advance(); val=self.advance().value; self.expect(":")
                cases.append(ASTNode("Case",val,[ASTNode("Stmt","...")]))
            elif self.peek().value=="default":
                self.advance(); self.expect(":")
                cases.append(ASTNode("Default","",[ASTNode("Stmt","...")]))
            else: self.advance()
        self.expect("}")
        return ASTNode("Switch",expr,cases)

    # --- Do/While ---
    def parse_do_while(self)->ASTNode:
        self.advance()
        body=self.parse_block()
        self.expect("while"); self.expect("(")
        cond=self.advance().value; self.expect(")")
        return ASTNode("DoWhile","",[body,ASTNode("Cond",cond)])

    def parse_block(self)->ASTNode:
        self.expect("{"); block=ASTNode("Block")
        while self.peek() and self.peek().value!="}":
            block.children.append(ASTNode("Stmt",self.advance().value))
        self.expect("}"); return block

    # --- Loop infinite ---
    def parse_loop(self)->ASTNode:
        self.advance()
        body=self.parse_block()
        return ASTNode("Loop","",[body])

    # --- Polynomial ---
    def parse_polynomial(self)->ASTNode:
        self.advance(); self.expect("(")
        expr=""
        while self.peek() and self.peek().value!=")":
            expr+=self.advance().value
        self.expect(")")
        return ASTNode("Polynomial",expr)

# --------------------------
# Obfuscation Engine
# --------------------------
def obfuscate_ast(ast:ASTNode,active=False):
    if ast.nodetype=="Pragma" and ast.value=="obfuscate":
        active=True
    if active and ast.nodetype in {"Var","Function","Assign"}:
        h=hashlib.sha1(ast.value.encode()).hexdigest()[:8]
        ast.value=f"obf_{h}"
    for c in ast.children:
        obfuscate_ast(c,active)

# --------------------------
# Polymorphism Engine
# --------------------------
def generalize_functions(ast:ASTNode):
    # Very simple demo: marks overloaded functions as "polymorphic"
    funcs=[c for c in ast.children if c.nodetype=="Function"]
    names={}
    for f in funcs:
        base=f.value.split("/")[0]
        names.setdefault(base,[]).append(f)
    for n,flist in names.items():
        if len(flist)>1:
            for f in flist: f.nodetype="PolyFunction"

# --------------------------
# Demo
# --------------------------
if __name__=="__main__":
    code="""
    #pragma obfuscate

    switch(x) {
        case 1: break;
        case 2: break;
        default: break;
    }

    do { print["hi"] } while(x < 10)

    loop { print["forever"] }

    poly(x^2 + 3x + 2)
    """
    toks=tokenize(code)
    print("TOKENS:",toks)

    parser=Parser(toks)
    ast=parser.parse_program()
    print("\nAST before obfuscation:\n",ast)

    obfuscate_ast(ast)
    generalize_functions(ast)

    print("\nAST after obfuscation + polymorphism:\n",ast)

# xyzc.py
# XYZ Compiler v9
# Adds: type inference, enum typing, polynomial typing
# NASM x86-64 codegen from AST
# Author: XYZ Project

import hashlib
from typing import List

# --------------------------
# AST
# --------------------------
class ASTNode:
    def __init__(self,nodetype:str,value:str="",children=None):
        self.nodetype=nodetype; self.value=value; self.children=children or []
    def __repr__(self,level=0):
        ind="  "*level; s=f"{ind}{self.nodetype}({self.value})\n"
        for c in self.children: s+=c.__repr__(level+1)
        return s

# --------------------------
# Type System
# --------------------------
class TypeEnv:
    def __init__(self):
        self.vars={}   # var -> type
        self.enums={}  # enum -> values
        self.funcs={}  # func -> return type
    def set_var(self,name,typ): self.vars[name]=typ
    def get_var(self,name): return self.vars.get(name,"unknown")
    def add_enum(self,name,vals): self.enums[name]=vals
    def get_enum(self,name): return self.enums.get(name,[])
    def set_func(self,name,ret): self.funcs[name]=ret
    def get_func(self,name): return self.funcs.get(name,"unknown")

def infer_type(node:ASTNode,env:TypeEnv):
    if node.nodetype=="Number":
        return "float" if "." in node.value else "int"
    if node.nodetype=="String": return "string"
    if node.nodetype=="Bool": return "bool"
    if node.nodetype=="Null": return "nulltype"
    if node.nodetype=="EnumVal":
        return node.value.split(".")[0]  # Colors.Red -> Colors
    if node.nodetype=="Polynomial": return "poly"
    if node.nodetype=="Assign":
        t=infer_type(node.children[0],env); env.set_var(node.value,t); return t
    if node.nodetype=="Var": return env.get_var(node.value)
    if node.nodetype=="BinaryOp":
        lt=infer_type(node.children[0],env); rt=infer_type(node.children[1],env)
        if lt!=rt: raise TypeError(f"Type mismatch: {lt} vs {rt}")
        return lt
    if node.nodetype=="If": return infer_type(node.children[1],env)
    if node.nodetype=="Call": return env.get_func(node.value,"int")
    return "unknown"

# --------------------------
# NASM Codegen
# --------------------------
class Codegen:
    def __init__(self):
        self.asm=[]
        self.label_counter=0
    def emit(self,line): self.asm.append(line)
    def new_label(self,base="L"): 
        self.label_counter+=1; return f"{base}{self.label_counter}"
    def gen(self,node:ASTNode):
        m=node.nodetype
        if m=="Number":
            self.emit(f"    mov rax,{node.value}")
        elif m=="BinaryOp":
            self.gen(node.children[0]); self.emit("    push rax")
            self.gen(node.children[1]); self.emit("    mov rbx,rax")
            self.emit("    pop rax")
            if node.value=="+": self.emit("    add rax,rbx")
            if node.value=="-": self.emit("    sub rax,rbx")
            if node.value=="*": self.emit("    imul rax,rbx")
            if node.value=="/": 
                self.emit("    cqo")
                self.emit("    idiv rbx")
        elif m=="Print":
            self.gen(node.children[0])
            self.emit("    ; print rax (placeholder)")
        elif m=="If":
            cond_lbl=self.new_label("IF")
            end_lbl=self.new_label("ENDIF")
            self.gen(node.children[0]) # condition
            self.emit("    cmp rax,0")
            self.emit(f"    je {cond_lbl}")
            self.gen(node.children[1]) # then
            self.emit(f"    jmp {end_lbl}")
            self.emit(f"{cond_lbl}:")
            if len(node.children)>2: self.gen(node.children[2]) # else
            self.emit(f"{end_lbl}:")
        elif m=="Loop":
            start=self.new_label("LOOP")
            self.emit(f"{start}:")
            for c in node.children[0].children: self.gen(c)
            self.emit(f"    jmp {start}")
        elif m=="DoWhile":
            start=self.new_label("DO")
            self.emit(f"{start}:")
            for c in node.children[0].children: self.gen(c)
            self.gen(node.children[1]) # condition
            self.emit("    cmp rax,0")
            self.emit(f"    jne {start}")
        elif m=="Switch":
            end=self.new_label("SWEND")
            for case in node.children:
                lbl=self.new_label("CASE")
                self.emit(f"; case {case.value}")
                self.emit(f"{lbl}:")
                for s in case.children: self.gen(s)
            self.emit(f"{end}:")
        elif m=="Polynomial":
            # naive expansion: assume x=rbx
            self.emit("; polynomial expansion")
            self.emit("    mov rax,rbx") # placeholder
        elif m=="Function":
            name=node.value
            self.emit(f"{name}:")
            for c in node.children: self.gen(c)
            self.emit("    ret")
        elif m=="Program":
            for c in node.children: self.gen(c)
        else:
            self.emit(f"; unhandled node {m}")
    def dump(self): return "\n".join(self.asm)

# --------------------------
# Demo
# --------------------------
if __name__=="__main__":
    # Mock AST to test NASM gen
    ast=ASTNode("Program","",[
        ASTNode("Function","main",[
            ASTNode("Block","",[
                ASTNode("Assign","x",[ASTNode("Number","5")]),
                ASTNode("Loop","",[ASTNode("Block","",[
                    ASTNode("Print","",[ASTNode("Var","x")])
                ])])
            ])
        ])
    ])

    env=TypeEnv()
    print("Inferred type for x:",infer_type(ast.children[0].children[0].children[0],env))

    cg=Codegen()
    cg.gen(ast)
    print("\n--- NASM Output ---\n")
    print(cg.dump())

class TypeEnv:
    def __init__(self):
        self.vars={}
        self.enums={}
        self.funcs={}
    def add_enum(self,name,values):
        self.enums[name]=values
    def is_enum_val(self,val):
        for enum,vals in self.enums.items():
            if val in [f"{enum}.{v}" for v in vals]:
                return enum
        return None

def infer_type(node,env:TypeEnv):
    if node.nodetype=="EnumVal":
        enum=env.is_enum_val(node.value)
        if not enum: raise TypeError(f"Unknown enum value {node.value}")
        return enum
    if node.nodetype=="Polynomial":
        return "poly"
    # other inference from v9 ...

class Codegen:
    def __init__(self):
        self.asm=[]
        self.label_counter=0
    def emit(self,line): self.asm.append(line)
    def new_label(self,base="L"): self.label_counter+=1; return f"{base}{self.label_counter}"

    def gen_print(self):
        # write syscall: write(1, rsi, rdx)
        self.emit("    mov rdi,1        ; fd=stdout")
        self.emit("    mov rsi,rsp      ; buf ptr (string pushed to stack)")
        self.emit("    mov rdx,rax      ; length in rax")
        self.emit("    mov rax,1        ; syscall write")
        self.emit("    syscall")

    def gen_polynomial(self,expr:str):
        # Simple Horner’s method expansion x^2+3x+2
        # Assume x in rbx, result in rax
        self.emit("    mov rax,rbx")
        self.emit("    imul rax,rbx")   # x^2
        self.emit("    mov rcx,rbx")
        self.emit("    imul rcx,3")
        self.emit("    add rax,rcx")
        self.emit("    add rax,2")

    def gen_factorial(self,name="Factorial"):
        self.emit(f"{name}:")
        self.emit("    push rbp")
        self.emit("    mov rbp,rsp")
        self.emit("    mov rax,[rbp+16]   ; arg n")
        self.emit("    cmp rax,0")
        end=self.new_label("FEND")
        self.emit(f"    je {end}")
        self.emit("    dec rax")
        self.emit("    push rax")
        self.emit("    call Factorial")
        self.emit("    mov rbx,[rbp+16]")
        self.emit("    imul rax,rbx")
        self.emit(f"{end}:")
        self.emit("    pop rbp")
        self.emit("    ret")

class Optimizer:
    def peephole(self,asm:list)->list:
        out=[]
        for i,line in enumerate(asm):
            if i>0 and "mov rax,rax" in line: continue
            out.append(line)
        return out

    def loop_unroll(self,ast:ASTNode)->ASTNode:
        # detect fixed loops like for(i=0;i<4;i++)
        # expand into repeated bodies
        return ast

    def const_fold(self,ast:ASTNode)->ASTNode:
        if ast.nodetype=="BinaryOp":
            if ast.children[0].nodetype=="Number" and ast.children[1].nodetype=="Number":
                l=int(ast.children[0].value); r=int(ast.children[1].value)
                val=l+r if ast.value=="+" else l-r
                return ASTNode("Number",str(val))
        return ast

class Peephole:
    def run(self, asm:list) -> list:
        out=[]
        for i,line in enumerate(asm):
            if line.strip()=="mov rax,rax": continue
            if i>0 and asm[i-1].startswith("push") and line.startswith("pop"):
                continue
            out.append(line)
        return out

class LoopUnroller:
    def run(self, ast):
        if ast.nodetype=="For":
            init,cond,step,body=ast.children
            if cond.value.startswith("i<4"):
                # expand 4 iterations
                unrolled=ASTNode("Block","")
                for _ in range(4): unrolled.children+=body.children
                return unrolled
        return ast

class ConstFolder:
    def fold(self,node):
        if node.nodetype=="BinaryOp":
            l=node.children[0]; r=node.children[1]
            if l.nodetype=="Number" and r.nodetype=="Number":
                lv=float(l.value); rv=float(r.value)
                if node.value=="+": return ASTNode("Number",str(lv+rv))
                if node.value=="*": return ASTNode("Number",str(lv*rv))
        return node

class Inliner:
    def run(self, call_node, func_defs):
        fn=func_defs.get(call_node.value)
        if fn and len(fn.children[1].children)<5:  # heuristic
            return fn.children[1]  # inline block
        return call_node

class CIAM:
    corrections = {"prnt":"print","pritn":"print"}
    macros = {"SAFEADD":"( (x)!=null && (y)!=null ? (x)+(y) : 0 )"}
    def expand(self,node):
        if node.nodetype=="Ident" and node.value in self.corrections:
            node.value=self.corrections[node.value]
        if node.nodetype=="MacroCall" and node.value=="SAFEADD":
            x,y=node.children
            return ASTNode("BinaryOp","+",
                [ASTNode("Var",x.value),ASTNode("Var",y.value)])
        return node

