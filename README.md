# 🌌 The XYZ Programming Language

*"Item-Oriented. Hex-Bodied. Self-Optimized."*

---

🧠 Language Features Supported
✅ Item-Oriented Constructs

Functions, variables, assignments, calls, returns

Lists, maps, indexing with bounds/type checking

Enums, pragmas, lambdas, and dotted method/field access

✅ Control Flow

If/else, while, for loops

Try/catch, throw, isolate, force, remove

Parallel blocks with thread pool execution

✅ Memory & Mutex Primitives

alloc, free, mutex, mutex_lock, mutex_unlock

✅ Built-in Print and Eval

print for output

eval("expression") for runtime evaluation

🔧 Compiler Pipeline
1. Lexer & Parser

Tokenizes and parses XYZ source into AST

Supports extended grammar including containers and control structures

2. Codegen

Emits NASM x64 assembly directly

Auto-links known syscalls and libc functions (e.g. printf, malloc)

Packs output into dodecagram binary stream

3. Object Emitter & Linker

Emits object files as JSON symbol maps

Links multiple object files into final .asm with externs and trace comments

4. Runtime Execution

MiniRuntime: interpreted execution with stack frames and closures

FastVM: bytecode compiler + optimized VM with hot-path inlining and memory pooling

5. Hot-Swap System

IPC server on localhost:4000

Accepts JSON payloads to swap function bodies at runtime

Supports live patching and demo mode

🚀 Execution Flow
You can now run:

bash
python xyzc.py mycode.xyz --emit-asm --hot-swap-server
And it will:

Compile mycode.xyz to NASM

Emit out.asm and optionally out.pkt

Start a hot-swap server for live updates

Execute main/0 via MiniRuntime or FastVM

🧪 Bonus Capabilities
Parallel execution of tasks via thread pool

Constant folding optimizer

Dodecagram packing for binary stream output

FastVM fallback to interpreter for unhandled opcodes

Inline caching and hot-path optimization

---

## 🔹 Core Identity

XYZ is a **next-generation systems and application language** designed for **direct NASM mapping** with **Ahead-of-Time (AOT) compilation**. It eliminates external toolchains by being **self-contained** and **self-optimizing**, allowing developers to focus entirely on *what* to build rather than *how* to manage toolchains, linking, or optimization flags.

---

## 🔹 Defining Features

### 1. **Item-Oriented Programming (IOP)**

* Objects are replaced with **items**—lightweight, native-first entities that avoid OOP overhead.
* Items unify the power of structs, classes, and modules while allowing inheritance, polymorphism, and directives.
* Everything in XYZ is an item: functions, bundles, packets, even memory allocations.

### 2. **Direct NASM Mapping**

* Source compiles *straight into NASM x64*.
* IR is written directly in **hexadecimal form**, making binaries transparent and audit-ready.
* **Dodecagram AST (base-12 model)** ensures compact, lossless AST storage: digits `0–9, a, b`.

### 3. **Framed Hot Routes & Hot Swaps**

* Hot sections of code are auto-identified and given **framed hot routes** for efficiency.
* Live **hot swaps** replace sections during execution without downtime.

### 4. **Self-Optimizing by Design**

* No flags, no config. The compiler detects usage patterns, hardware, and user tendencies, then builds tuned libraries automatically.
* Code deployment is **pattern-matched**—scheduling and parallelism are inferred without developer micromanagement.

### 5. **Error & Memory Models**

* **Errors:** `try`, `catch`, `throw`, `except`, `isolate`, `delete`.
* **Memory:** `collect`, `store`, `flush`, `sweep`.
* Fully **register-based stack/heap management**, blending low-level control with high-level safety.

### 6. **Universal Grammar**

* One syntax for **system programming** *and* **scripting**.
* Developers can write drivers, kernels, or one-liners with the same concise grammar.

---

## 🔹 Ecosystem & Interop

* **Automatic Linking:** All imports/exports are resolved at compile time—no manual linker steps.
* **C Interop:** Native bidirectional binding with **C** (ABI, ISA, FFI, calls, wraps).
* **Deployment:** Built-in deployment system abstracts containers, services, and distribution.

---

## 🔹 Example Syntax

```xyz
; Single-line comment
-- Multiline
-- comment block

Start() print {"Hello, world"} run

main() print ["Hello, world"] start
```

* Curly `{}` or square `[]` are interchangeable for literals.
* Minimalistic keywords: `start`, `run`, `print`.

---

## 🔹 Advantages Over Other Languages

* **Faster than C** through:

  * NASM-level mapping
  * Register-based optimization
  * Eliminating undefined behavior entirely
* **Safer than Rust** by embedding error/memory models directly into grammar.
* **Simpler than Go/Python** with a single universal grammar and no external runtime.

---

## 🔹 Use Cases

* **Systems Programming:** kernels, OS components, embedded hardware.
* **High-Performance Apps:** game engines, rendering pipelines, real-time physics.
* **Enterprise:** deployment-ready server applications with zero external tooling.
* **Research/Math:** polynomials, derivatives, decimals, negative numbers fully baked-in.

---

## 🔹 Formal Tagline

**XYZ: The Self-Optimizing Item-Oriented Language — Faster Than C, Simpler Than Scripting.**

---

run .cl file

nvcc -arch=sm_75 -ptx force.cu -o force.ptx

## currently


---

🧠 Architectural Highlights

✅ Full Pipeline Execution

• Lex → Parse → Typecheck → IR → Codegen → Runtime all in one flow.
• Supports fallback tiers: `ProCompiler`, `FastRuntime`, `MiniRuntime`.


✅ Hot-Swap & Mutation

• `HotSwapRegistry` enables live function replacement.
• `SelfExpander` can synthesize new functions and register them dynamically.


✅ Macro & Type System

• `MacroEngine` supports compile-time expansion.
• `TypeRegistry` and `StructRegistry` allow rich type modeling, including vectors, matrices, and boxed structs.


✅ Multi-Runtime Strategy

• `MiniRuntime`: stack-frame emulation, closures, mutexes.
• `FastRuntime`: bytecode VM with inline caching, memory pooling, and hot-path optimization.
• `AdvancedEngine`: SSA optimizer, type inference, JIT hooks, profiler integration.


---

⚙️ Codegen & Linking

• Emits annotated NASM x64 assembly.
• Includes a dodecagram binary packer and object file emitter.
• Can auto-link multiple object files and emit final assembly with syscall and FFI mapping.


---

🔁 Live System Features

• Rapid Checkout: snapshot and restore of symbol tables and hot-swap states.
• HotSwapServer: JSON-based IPC server for live mutation.
• Macro Synthesis: vector ops, symbolic dispatch, and runtime fusion.


---

🧪 Advanced Capabilities

• JITCompiler: compiles hot functions to native Python callables.
• SSAOptimizer: constant folding, dead code elimination.
• TypeInfer: symbolic type inference over IR.
• FastVM: inline caching, parallel execution, fallback interpreter.


---

🧬 What This Enables

• A symbolic runtime that can mutate, optimize, and dispatch behavior dynamically.
• A compiler that feels like a living system—adaptive, introspective, and expressive.
• A language that can rival C++ in performance while remaining lean and mutation-friendly.


---

🧬 Examples of Item-Oriented Programming (IOP)
Here’s how IOP manifests in XYZ:

xyz
Point2D.new(3, 4)
Rect.add(a, b)
list_add([1,2], [3,4])
These aren’t method calls—they’re item invocations. Each item is:

Autogen’d via SelfExpander

Registered in HotSwapRegistry

Executed via MiniRuntime or FastVM

🔹 Behind the Scenes:
Point2D.new creates a vector item with fields x and y

Rect.add invokes list_add on two structured items

Items are mutable, composable, and hot-swappable

You can even swap Rect.add/2 mid-execution to change behavior without restarting the daemon.

🛠️ Practical Applications of XYZ
XYZ isn’t just a compiler—it’s a backend ritual engine. Here’s where it shines:

🔧 Embedded Systems
Direct NASM mapping for firmware

Blessing printers, inventory scanners, GPS modules

🧮 Simulation Engines
Branching state, time-travel ledgers

Wrestling matchups, communal overlays

🛍️ POS Daemons & Scrollkeepers
Struct registry for inventory

list_add for receipt glyphs

Hot-swap for live ritual updates

🧠 DSL Prototyping
Build your own language with ProCompiler

Emit bytecode, NASM, or interpret AST

🚀 How IOP Improves Performance
Item-Oriented Programming is lean, direct, and expressive:

🔹 Performance Gains:
No OOP overhead: No vtables, no inheritance chains

Direct NASM mapping: Items compile to registers and syscalls

Hot-swap execution: No restart needed for logic updates

Self-optimizing compiler: Adapts to usage patterns and hardware

🔹 Architectural Benefits:
Items unify structs, modules, and functions

One grammar for scripting, systems, and deployment

Mega features like autogen vector ops and type registries

## _____

🔗 Interoperability Mechanisms in XYZ
1. Syscall-Level Codegen
XYZ’s Codegen module emits raw x86 NASM with direct syscall mapping (read, write, exit, etc.).

This allows XYZ-compiled binaries to interoperate with C, Rust, or Zig via shared memory, pipes, or syscall orchestration.

2. Object Linking via JSON Scrolls
XYZ serializes compiled functions into JSON-based object files (XYZOBJ1 format).

These can be linked into unified .text sections, enabling modular integration with other compilers or loaders.

3. HotSwapRegistry as a Ritual FFI
Functions are registered by name/arity (name/2) and can be swapped at runtime.

External systems (e.g., Python, Node.js) can inject new logic into XYZ’s runtime by emitting compatible AST or bytecode.

4. Mega Builtins and Global Injection
Builtins like list_add are injected into both MiniRuntime and FastVM via enable_mega_features.

You can inject Python or C functions into XYZ’s runtime by registering them in vm.globals.

5. AST-Level Interop
XYZ’s AST is Python-native (ASTNode, FuncDef, Call, etc.), making it easy to generate or manipulate from other Python-based DSLs or compilers.

You can write a Bonus or Instryx-to-XYZ transpiler by emitting AST directly.

🧪 Practical Interop Scenarios
Scenario	Interop Pathway
Call C functions from XYZ	Emit NASM with syscall or link .o files
Inject Python logic into XYZ	Register Python functions in vm.globals
Transpile Bonus to XYZ	Emit XYZ AST from Bonus parser
Use XYZ in Rust/Zig project	Compile to .o and link via system linker
Live-edit XYZ from Node.js	Emit AST or bytecode and hot-swap via registry
🧬 Mythic Integration Possibilities
Lettera overlays: Use XYZ to compile spatial glyphs and inject them into a Rust-based physics engine.

Salem daemons: Wrap XYZ’s scrollkeeper logic into a Python POS system using JSON object linking.

Astronomy rituals: Use XYZ to compile telescope control logic and link it with C++ drivers.

## _____

🧩 Modular Pipelines
The file encodes four distinct compilation and runtime pipelines:

ProCompiler:

ProLexer → ProParser → TypeChecker → IRBuilder → ExecutionEngine

Feels like the high priest of the ceremony—structured, typed, and IR-driven.

Legacy Compiler:

Parser → Optimizer → Codegen → Assembly/Object/Linker

A throwback ritual, still honored for its rawness and directness.

FastRuntime:

AST → Bytecode → FastVM

A bytecode priesthood with its own opcodes and stack-based VM.

MiniRuntime:

AST interpreter with hot-swap and mega features

The most communal and mutable—perfect for live rituals and scrollkeeping.

🧙‍♂️ Ceremonial Constructs
TypeRegistry & StructRegistry:

Registers types like Int32, Float64, Complex, and structs like Point2D, Rect.

These are your ceremonial glyphs—defining the shape of the ritual space.

ASTNode Hierarchy:

Rich symbolic language: FuncDef, Call, Return, BinOp, Parallel, TryCatch, Throw, Enum, Force, Remove.

Each node is a ritual gesture, encoded for interpretation or compilation.

HotSwapRegistry:

Enables live function replacement keyed by name/arity.

Think of it as scrollkeepers rewriting incantations mid-ceremony.

MacroEngine & SelfExpander:

Lightweight macro expansion and autogen for vector ops.

Symbolic expansion of glyphs into living code.

🧠 Runtime Engines
MiniRuntime:

Stack of frames, closure support, mega builtins like list_add.

Interprets AST with emotional resonance—perfect for communal onboarding.

FastVM:

Bytecode execution with opcodes like ADD, CALL, RET, POW.

Stack-based, depth-limited, and supports global rituals like print.

🛠️ Codegen & Linking
Codegen:

Emits x86-like assembly with syscall mapping (read, write, exit).

Includes safe division, power loops, and conditional jumps.

Object Writer & Linker:

Serializes symbols into JSON object files and links them into a unified .text section.

Like binding scrolls into a single ceremonial tome.

🔮 Grand Resolution
The XYZ_GRAND_EXECUTE=1 mode attempts all pipelines in order:

ProCompiler

FastRuntime

MiniRuntime

If XYZ_GRAND_STRICT=1 and all fail, it raises a GrandResolutionError with diagnostics—a final judgment from the compiler oracle.

## _____

✅ Correct Usage Examples
Run a program directly:

bash
python xyzc.py hello.xyz --run
Emit NASM assembly:

bash
python xyzc.py hello.xyz --emit-asm -o hello.asm
Emit object file:

bash
python xyzc.py hello.xyz --emit-obj -o hello.obj
Link multiple object files:

bash
python xyzc.py foo.obj bar.obj --link -o final.asm
Enable Mega features (structs, list_add, etc.):

bash
python xyzc.py vectors.xyz --run --mega
🧠 Why It Matters
The source argument is required because xyzc needs at least one .xyz file to parse, compile, or run. Without it, the compiler doesn’t know what ceremony to perform.

If you’re testing, you can start with a minimal file like:

xyz
func main() {
  print(42)
  return 0
}
Save that as hello.xyz and try one of the commands above.

🧪 Example .xyz
func main() {
    msg = "Hello, world!"
    print(msg)
    print(1234)
    return 0
}

Output after compiling + running:
Hello, world!
1234


✅ Now print supports both strings and integers.

## _____

