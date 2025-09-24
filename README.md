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
