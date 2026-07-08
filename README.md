# PuzzleHunter GPU (Public Key Edition)

**PuzzleHunter GPU** is a high-performance, compact OpenCL tool designed to search for compressed secp256k1 private keys by matching a target **compressed public key prefix**. It is optimized for AMD Radeon GPUs, but other OpenCL-capable devices may work as well.

The program generates candidate private keys inside a selected range (puzzle range or custom range), computes the corresponding compressed public key point directly on the GPU, and compares its prefix bytes with the target public key.

---

## ⚡ Key Optimizations & Performance Achievements

The GPU mathematical pipeline has been optimized, offering extremely high hashrates (exceeding **1.7 Gkeys/s** on an **AMD Radeon RX 6600 XT**):

1. **Custom Fast 256-bit Squaring (Elliptic Curve Arithmetic)**:
   - Implemented a dedicated `squareModP256k_internal` function in [secp256k1.cl](secp256k1.cl) (derived from the optimized HASH160 codebase).
   - By exploiting the symmetry of the cross-terms ($a_i \cdot a_j = a_j \cdot a_i$), the number of 32-bit multiplications is reduced from **64** to **36** (a **43%** reduction for squaring operations).
   - This optimization is applied across all modular squaring steps, including modular inversion.
2. **Optimized GPU Random Number Generator (xoshiro256+)**:
   - Replaced the more complex `xoshiro256**` with the lightweight `xoshiro256+` algorithm in [puzzle_hunter_kernel.cl](puzzle_hunter_kernel.cl).
   - This eliminates two heavy 64-bit multiplications and a 64-bit rotation per generated seed on the GPU, streamlining the initialization path for new search batches.
3. **Compiler Optimization Flags**:
   - Added compiler options `-cl-denorms-are-zero`, `-cl-no-signed-zeros`, and `-cl-strict-aliasing` for the OpenCL compiler to enable aggressive driver-level optimization.

---

## 📁 Project Layout

```text
.
|-- main_gpu.cpp              # Host C++ code (OpenCL setup, CLI, keyboard input, CPU verification)
|-- puzzle_hunter_kernel.cl   # Main OpenCL search kernels
|-- secp256k1.cl              # OpenCL secp256k1 math helper functions
|-- Int.* / Point.*           # CPU bigint/point math libraries (for verification)
|-- SECP256K1.*               # CPU GTable generation and candidate verification
|-- OpenCL/                   # Minimal OpenCL headers and import library for Windows MinGW
|-- Makefile                  # Build script
`-- hunter.exe                # Built executable (after compilation)
```

---

## 🛠 Requirements & Build

To compile the project, you need a C++ compiler supporting C++17 and the OpenCL SDK.

### Build on Windows (via MSYS2 / MinGW-w64):

1. Install gcc and make in the MSYS2 terminal:
   ```powershell
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
   ```
2. Make sure the paths to `g++` and `mingw32-make` are added to your system `PATH`.
3. Build the project:
   ```powershell
   mingw32-make clean
   mingw32-make
   ```

### Build on Linux (Ubuntu / Debian):

1. Install the required packages:
   ```bash
   sudo apt install build-essential ocl-icd-opencl-dev
   ```
2. Build the project:
   ```bash
   make clean
   make
   ```

---

## 🚀 Usage & Parameters

```powershell
.\hunter.exe -k <compressed_public_key_hex> [-p <puzzle> | -r <startHex:endHex>] [-b <prefix_bytes>] [-G <blocks>] [-t <threads>] [-n <points>] [-w <work_size>]
```

### Command Line Arguments:

- `-k <hex>`: Target compressed public key (exactly 66 hex characters starting with `02` or `03`).
- `-b <bytes>`: Number of prefix bytes to check (1..33, default: 4).
- `-p <bits>`: Puzzle range bit length. Search space: `[2^(bits-1), 2^bits - 1]`.
- `-r <start:end>`: Custom search range (`startHex:endHex`).
- `-G <blocks>`: Number of OpenCL blocks (default: 64).
- `-t <threads>`: Number of threads per block (default: 256).
- `-n <points>`: Number of points checked per thread per loop iteration (BATCH_SIZE) (default: 1024).
- `-w <size>`: OpenCL global work size (manual override for blocks \* threads).
- `--bench <n>`: Run `n` benchmark loops to test performance.
- `--profile <n>`: Time RNG, base multiply, and batch stages separately.
- `--selftest`: Compare known CPU/GPU points and run a full match test.

---

## 💡 Recommended Parameters for RX 6600 XT

For the **AMD Radeon RX 6600 XT**, you can use the following profiles to find the best balance between speed, stability, and system responsiveness:

### 1. Balanced Profile 🌿

- **Parameters:** `-k <KEY> -p <BIT> -G 512 -t 256 -n 1024`
- **Speed:** High
- **Pros:** Safe hotspot temperatures, very high system responsiveness.

### 2. Max Performance Profile ⚡ (Recommended)

- **Parameters:** `-k <KEY> -p <BIT> -G 384 -t 256 -n 1024` or `-G 768 -t 256 -n 1024`
- **Speed:** Very High (~1.7+ Gkeys/s)
- **Pros:** Optimal occupancy and throughput.

### 3. Aggressive Profile 🔥

- **Parameters:** `-k <KEY> -p <BIT> -G 1024 -t 256 -n 2048`
- **Speed:** Maximum
- **Cons:** May cause slight GUI lag. Reduce `-n` or `-G` if you experience driver resets (TDR errors).

---

## 📈 Examples

1. **Run a benchmark to test GPU hashrate:**

   ```powershell
   .\hunter.exe -k 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d -p 40 -b 4 -G 384 -t 256 -n 1024 --bench 5
   ```

2. **Run self-test mode:**

   ```powershell
   .\hunter.exe -k 02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9 -p 2 -b 33 -G 384 -t 256 -n 1024 --selftest
   ```

3. **Search for a full match inside Puzzle 35 (first 4 bytes checked on GPU):**
   ```powershell
   .\hunter.exe -k 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d -p 35 -b 4 -G 384 -t 256 -n 1024
   ```

### Example Search Output (Puzzle 35, partial and full match):

```text
=== PuzzleHunter OpenCL GPU v1.0 ===
Preparing CPU : GTable
Platform      : AMD Accelerated Parallel Processing
Device        : AMD Radeon RX 6600 XT
Device type   : GPU
Compute units : 16
Max clock     : 2428 MHz
Global memory : 8176 MB
Target        : 02f6a8148a62320e...feada13d
Prefix bytes  : 5
Puzzle        : 35
Range         : 400000000:7FFFFFFFF
Max Byte Count: 5
Blocks        : 384
Threads       : 256
Points/Thread : 1024
Checked/loop  : 201326592
Search started. Press [SPACE] to pause, [ESC] to exit.
Time          : 00:01 | Speed: 1885.99 Mkeys/s | Total: 43285217280
================== PARTIAL MATCH FOUND! ============
Prefix bytes  : 4
Private Key   : 00000000000000000000000000000000000000000000000000000004ddb7e1bc
Found PubKey  : 02f6a8149c07e9a1a7f3c7e979916a52f2c57228403a50ad150cf68afd8daea077
Target PubKey : 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d

================== FOUND MATCH! ====================
Match type    : FULL
Private Key   : 00000000000000000000000000000000000000000000000000000004aed21170
Public Key    : 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d
```

---

## 💾 Output File

Upon finding a full match, the program saves the results in a file named `KEYFOUND.txt` in the current directory. The file contains:

- Private Key
- Public Key
- Target Public Key
- Total checked count
- Elapsed time
- Average search speed

---

## Thanks

If this project helped you and you want to say thanks:

```text
BTC: bc1qa3c5xdc6a3n2l3w0sq3vysustczpmlvhdwr8vc
```
