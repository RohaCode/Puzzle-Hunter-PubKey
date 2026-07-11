#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#endif
#include <thread>

#include <CL/cl.h>
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif

#include "Int.h"
#include "Point.h"
#include "SECP256K1.h"
#include <memory>

#pragma pack(push, 1)
typedef struct { unsigned int v[8]; } uint256_gpu_t;
typedef struct { uint256_gpu_t x, y; } point_affine_gpu_t;
typedef struct { uint64_t s0, s1, s2, s3; } xoshiro_state_gpu_t;
typedef struct {
    unsigned int flag;
    uint256_gpu_t privKey;
    uint256_gpu_t x;
    uint256_gpu_t y;
} puzzle_hunter_result_gpu_t;
#pragma pack(pop)

#ifndef _WIN32
static struct termios orig_termios;
static bool termios_initialized = false;

static void disableRawMode() {
    if (termios_initialized) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    }
}

static void enableRawMode() {
    if (tcgetattr(STDIN_FILENO, &orig_termios) == -1) return;
    termios_initialized = true;
    atexit(disableRawMode);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

static int _kbhit() {
    if (!termios_initialized) {
        enableRawMode();
    }
    struct timeval tv = { 0, 0 };
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

static int _getch() {
    if (!termios_initialized) {
        enableRawMode();
    }
    char ch = 0;
    if (read(STDIN_FILENO, &ch, 1) < 0) return 0;
    return ch;
}
#endif

static uint64_t splitmix64_next(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void checkCl(cl_int err, const char* what) {
    if(err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with OpenCL error " << err;
        throw std::runtime_error(oss.str());
    }
}

static std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if(!file) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static std::string getDeviceInfoString(cl_device_id device, cl_device_info param) {
    size_t size = 0;
    if(clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "";
    }
    std::vector<char> value(size + 1, 0);
    if(clGetDeviceInfo(device, param, size, value.data(), nullptr) != CL_SUCCESS) {
        return "";
    }
    return std::string(value.data());
}

static std::string getPlatformInfoString(cl_platform_id platform, cl_platform_info param) {
    size_t size = 0;
    if(clGetPlatformInfo(platform, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "";
    }
    std::vector<char> value(size + 1, 0);
    if(clGetPlatformInfo(platform, param, size, value.data(), nullptr) != CL_SUCCESS) {
        return "";
    }
    return std::string(value.data());
}

static std::string getDisplayDeviceName(cl_device_id device) {
    const cl_device_info CL_DEVICE_BOARD_NAME_AMD_LOCAL = 0x4038;
    std::string openclName = getDeviceInfoString(device, CL_DEVICE_NAME);
    std::string boardName = getDeviceInfoString(device, CL_DEVICE_BOARD_NAME_AMD_LOCAL);
    if(!boardName.empty()) return boardName;
    return openclName.empty() ? "Unknown OpenCL GPU" : openclName;
}

static std::string deviceTypeToString(cl_device_type type) {
    if(type & CL_DEVICE_TYPE_GPU) return "GPU";
    if(type & CL_DEVICE_TYPE_CPU) return "CPU";
    if(type & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    return "UNKNOWN";
}

static cl_ulong getDeviceInfoUlong(cl_device_id device, cl_device_info param) {
    cl_ulong value = 0;
    clGetDeviceInfo(device, param, sizeof(value), &value, nullptr);
    return value;
}

static cl_uint getDeviceInfoUint(cl_device_id device, cl_device_info param) {
    cl_uint value = 0;
    clGetDeviceInfo(device, param, sizeof(value), &value, nullptr);
    return value;
}

static void printDeviceSummary(cl_platform_id platform, cl_device_id device) {
    cl_device_type type = 0;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
    std::cout << "Platform      : " << getPlatformInfoString(platform, CL_PLATFORM_NAME) << "\n";
    std::cout << "Device        : " << getDisplayDeviceName(device) << "\n";
    std::cout << "Device type   : " << deviceTypeToString(type) << "\n";
    std::cout << "Compute units : " << getDeviceInfoUint(device, CL_DEVICE_MAX_COMPUTE_UNITS) << "\n";
    std::cout << "Max clock     : " << getDeviceInfoUint(device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz\n";
    std::cout << "Global memory : " << (getDeviceInfoUlong(device, CL_DEVICE_GLOBAL_MEM_SIZE) / (1024 * 1024)) << " MB\n" << std::flush;
}

static Int hexToInt(const std::string& hex) {
    Int number;
    char buf[65] = {0};
    std::string padded = hex.size() >= 64 ? hex.substr(hex.size() - 64) : std::string(64 - hex.size(), '0') + hex;
    std::strncpy(buf, padded.c_str(), 64);
    number.SetBase16(buf);
    return number;
}

static std::string intToHex(const Int& value) {
    Int temp;
    temp.Set((Int*)&value);
    return temp.GetBase16();
}

static std::string compactHex(const Int& value) {
    std::string hex = intToHex(value);
    size_t first = hex.find_first_not_of('0');
    if(first == std::string::npos) return "0";
    return hex.substr(first);
}

static std::string int128ToString(unsigned __int128 n) {
    if (n == 0) return "0";
    std::string s = "";
    while (n > 0) {
        s += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

static unsigned __int128 stringToInt128(const std::string& s) {
    unsigned __int128 val = 0;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            val = val * 10 + (c - '0');
        }
    }
    return val;
}

static void intToGpu(const Int& src, uint256_gpu_t& dst) {
    for(int i = 0; i < 8; i++) {
        dst.v[i] = ((Int&)src).bits[7 - i];
    }
}

static std::string gpuToHex(const uint256_gpu_t& value) {
    std::ostringstream oss;
    for(int i = 0; i < 8; i++) {
        oss << std::hex << std::setw(8) << std::setfill('0') << value.v[i];
    }
    return oss.str();
}

static std::string pointToCompressedHex(const uint256_gpu_t& x, const uint256_gpu_t& y) {
    return std::string((y.v[7] & 1U) ? "03" : "02") + gpuToHex(x);
}

static std::string pointToCompressedHex(const Point& p) {
    uint256_gpu_t x;
    uint256_gpu_t y;
    intToGpu(p.x, x);
    intToGpu(p.y, y);
    return pointToCompressedHex(x, y);
}

static bool intLess(const Int& a, const Int& b) {
    Int aa;
    Int bb;
    aa.Set((Int*)&a);
    bb.Set((Int*)&b);
    return aa.IsLower(&bb);
}

static void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " -k <public_key_hex> [-p <puzzle> | -r <startHex:endHex>] [-b <prefix_len>] [-G <blocks>] [-t <threads>] [-n <points>] [-w <work_size>]\n";
    std::cerr << "  -k : Target compressed public key (66 hex chars)\n";
    std::cerr << "  -b : Number of prefix bytes to compare (1-33) (default: 4)\n";
    std::cerr << "  -G : Number of OpenCL blocks (default: 64)\n";
    std::cerr << "  -t : Number of threads per block (default: 256)\n";
    std::cerr << "  -n : Points per thread (BATCH_SIZE) (default: 1024)\n";
    std::cerr << "  -p : Puzzle size, searches [2^(p-1), 2^p - 1]\n";
    std::cerr << "  -r : Custom range startHex:endHex\n";
    std::cerr << "  -w : OpenCL global work size (blocks * threads)\n";
    std::cerr << "  --bench <loops> : Run fixed kernel loops and print speed without stopping on match\n";
}

static std::string formatElapsed(double elapsedSeconds) {
    uint64_t totalSeconds = (uint64_t)elapsedSeconds;
    uint64_t hours = totalSeconds / 3600;
    uint64_t minutes = (totalSeconds / 60) % 60;
    uint64_t seconds = totalSeconds % 60;
    std::ostringstream oss;
    oss << std::setfill('0');
    if(hours > 0) oss << hours << ':';
    oss << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
    return oss.str();
}

static void printProgressLine(double elapsedSeconds, double speedMkeys, unsigned __int128 totalChecked) {
    std::ostringstream oss;
    oss << "Time          : " << formatElapsed(elapsedSeconds)
        << " | Speed: " << std::fixed << std::setprecision(2)
        << speedMkeys << " Mkeys/s | Total: " << int128ToString(totalChecked);
    std::cout << '\r' << oss.str() << std::string(24, ' ') << std::flush;
}

static int getConsoleCursorY() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) return (int)info.dwCursorPosition.Y;
#endif
    return -1;
}

static void setConsoleCursorY(int y) {
#ifdef _WIN32
    if(y >= 0) {
        HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO info;
        if(GetConsoleScreenBufferInfo(out, &info)) {
            COORD pos = info.dwCursorPosition;
            pos.X = 0;
            pos.Y = (SHORT)y;
            SetConsoleCursorPosition(out, pos);
        }
    }
#endif
}

static void clearConsoleLine() {
    std::cout << '\r' << std::string(120, ' ') << '\r';
}

static void printPartialBlock(int prefixLen, const std::string& privHex, const std::string& foundPubHex, const std::string& targetPubHex) {
    std::cout << "================== PARTIAL MATCH FOUND! ============\n";
    std::cout << "Prefix bytes  : " << prefixLen << "\n";
    std::cout << "Private Key   : " << privHex << "\n";
    std::cout << "Found PubKey  : " << foundPubHex << "\n";
    std::cout << "Target PubKey : " << targetPubHex << "\n";
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== PuzzleHunter OpenCL GPU v1.0 ===\n" << std::flush;
        std::string targetPubHex, rangeArg;
        int prefixLen = 4, puzzleBits = 0, benchLoops = 0, profileLoops = 0;
        bool puzzleProvided = false, rangeProvided = false, selfTest = false, workSizeProvided = false;
        int blocks = 64, threads = 256, pointsPerThread = 1024;
        size_t globalWorkSize = 0;

        for(int i = 1; i < argc; i++) {
            if(!std::strcmp(argv[i], "-k") && i + 1 < argc) targetPubHex = argv[++i];
            else if(!std::strcmp(argv[i], "-b") && i + 1 < argc) prefixLen = std::stoi(argv[++i]);
            else if(!std::strcmp(argv[i], "-G") && i + 1 < argc) blocks = std::stoi(argv[++i]);
            else if(!std::strcmp(argv[i], "-t") && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if(!std::strcmp(argv[i], "-n") && i + 1 < argc) pointsPerThread = std::stoi(argv[++i]);
            else if(!std::strcmp(argv[i], "-p") && i + 1 < argc) { puzzleBits = std::stoi(argv[++i]); puzzleProvided = true; }
            else if(!std::strcmp(argv[i], "-r") && i + 1 < argc) { rangeArg = argv[++i]; rangeProvided = true; }
            else if(!std::strcmp(argv[i], "-w") && i + 1 < argc) { globalWorkSize = (size_t)std::stoull(argv[++i]); workSizeProvided = true; }
            else if(!std::strcmp(argv[i], "--selftest")) selfTest = true;
            else if(!std::strcmp(argv[i], "--bench") && i + 1 < argc) benchLoops = std::stoi(argv[++i]);
            else if(!std::strcmp(argv[i], "--profile") && i + 1 < argc) profileLoops = std::stoi(argv[++i]);
            else { std::cerr << "Unknown parameter: " << argv[i] << "\n"; printUsage(argv[0]); return 1; }
        }

        if(targetPubHex.size() != 66 || prefixLen <= 0 || prefixLen > 33 || (!puzzleProvided && !rangeProvided)) {
            printUsage(argv[0]); return 1;
        }

        Int minKey, maxKey;
        if(puzzleProvided) {
            if(puzzleBits <= 0 || puzzleBits > 256) throw std::runtime_error("Invalid puzzle value.");
            Int one((uint64_t)1);
            minKey.Set(&one); minKey.ShiftL(puzzleBits - 1);
            maxKey.Set(&one); maxKey.ShiftL(puzzleBits); maxKey.Sub(&one);
        } else {
            const size_t colon = rangeArg.find(':');
            if(colon == std::string::npos) throw std::runtime_error("Invalid range format.");
            minKey = hexToInt(rangeArg.substr(0, colon));
            maxKey = hexToInt(rangeArg.substr(colon + 1));
        }

        if(!intLess(minKey, maxKey)) throw std::runtime_error("Range start must be less than range end.");

        Int rangeSize; rangeSize.Sub(&maxKey, &minKey);
        Int twoBatchSize((uint64_t)(2 * pointsPerThread));
        if (rangeSize.IsLower(&twoBatchSize)) throw std::runtime_error("Range size is too small.");

        Int minKeyRestricted; minKeyRestricted.Set(&minKey);
        Int batchSizeInt((uint64_t)pointsPerThread); minKeyRestricted.Add(&batchSizeInt);
        Int rangeLimitRestricted; rangeLimitRestricted.Sub(&rangeSize, &twoBatchSize);

        Int rangeMask;
        int bitLen = rangeLimitRestricted.GetBitLength();
        if (bitLen == 0) rangeMask.SetInt32(0);
        else if (bitLen >= 256) { for(int i=0; i<8; i++) rangeMask.bits[i] = 0xFFFFFFFF; }
        else { Int one((uint64_t)1); rangeMask.Set(&one); rangeMask.ShiftL(bitLen); rangeMask.Sub(&one); }

        uint256_gpu_t gpuMinKey, gpuMaxKey, gpuRangeMask, gpuRangeLimit;
        intToGpu(minKeyRestricted, gpuMinKey); intToGpu(maxKey, gpuMaxKey);
        intToGpu(rangeMask, gpuRangeMask); intToGpu(rangeLimitRestricted, gpuRangeLimit);

        std::vector<unsigned char> targetPrefix(33, 0);
        for(int i = 0; i < 33; i++) targetPrefix[i] = (unsigned char)std::stoul(targetPubHex.substr(i * 2, 2), nullptr, 16);

        int maxBytes = (maxKey.GetBitLength() + 7) / 8;
        if(maxBytes < 1) maxBytes = 1;
        if(maxBytes > 32) maxBytes = 32;

        std::cout << "Preparing CPU : GTable (" << maxBytes << " bytes)\n" << std::flush;
        auto secp = std::make_unique<Secp256K1>(); secp->Init();
        std::vector<point_affine_gpu_t> gpuGTable(maxBytes * 256);
        for(int i = 0; i < maxBytes * 256; i++) { intToGpu(secp->GTable[i].x, gpuGTable[i].x); intToGpu(secp->GTable[i].y, gpuGTable[i].y); }

        const size_t batchWidth = 1024;
        std::vector<point_affine_gpu_t> gpuBatchTable(batchWidth);
        for(size_t i = 0; i < batchWidth; i++) {
            Int idx((uint64_t)(i + 1)); Point nextP = secp->ComputePublicKey(&idx);
            intToGpu(nextP.x, gpuBatchTable[i].x); intToGpu(nextP.y, gpuBatchTable[i].y);
        }

        cl_int err = CL_SUCCESS; cl_platform_id platform = nullptr; cl_device_id device = nullptr;
        checkCl(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
        checkCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");
        printDeviceSummary(platform, device);

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); checkCl(err, "clCreateContext");
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err); checkCl(err, "clCreateCommandQueue");
        std::string source = readKernelSource("puzzle_hunter_kernel.cl");
        const char* srcPtr = source.c_str(); size_t srcLen = source.size();
        cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err); checkCl(err, "clCreateProgramWithSource");

        std::string buildOpts = "-I . -cl-mad-enable -cl-fast-relaxed-math -cl-denorms-are-zero -cl-no-signed-zeros -cl-strict-aliasing";
        buildOpts += " -D BATCH_SIZE=" + std::to_string(pointsPerThread);
        buildOpts += " -D LOCAL_SIZE=" + std::to_string(threads);
        buildOpts += " -D MAX_BYTE_COUNT=" + std::to_string(maxBytes);

        err = clBuildProgram(program, 1, &device, buildOpts.c_str(), nullptr, nullptr);
        if(err != CL_SUCCESS) {
            size_t logSize = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize + 1, 0); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "Build Log:\n" << log.data() << "\n"; return 1;
        }

        cl_kernel startKernel = clCreateKernel(program, "puzzle_hunter_search_gpu", &err);
        cl_kernel threadBatchKernel = clCreateKernel(program, "puzzle_hunter_check_batch_thread_gpu", &err);

        if(!workSizeProvided) globalWorkSize = (size_t)blocks * threads;
        if(globalWorkSize < (size_t)threads) globalWorkSize = (size_t)threads;
        size_t localWorkSize = (size_t)threads;
        if(globalWorkSize % localWorkSize != 0) globalWorkSize = ((globalWorkSize + localWorkSize - 1) / localWorkSize) * localWorkSize;

        const size_t groupsPerLaunch = globalWorkSize;
        const size_t checkedPerGroup = (size_t)pointsPerThread * 2;
        std::vector<xoshiro_state_gpu_t> rngStates(groupsPerLaunch);

        auto now_entropy = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        uint64_t seedBase = (uint64_t)now_entropy ^ (uint64_t)GetCurrentProcessId() ^ __rdtsc();
        seedBase = (seedBase ^ (seedBase >> 30)) * 0xbf58476d1ce4e5b9ULL;
        seedBase = (seedBase ^ (seedBase >> 27)) * 0x94d049bb133111ebULL;
        seedBase = seedBase ^ (seedBase >> 31);

        for(size_t i = 0; i < groupsPerLaunch; i++) {
            uint64_t ts = seedBase ^ (uint64_t)i;
            ts = (ts ^ (ts >> 30)) * 0xbf58476d1ce4e5b9ULL;
            rngStates[i].s0 = splitmix64_next(ts); rngStates[i].s1 = splitmix64_next(ts);
            rngStates[i].s2 = splitmix64_next(ts); rngStates[i].s3 = splitmix64_next(ts);
        }

        puzzle_hunter_result_gpu_t zeroResult = {};
        cl_mem bufRngStates = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(xoshiro_state_gpu_t) * rngStates.size(), rngStates.data(), &err);
        cl_mem bufGTable = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(point_affine_gpu_t) * gpuGTable.size(), gpuGTable.data(), &err);
        cl_mem bufBatchTable = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(point_affine_gpu_t) * gpuBatchTable.size(), gpuBatchTable.data(), &err);
        cl_mem bufPrefix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, targetPrefix.size(), targetPrefix.data(), &err);
        cl_mem bufResult = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(puzzle_hunter_result_gpu_t), &zeroResult, &err);
        cl_mem bufStartPoints = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(point_affine_gpu_t) * groupsPerLaunch, nullptr, &err);
        cl_mem bufKeys = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint256_gpu_t) * groupsPerLaunch, nullptr, &err);
        cl_mem bufThreadChain = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint256_gpu_t) * groupsPerLaunch * (pointsPerThread - 1), nullptr, &err);

        checkCl(clSetKernelArg(startKernel, 0, sizeof(cl_mem), &bufRngStates), "karg 0");
        checkCl(clSetKernelArg(startKernel, 1, sizeof(cl_mem), &bufGTable), "karg 1");
        checkCl(clSetKernelArg(startKernel, 2, sizeof(cl_mem), &bufPrefix), "karg 2");
        checkCl(clSetKernelArg(startKernel, 3, sizeof(int), &prefixLen), "karg 3");
        checkCl(clSetKernelArg(startKernel, 4, sizeof(cl_mem), &bufStartPoints), "karg 4");
        checkCl(clSetKernelArg(startKernel, 5, sizeof(cl_mem), &bufResult), "karg 5");
        checkCl(clSetKernelArg(startKernel, 6, sizeof(cl_mem), &bufKeys), "karg 6");
        checkCl(clSetKernelArg(startKernel, 7, sizeof(uint256_gpu_t), &gpuMinKey), "karg 7");
        checkCl(clSetKernelArg(startKernel, 8, sizeof(uint256_gpu_t), &gpuMaxKey), "karg 8");
        checkCl(clSetKernelArg(startKernel, 9, sizeof(uint256_gpu_t), &gpuRangeMask), "karg 9");
        checkCl(clSetKernelArg(startKernel, 10, sizeof(uint256_gpu_t), &gpuRangeLimit), "karg 10");

        checkCl(clSetKernelArg(threadBatchKernel, 0, sizeof(cl_mem), &bufKeys), "karg 0b");
        checkCl(clSetKernelArg(threadBatchKernel, 1, sizeof(cl_mem), &bufBatchTable), "karg 1b");
        checkCl(clSetKernelArg(threadBatchKernel, 2, sizeof(cl_mem), &bufPrefix), "karg 2b");
        checkCl(clSetKernelArg(threadBatchKernel, 3, sizeof(int), &prefixLen), "karg 3b");
        checkCl(clSetKernelArg(threadBatchKernel, 4, sizeof(cl_mem), &bufStartPoints), "karg 4b");
        checkCl(clSetKernelArg(threadBatchKernel, 5, sizeof(cl_mem), &bufResult), "karg 5b");
        checkCl(clSetKernelArg(threadBatchKernel, 6, sizeof(cl_mem), &bufThreadChain), "karg 6b");

        unsigned __int128 totalChecked = 0;
        double totalElapsedSeconds = 0.0;
        std::string progressFilename = "progress_pub_" + targetPubHex + ".txt";

        if (benchLoops == 0) {
            std::ifstream pfile(progressFilename);
            if (pfile) {
                std::string line;
                if (std::getline(pfile, line)) {
                    totalChecked = stringToInt128(line);
                }
                if (std::getline(pfile, line)) {
                    try {
                        totalElapsedSeconds = std::stod(line);
                    } catch (...) {
                        totalElapsedSeconds = 0.0;
                    }
                }
                std::cout << "Loaded previous progress: " << int128ToString(totalChecked) 
                          << " keys checked, " << formatElapsed(totalElapsedSeconds) << " elapsed.\n";
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto lastSaveTime = start;

        std::cout << "Target        : " << targetPubHex.substr(0, 16) << "..." << targetPubHex.substr(58) << "\n";
        std::cout << "Prefix bytes  : " << prefixLen << "\n";
        if (puzzleProvided) {
            std::cout << "Puzzle        : " << puzzleBits << " bits\n";
        }
        std::cout << "Range         : " << compactHex(minKey) << ":" << compactHex(maxKey) << "\n";
        std::cout << "Max Byte Count: " << maxBytes << "\n";
        std::cout << "Blocks        : " << (groupsPerLaunch / localWorkSize) << "\n";
        std::cout << "Threads       : " << localWorkSize << "\n";
        std::cout << "Points/Thread : " << pointsPerThread << "\n";
        std::cout << "Checked/loop  : " << (groupsPerLaunch * checkedPerGroup) << "\n";
        if (benchLoops > 0) {
            std::cout << "Benchmark     : " << benchLoops << " loops\n";
        }
        std::cout << "Search started. Press [SPACE] to pause, [ESC] to exit.\n";

        int completedLoops = 0;
        int partialBlockY = -1; bool paused = false;
        while(true) {
            if (_kbhit()) {
                int ch = _getch();
                if (ch == 27) {
                    if (benchLoops == 0) {
                        double finalElapsed = totalElapsedSeconds + std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
                        std::ofstream pfile(progressFilename);
                        if (pfile) {
                            pfile << int128ToString(totalChecked) << "\n";
                            pfile << finalElapsed << "\n";
                        }
                        std::cout << "\nProgress saved. Total checked: " << int128ToString(totalChecked) << "\n";
                    }
                    break;
                }
                if (ch == ' ' || ch == 'p' || ch == 'P') {
                    paused = !paused;
                    if (paused) {
                        const auto now = std::chrono::high_resolution_clock::now();
                        totalElapsedSeconds += std::chrono::duration<double>(now - start).count();
                        std::cout << "\r[PAUSED] Press Space to resume..." << std::string(60, ' ') << std::flush;
                    } else {
                        start = std::chrono::high_resolution_clock::now();
                    }
                }
            }
            if (paused) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }

            checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_FALSE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr), "reset");
            checkCl(clEnqueueNDRangeKernel(queue, startKernel, 1, nullptr, &groupsPerLaunch, nullptr, 0, nullptr, nullptr), "start");
            checkCl(clEnqueueNDRangeKernel(queue, threadBatchKernel, 1, nullptr, &groupsPerLaunch, &localWorkSize, 0, nullptr, nullptr), "batch");

            totalChecked += (unsigned __int128)groupsPerLaunch * checkedPerGroup;
            puzzle_hunter_result_gpu_t res = {};
            checkCl(clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(res), &res, 0, nullptr, nullptr), "read");

            const auto now = std::chrono::high_resolution_clock::now();
            const double elapsed = totalElapsedSeconds + std::chrono::duration<double>(now - start).count();
            printProgressLine(elapsed, (double)totalChecked / elapsed / 1e6, totalChecked);

            // Periodically save progress (every 60 seconds)
            if (benchLoops == 0 && std::chrono::duration<double>(now - lastSaveTime).count() >= 60.0) {
                std::ofstream pfile(progressFilename);
                if (pfile) {
                    pfile << int128ToString(totalChecked) << "\n";
                    pfile << elapsed << "\n";
                }
                lastSaveTime = now;
            }

            if(res.flag != 0) {
                const std::string privHex = gpuToHex(res.privKey), pubHex = pointToCompressedHex(res.x, res.y);
                Int verifyKey = hexToInt(privHex); Point verifyPoint = secp->ComputePublicKey(&verifyKey);
                const std::string cpuPubHex = pointToCompressedHex(verifyPoint);
                if(cpuPubHex != pubHex) {
                    std::cout << "\nWarning: Failed CPU verification.\n";
                    checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr), "reset_err");
                    continue;
                }
                if(cpuPubHex == targetPubHex) {
                    std::cout << "\n================== FOUND FULL MATCH! ====================\n";
                    std::cout << "Private Key: " << privHex << "\nPublic Key:  " << cpuPubHex << "\n";
                    std::ofstream f("KEYFOUND.txt");
                    if(f) {
                        f << "Private: " << privHex << "\nPublic: " << cpuPubHex << "\nTotal Checked: " << int128ToString(totalChecked) << "\n";
                        f << "Elapsed Time:  " << formatElapsed(elapsed) << "\n";
                        f << "Speed:         " << (double)totalChecked / elapsed / 1e6 << " Mkeys/s\n";
                    }
                    if (benchLoops == 0) {
                        std::ofstream pfile(progressFilename);
                        if (pfile) {
                            pfile << int128ToString(totalChecked) << "\n";
                            pfile << elapsed << "\n";
                        }
                    }
                    break;
                } else {
                    clearConsoleLine();
                    if(partialBlockY < 0) { partialBlockY = getConsoleCursorY(); printPartialBlock(prefixLen, privHex, cpuPubHex, targetPubHex); }
                    else { const int py = getConsoleCursorY(); setConsoleCursorY(partialBlockY); for(int i=0;i<5;i++) clearConsoleLine(); setConsoleCursorY(partialBlockY); printPartialBlock(prefixLen, privHex, cpuPubHex, targetPubHex); setConsoleCursorY(py); }
                    printProgressLine(elapsed, (double)totalChecked / elapsed / 1e6, totalChecked);
                    checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr), "reset_part");
                }
            }
            completedLoops++;
            if(benchLoops > 0 && completedLoops >= benchLoops) {
                std::cout << "\nBenchmark done.\n";
                break;
            }
        }
        clReleaseMemObject(bufResult); clReleaseMemObject(bufStartPoints); clReleaseMemObject(bufKeys); clReleaseMemObject(bufThreadChain);
        clReleaseMemObject(bufPrefix); clReleaseMemObject(bufBatchTable); clReleaseMemObject(bufGTable); clReleaseMemObject(bufRngStates);
        clReleaseKernel(startKernel); clReleaseKernel(threadBatchKernel); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return 0;
    } catch(const std::exception& ex) { std::cerr << "\nError: " << ex.what() << "\n"; return 1; }
}
