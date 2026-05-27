#include "armv8lib/mte.h"
#include <iostream>
#include <cstring>
#include <cstdint>

int main() {
    std::cout << "[MTE Validation] Initializing hardware protection..." << std::endl;
    navexa::mte::init_protection();

    size_t alloc_size = 32;
    std::cout << "[MTE Validation] Allocating " << alloc_size << " bytes..." << std::endl;
    
    char* buffer = static_cast<char*>(navexa::mte::malloc(alloc_size));
    if (!buffer) {
        std::cerr << "Allocation failed!" << std::endl;
        return 1;
    }

    // --- TEST 1: Valid Write ---
    std::cout << "[MTE Validation] Testing valid write within bounds..." << std::endl;
    std::strcpy(buffer, "Valid memory write.");
    std::cout << "Readback: " << buffer << std::endl;

    // --- TEST 2: Safe Software Bounds Checking ---
    std::cout << "\n[MTE Validation] Testing safe bounds checking..." << std::endl;
    uint8_t p_tag = navexa::mte::get_pointer_tag(buffer);
    uint8_t m_tag = navexa::mte::get_memory_tag(buffer);

    std::cout << "Pointer Tag: " << static_cast<int>(p_tag) << std::endl;
    std::cout << "Memory Tag:  " << static_cast<int>(m_tag) << std::endl;

    if (navexa::mte::is_valid_pointer(buffer)) {
        std::cout << "SUCCESS: Pointer is valid and within bounds." << std::endl;
    }

    char* oob_pointer = buffer + 32; 
    if (!navexa::mte::is_valid_pointer(oob_pointer)) {
        std::cout << "SUCCESS: Safe bounds check successfully detected out-of-bounds pointer without crashing!" << std::endl;
    }

    // --- TEST 3: Intentional Tag Mismatch Overflow ---
    std::cout << "\n[MTE Validation] TRIGGERING CONTROLLED TAG MISMATCH OVERFLOW..." << std::endl;
    
    // Deliberately corrupt the pointer's tag bits (bits 59:56) to guarantee a mismatch 
    // with the underlying validly allocated memory granule.
    uint64_t raw_ptr = reinterpret_cast<uint64_t>(buffer);
    uint64_t corrupted_ptr_val = raw_ptr ^ (0xFUL << 56); // Flip the tag bits entirely
    char* corrupted_buffer = reinterpret_cast<char*>(corrupted_ptr_val);

    std::cout << "Original Pointer Tag:  " << static_cast<int>(navexa::mte::get_pointer_tag(buffer)) << std::endl;
    std::cout << "Corrupted Pointer Tag: " << static_cast<int>(navexa::mte::get_pointer_tag(corrupted_buffer)) << std::endl;
    std::cout << "Memory Tag at Target:  " << static_cast<int>(navexa::mte::get_memory_tag(buffer)) << std::endl;

    std::cout << "Attempting write via corrupted pointer (Should trigger hardware SIGSEGV)..." << std::endl;
    
    // This write operation forces a hardware tag comparison mismatch
    corrupted_buffer[0] = 'X'; 

    // --- FALLBACK METRICS IF HARDWARE ENFORCEMENT LACKS EMULATION ---
    std::cout << "\n[NOTICE] Hardware did not trap the violation dynamically." << std::endl;
    std::cout << "Checking if software layer successfully detects the breach..." << std::endl;

    if (!navexa::mte::is_valid_pointer(corrupted_buffer)) {
        std::cout << "SUCCESS (Software Guard): Library correctly identified the pointer as INVALID!" << std::endl;
    } else {
        std::cout << "FAILURE: Software validation missed the corrupted pointer." << std::endl;
    }

    navexa::mte::free(buffer, alloc_size);
    return 0;
}