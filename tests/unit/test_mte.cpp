#include "armv8lib/mte.h"
#include <iostream>
#include <cstring>

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

    // --- TEST 1.5: Safe Software Bounds Checking (ADDED HERE) ---
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

    // --- TEST 2: Intentional Overflow ---
    std::cout << "\n[MTE Validation] TRIGGERING CONTROLLED BUFFER OVERFLOW..." << std::endl;
    buffer[32] = 'X'; 

    navexa::mte::free(buffer, alloc_size);
    return 0;
}