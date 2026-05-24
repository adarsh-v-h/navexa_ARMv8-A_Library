#include "armv8lib/mte.h"
#include <cstdint>

namespace navexa {
namespace mte {

uint8_t get_pointer_tag(const void* ptr) {
    if (!ptr) return 0;
    
    // The tag is stored in bits 59:56 of the 64-bit pointer
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    return static_cast<uint8_t>((addr >> 56) & 0xF);
}

uint8_t get_memory_tag(const void* ptr) {
    if (!ptr) return 0;

    uint64_t mem_tag;
    // ldg (Load Allocation Tag) fetches the hardware tag for the granule pointed to by %1.
    // It places the tag into bits 59:56 of the destination register %0.
    asm volatile("ldg %0, [%1]" : "=r"(mem_tag) : "r"(ptr) : "memory");
    
    return static_cast<uint8_t>((mem_tag >> 56) & 0xF);
}

bool is_valid_pointer(const void* ptr) {
    if (!ptr) return false;
    
    // If the tag embedded in the pointer matches the tag locked in memory, 
    // the pointer is within its valid allocated bounds.
    return get_pointer_tag(ptr) == get_memory_tag(ptr);
}

} // namespace mte
} // namespace navexa