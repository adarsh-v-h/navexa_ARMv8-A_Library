#include "armv8lib/mte.h"
#include <sys/mman.h>
#include <sys/prctl.h>
#include <cstdint>
#include <cstdio>  // Added for perror and printf

namespace navexa {
namespace mte {

void init_protection() {
    // Enable synchronous tag checking and explicitly check for environment support
    if (prctl(PR_SET_TAGGED_ADDR_CTRL, PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC, 0, 0, 0) < 0) {
        std::perror("[MTE SYSTEM WARNING] prctl failed. Your current QEMU build or host kernel does not support hardware MTE enforcement");
    } else {
        std::printf("[MTE SYSTEM INFO] Hardware MTE initialized successfully in Synchronous mode.\n");
    }
}

void* malloc(size_t size) {
    if (size == 0) return nullptr;
    
    // Enforce 16-byte granule alignment
    size_t aligned_size = (size + 15) & ~15; 
    
    // Request MTE-capable memory from the kernel
    void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE | PROT_MTE, 
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                     
    if (ptr == MAP_FAILED) return nullptr;

    uint64_t tagged_ptr;
    // Generate a random 4-bit tag and insert it into bits 59:56
    asm("irg %0, %1" : "=r"(tagged_ptr) : "r"(ptr)); 

    uint64_t current_block = tagged_ptr;
    
    // Loop and apply the tag to every 16-byte granule
    for (size_t i = 0; i < aligned_size; i += 16) {
        asm volatile("stg %0, [%0]" : : "r"(current_block) : "memory"); 
        current_block += 16;
    }
    
    return reinterpret_cast<void*>(tagged_ptr);
}

void free(void* ptr, size_t size) {
    if (!ptr) return;
    
    size_t aligned_size = (size + 15) & ~15;
    uint64_t current_block = reinterpret_cast<uint64_t>(ptr);
    
    // Wipe the tags before returning memory to the OS
    for (size_t i = 0; i < aligned_size; i += 16) {
        // Strip the tag (bits 59:56) to access the raw physical block
        uint64_t untagged_block = current_block & (~(0xFUL << 56));
        asm volatile("stg %0, [%0]" : : "r"(untagged_block) : "memory"); 
        current_block += 16;
    }
    
    munmap(ptr, aligned_size);
}

} // namespace mte
} // namespace navexa