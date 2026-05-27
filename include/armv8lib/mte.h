#include <cstdint>
#pragma once
#include <cstddef>

namespace navexa {
namespace mte {

/**
 * @brief Initializes MTE protection for the current thread.
 * Enables PR_MTE_TCF_SYNC to catch tag faults immediately.
 */
void init_protection();

/**
 * @brief Allocates 16-byte aligned memory mapped with PROT_MTE and generates a hardware tag.
 * * @param size Requested allocation size in bytes.
 * @return Pointer to the tagged memory block, or nullptr on failure.
 */
void* malloc(size_t size);

/**
 * @brief Clears the hardware tags and unmaps the memory block.
 * * @param ptr Pointer to the tagged memory block.
 * @param size Original requested size of the allocation.
 */
void free(void* ptr, size_t size);

/**
 * @brief Extracts the 4-bit security tag embedded in the top byte of the pointer.
 * @param ptr The tagged pointer.
 * @return The 4-bit tag (0-15).
 */
uint8_t get_pointer_tag(const void* ptr);

/**
 * @brief Reads the hardware allocation tag stored in physical memory for a given address.
 * Uses the ARMv8.5 'ldg' (Load Allocation Tag) instruction.
 * @param ptr The memory address to query.
 * @return The 4-bit hardware tag (0-15).
 */
uint8_t get_memory_tag(const void* ptr);

/**
 * @brief Safely validates if a pointer's tag matches the physical memory tag.
 * Allows bounds checking without triggering a fatal SIGSEGV hardware trap.
 * @param ptr The pointer to validate.
 * @return true if tags match (safe to access), false if mismatch (out of bounds / use-after-free).
 */
bool is_valid_pointer(const void* ptr);

} // namespace mte
} // namespace navexa