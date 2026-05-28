// =============================================================================
// tests/unit/test_virt.cpp
// Tests for armv8::virt virtualization helpers
// =============================================================================

#include "armv8lib/virt/virt.hpp"
#include <gtest/gtest.h>

using namespace armv8::virt;

// =============================================================================
// VmContext tests
// =============================================================================

TEST(VmContextTest, DefaultInvalid) {
    VmContext ctx;
    EXPECT_FALSE(ctx.is_valid());  // vmid=0 means invalid
}

TEST(VmContextTest, ValidAfterVmid) {
    VmContext ctx;
    ctx.vmid = 1;
    EXPECT_TRUE(ctx.is_valid());
}

TEST(VmContextTest, RegisterAccess) {
    VmContext ctx;
    ctx.vmid = 1;
    ctx.x(0) = 0xDEADBEEF;
    ctx.x(30) = 0xCAFEBABE;
    EXPECT_EQ(ctx.x(0),  0xDEADBEEFu);
    EXPECT_EQ(ctx.x(30), 0xCAFEBABEu);
}

TEST(VmContextTest, RegisterOutOfRange) {
    VmContext ctx;
    EXPECT_THROW(ctx.x(31), std::out_of_range);
}

TEST(VmContextTest, Reset) {
    VmContext ctx;
    ctx.vmid = 1;
    ctx.x(0) = 999;
    ctx.sp   = 0x1000;
    ctx.reset();
    EXPECT_EQ(ctx.x(0), 0u);
    EXPECT_EQ(ctx.sp, 0u);
}

// =============================================================================
// Stage2Entry tests
// =============================================================================

TEST(Stage2EntryTest, DefaultInvalid) {
    Stage2Entry e;
    EXPECT_FALSE(e.valid);
}

TEST(Stage2EntryTest, Translation) {
    Stage2Entry e(0x1000, 0x2000);
    EXPECT_EQ(e.translate(0x1000), 0x2000u);
}

TEST(Stage2EntryTest, TranslationMismatch) {
    Stage2Entry e(0x1000, 0x2000);
    EXPECT_THROW(e.translate(0x3000), std::runtime_error);
}

TEST(Stage2EntryTest, Permissions) {
    Stage2Entry rw(0x1000, 0x2000, Stage2MemAttr::Normal_WB, Stage2Perm::RW);
    EXPECT_TRUE(rw.permits_read());
    EXPECT_TRUE(rw.permits_write());

    Stage2Entry ro(0x1000, 0x2000, Stage2MemAttr::Normal_WB, Stage2Perm::Read);
    EXPECT_TRUE(ro.permits_read());
    EXPECT_FALSE(ro.permits_write());

    Stage2Entry none(0x1000, 0x2000, Stage2MemAttr::Normal_WB, Stage2Perm::None);
    EXPECT_FALSE(none.permits_read());
    EXPECT_FALSE(none.permits_write());
}

// =============================================================================
// VirtInterrupt tests
// =============================================================================

TEST(VirtInterruptTest, DefaultNotPending) {
    VirtInterrupt irq;
    EXPECT_FALSE(irq.should_deliver());
}

TEST(VirtInterruptTest, AssertAndDeassert) {
    VirtInterrupt irq(32, VirtIntType::IRQ, 0, 1);
    EXPECT_FALSE(irq.should_deliver());
    irq.assert_int();
    EXPECT_TRUE(irq.should_deliver());
    irq.deassert_int();
    EXPECT_FALSE(irq.should_deliver());
}

TEST(VirtInterruptTest, DisabledDoesNotDeliver) {
    VirtInterrupt irq(32, VirtIntType::IRQ, 0, 1);
    irq.enabled = false;
    irq.assert_int();
    EXPECT_FALSE(irq.should_deliver());
}

// =============================================================================
// Context switch tests
// =============================================================================

TEST(ContextSwitchTest, ValidSwitch) {
    VmContext vm1, vm2;
    vm1.vmid = 1; vm1.x(0) = 100;
    vm2.vmid = 2; vm2.x(0) = 200;

    auto result = simulate_context_switch(vm1, vm2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.from_vmid, 1u);
    EXPECT_EQ(result.to_vmid,   2u);
    EXPECT_GT(result.cycles_est, 0u);
}

TEST(ContextSwitchTest, InvalidContextFails) {
    VmContext vm1, vm2;
    // both vmid=0 → invalid
    auto result = simulate_context_switch(vm1, vm2);
    EXPECT_FALSE(result.success);
}

TEST(ContextSwitchTest, StatePreservedAfterSwitch) {
    VmContext vm1, vm2;
    vm1.vmid = 1; vm1.x(5) = 0xABCD;
    vm2.vmid = 2; vm2.x(5) = 0x1234;

    simulate_context_switch(vm1, vm2);
    // vm1 state should still be intact (saved)
    EXPECT_EQ(vm1.x(5), 0xABCDu);
    // vm2 state unchanged (restored)
    EXPECT_EQ(vm2.x(5), 0x1234u);
}
