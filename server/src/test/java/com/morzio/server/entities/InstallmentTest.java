package com.morzio.server.entities;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class InstallmentTest {

    private Installment installment;
    private InstallmentPlan installmentPlan;

    @BeforeEach
    void setUp() {
        installmentPlan = new InstallmentPlan();
        installmentPlan.setId(1L);
        installmentPlan.setInstallmentCount(12L);
        installmentPlan.setAmountPerInstallment(100L);
        installmentPlan.setTotalAmount(1200L);
        installmentPlan.setInterestRate(5L);

        installment = new Installment();
        installment.setId(1L);
        installment.setAmount(100L);
        installment.setStatus("pending");
        installment.setInstallmentPlan(installmentPlan);
    }

    @Test
    void testInstallmentCreation() {
        assertNotNull(installment);
        assertEquals(1L, installment.getId());
        assertEquals(100L, installment.getAmount());
        assertEquals("pending", installment.getStatus());
    }

    @Test
    void testInstallmentPlanRelationship() {
        assertNotNull(installment.getInstallmentPlan());
        assertEquals(installmentPlan.getId(), installment.getInstallmentPlan().getId());
    }

    @Test
    void testInstallmentStatusUpdate() {
        installment.setStatus("completed");
        assertEquals("completed", installment.getStatus());
    }

    @Test
    void testInstallmentAmountUpdate() {
        installment.setAmount(150L);
        assertEquals(150L, installment.getAmount());
    }

    @Test
    void testInstallmentEquality() {
        Installment installment2 = new Installment(1L, 100L, "pending", installmentPlan);
        assertEquals(installment, installment2);
    }

    @Test
    void testInstallmentNoArgsConstructor() {
        Installment inst = new Installment();
        assertNull(inst.getId());
        assertNull(inst.getAmount());
        assertNull(inst.getStatus());
    }
}
