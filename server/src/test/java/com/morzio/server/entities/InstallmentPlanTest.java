package com.morzio.server.entities;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class InstallmentPlanTest {

    private InstallmentPlan installmentPlan;
    private PaymentSession paymentSession;

    @BeforeEach
    void setUp() {
        paymentSession = new PaymentSession();
        paymentSession.setStatus("active");
        paymentSession.setAmount(1200L);

        installmentPlan = new InstallmentPlan();
        installmentPlan.setId(1L);
        installmentPlan.setInstallmentCount(12L);
        installmentPlan.setAmountPerInstallment(100L);
        installmentPlan.setTotalAmount(1200L);
        installmentPlan.setInterestRate(5L);
        installmentPlan.setPaymentSession(paymentSession);
    }

    @Test
    void testInstallmentPlanCreation() {
        assertNotNull(installmentPlan);
        assertEquals(1L, installmentPlan.getId());
        assertEquals(12L, installmentPlan.getInstallmentCount());
        assertEquals(100L, installmentPlan.getAmountPerInstallment());
        assertEquals(1200L, installmentPlan.getTotalAmount());
        assertEquals(5L, installmentPlan.getInterestRate());
    }

    @Test
    void testPaymentSessionRelationship() {
        assertNotNull(installmentPlan.getPaymentSession());
        assertEquals(paymentSession, installmentPlan.getPaymentSession());
    }

    @Test
    void testInstallmentCountUpdate() {
        installmentPlan.setInstallmentCount(24L);
        assertEquals(24L, installmentPlan.getInstallmentCount());
    }

    @Test
    void testAmountPerInstallmentUpdate() {
        installmentPlan.setAmountPerInstallment(50L);
        assertEquals(50L, installmentPlan.getAmountPerInstallment());
    }

    @Test
    void testTotalAmountUpdate() {
        installmentPlan.setTotalAmount(2400L);
        assertEquals(2400L, installmentPlan.getTotalAmount());
    }

    @Test
    void testInterestRateUpdate() {
        installmentPlan.setInterestRate(10L);
        assertEquals(10L, installmentPlan.getInterestRate());
    }

    @Test
    void testInstallmentPlanEquality() {
        InstallmentPlan plan2 = new InstallmentPlan(1L, 12L, 100L, 1200L, 5L, paymentSession);
        assertEquals(installmentPlan, plan2);
    }

    @Test
    void testInstallmentPlanNoArgsConstructor() {
        InstallmentPlan plan = new InstallmentPlan();
        assertNull(plan.getId());
        assertNull(plan.getInstallmentCount());
        assertNull(plan.getAmountPerInstallment());
    }
}
