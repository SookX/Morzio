package com.morzio.server.entities;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.sql.Timestamp;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class PaymentSessionTest {

    private PaymentSession paymentSession;

    @BeforeEach
    void setUp() {
        paymentSession = new PaymentSession();
        paymentSession.setStatus("active");
        paymentSession.setAmount(1000L);
    }

    @Test
    void testPaymentSessionCreation() {
        assertNotNull(paymentSession);
        assertEquals("active", paymentSession.getStatus());
        assertEquals(1000L, paymentSession.getAmount());
    }

    @Test
    void testPaymentSessionIdGeneration() {
        paymentSession.generateId();
        assertNotNull(paymentSession.getId());
        assertInstanceOf(UUID.class, paymentSession.getId());
    }

    @Test
    void testPaymentSessionStatusUpdate() {
        paymentSession.setStatus("completed");
        assertEquals("completed", paymentSession.getStatus());
    }

    @Test
    void testPaymentSessionAmountUpdate() {
        paymentSession.setAmount(2000L);
        assertEquals(2000L, paymentSession.getAmount());
    }

    @Test
    void testPaymentSessionWithTimestamp() {
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        paymentSession.setCreated(timestamp);
        assertNotNull(paymentSession.getCreated());
        assertEquals(timestamp, paymentSession.getCreated());
    }

    @Test
    void testPaymentSessionEquality() {
        UUID testId = UUID.randomUUID();
        PaymentSession session2 = new PaymentSession(testId, "active", 1000L, null);
        PaymentSession session1 = new PaymentSession(testId, "active", 1000L, null);
        assertEquals(session1, session2);
    }

    @Test
    void testPaymentSessionNoArgsConstructor() {
        PaymentSession session = new PaymentSession();
        assertNull(session.getId());
        assertNull(session.getStatus());
        assertNull(session.getAmount());
        assertNull(session.getCreated());
    }

    @Test
    void testPaymentSessionAllArgsConstructor() {
        UUID id = UUID.randomUUID();
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        PaymentSession session = new PaymentSession(id, "pending", 500L, timestamp);
        assertEquals(id, session.getId());
        assertEquals("pending", session.getStatus());
        assertEquals(500L, session.getAmount());
        assertEquals(timestamp, session.getCreated());
    }
}
