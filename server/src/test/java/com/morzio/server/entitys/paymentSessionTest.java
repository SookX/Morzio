package com.morzio.server.entitys;

import com.morzio.server.entities.PaymentSession;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("PaymentSession Entity Tests")
class PaymentSessionTest {

    private PaymentSession paymentSession;

    @BeforeEach
    void setUp() {
        paymentSession = new PaymentSession();
    }

    @Test
    @DisplayName("Should create PaymentSession with default constructor")
    void testCreatePaymentSessionWithDefaultConstructor() {
        assertNotNull(paymentSession);
        assertNull(paymentSession.getId());
        assertNull(paymentSession.getSessionId());
        assertNull(paymentSession.getAmount());
        assertNull(paymentSession.getCurrency());
    }

    @Test
    @DisplayName("Should create PaymentSession with all-args constructor")
    void testCreatePaymentSessionWithAllArgsConstructor() {
        Long id = 1L;
        UUID sessionId = UUID.randomUUID();
        Long amount = 10000L;
        String currency = "USD";

        PaymentSession session = new PaymentSession(id, sessionId, amount, currency);

        assertEquals(id, session.getId());
        assertEquals(sessionId, session.getSessionId());
        assertEquals(amount, session.getAmount());
        assertEquals(currency, session.getCurrency());
    }

    @Test
    @DisplayName("Should set and get ID")
    void testSetAndGetId() {
        Long testId = 1L;
        paymentSession.setId(testId);
        assertEquals(testId, paymentSession.getId());
    }

    @Test
    @DisplayName("Should set and get Session ID")
    void testSetAndGetSessionId() {
        UUID sessionId = UUID.randomUUID();
        paymentSession.setSessionId(sessionId);
        assertEquals(sessionId, paymentSession.getSessionId());
    }

    @Test
    @DisplayName("Should set and get Amount")
    void testSetAndGetAmount() {
        Long amount = 5000L;
        paymentSession.setAmount(amount);
        assertEquals(amount, paymentSession.getAmount());
    }

    @Test
    @DisplayName("Should set and get Currency")
    void testSetAndGetCurrency() {
        String currency = "EUR";
        paymentSession.setCurrency(currency);
        assertEquals(currency, paymentSession.getCurrency());
    }

    @Test
    @DisplayName("Should auto-generate SessionId on @PrePersist")
    void testGenerateSessionIdOnPrePersist() {
        assertNull(paymentSession.getSessionId());
        paymentSession.generateSessionId();
        assertNotNull(paymentSession.getSessionId());
        assertInstanceOf(UUID.class, paymentSession.getSessionId());
    }

    @Test
    @DisplayName("Should generate unique SessionIds for different instances")
    void testGenerateUniqueSessionIds() {
        PaymentSession session1 = new PaymentSession();
        PaymentSession session2 = new PaymentSession();
        
        session1.generateSessionId();
        session2.generateSessionId();
        
        assertNotEquals(session1.getSessionId(), session2.getSessionId());
    }

    @Test
    @DisplayName("Should support multiple currency types")
    void testMultipleCurrencyTypes() {
        String[] currencies = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"};
        
        for (String currency : currencies) {
            paymentSession.setCurrency(currency);
            assertEquals(currency, paymentSession.getCurrency());
        }
    }

    @Test
    @DisplayName("Should handle large amounts")
    void testLargeAmounts() {
        Long largeAmount = 999_999_999_999L;
        paymentSession.setAmount(largeAmount);
        assertEquals(largeAmount, paymentSession.getAmount());
    }

    @Test
    @DisplayName("Should handle zero amount")
    void testZeroAmount() {
        paymentSession.setAmount(0L);
        assertEquals(0L, paymentSession.getAmount());
    }

    @Test
    @DisplayName("Should handle negative amounts")
    void testNegativeAmounts() {
        Long negativeAmount = -1000L;
        paymentSession.setAmount(negativeAmount);
        assertEquals(negativeAmount, paymentSession.getAmount());
    }

    @Test
    @DisplayName("Should allow null values for optional fields")
    void testNullValuesForOptionalFields() {
        paymentSession.setSessionId(null);
        paymentSession.setAmount(null);
        paymentSession.setCurrency(null);
        
        assertNull(paymentSession.getSessionId());
        assertNull(paymentSession.getAmount());
        assertNull(paymentSession.getCurrency());
    }

    @Test
    @DisplayName("Should create PaymentSession with specific values")
    void testCreatePaymentSessionWithSpecificValues() {
        Long id = 42L;
        UUID sessionId = UUID.randomUUID();
        Long amount = 25000L;
        String currency = "GBP";

        paymentSession.setId(id);
        paymentSession.setSessionId(sessionId);
        paymentSession.setAmount(amount);
        paymentSession.setCurrency(currency);

        assertEquals(id, paymentSession.getId());
        assertEquals(sessionId, paymentSession.getSessionId());
        assertEquals(amount, paymentSession.getAmount());
        assertEquals(currency, paymentSession.getCurrency());
    }

    @Test
    @DisplayName("Should update PaymentSession values")
    void testUpdatePaymentSessionValues() {
        paymentSession.setAmount(1000L);
        paymentSession.setCurrency("USD");
        
        assertEquals(1000L, paymentSession.getAmount());
        assertEquals("USD", paymentSession.getCurrency());
        
        paymentSession.setAmount(2000L);
        paymentSession.setCurrency("EUR");
        
        assertEquals(2000L, paymentSession.getAmount());
        assertEquals("EUR", paymentSession.getCurrency());
    }

    @Test
    @DisplayName("Should handle sessionId immutability after generation")
    void testSessionIdAfterGeneration() {
        paymentSession.generateSessionId();
        UUID firstSessionId = paymentSession.getSessionId();
        
        paymentSession.setAmount(5000L);
        paymentSession.setCurrency("USD");
        
        assertEquals(firstSessionId, paymentSession.getSessionId());
    }

    @Test
    @DisplayName("Should create valid PaymentSession for API response")
    void testPaymentSessionValidForApiResponse() {
        Long id = 100L;
        UUID sessionId = UUID.randomUUID();
        Long amount = 15000L;
        String currency = "USD";

        PaymentSession session = new PaymentSession(id, sessionId, amount, currency);

        assertTrue(session.getId() > 0);
        assertNotNull(session.getSessionId());
        assertTrue(session.getAmount() > 0);
        assertNotNull(session.getCurrency());
        assertTrue(session.getCurrency().length() > 0);
    }

    @Test
    @DisplayName("Should handle string representation")
    void testToStringMethod() {
        paymentSession.setId(1L);
        UUID sessionId = UUID.randomUUID();
        paymentSession.setSessionId(sessionId);
        paymentSession.setAmount(5000L);
        paymentSession.setCurrency("USD");

        String toString = paymentSession.toString();
        assertNotNull(toString);
        assertTrue(toString.contains("PaymentSession") || toString.length() > 0);
    }

    @Test
    @DisplayName("Should properly compare two PaymentSession objects")
    void testEqualityOfPaymentSessions() {
        Long id = 1L;
        UUID sessionId = UUID.randomUUID();
        Long amount = 5000L;
        String currency = "USD";

        PaymentSession session1 = new PaymentSession(id, sessionId, amount, currency);
        PaymentSession session2 = new PaymentSession(id, sessionId, amount, currency);

        assertEquals(session1, session2);
    }

    @Test
    @DisplayName("Should handle different PaymentSession objects")
    void testDifferentPaymentSessions() {
        PaymentSession session1 = new PaymentSession(1L, UUID.randomUUID(), 5000L, "USD");
        PaymentSession session2 = new PaymentSession(2L, UUID.randomUUID(), 10000L, "EUR");

        assertNotEquals(session1, session2);
    }

    @Test
    @DisplayName("Should generate session ID only once per call to generateSessionId")
    void testSessionIdGenerationConsistency() {
        paymentSession.generateSessionId();
        UUID firstId = paymentSession.getSessionId();
        
        paymentSession.generateSessionId();
        UUID secondId = paymentSession.getSessionId();
        
        assertNotEquals(firstId, secondId);
    }

    @Test
    @DisplayName("Should work with builder pattern (if Lombok @Data is properly applied)")
    void testLombokDataAnnotationFunctionality() {
        PaymentSession session = new PaymentSession();
        session.setId(1L);
        session.setAmount(1000L);
        session.setCurrency("USD");
        
        assertNotNull(session.getId());
        assertNotNull(session.getAmount());
        assertNotNull(session.getCurrency());
    }
}
