package com.morzio.server.controllers;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.util.UUID;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.morzio.server.controllers.PaymentsController.PaymentController;
import com.morzio.server.dtos.PaymentService.PaymentInitiateRequestDto;
import com.morzio.server.dtos.PaymentService.PaymentInitiateResponseDto;
import com.morzio.server.services.PaymentService.PaymentServiceImpl.PaymentServiceImpl;

@ExtendWith(MockitoExtension.class)
class PaymentControllerTest {

    @Mock
    private PaymentServiceImpl paymentService;

    @InjectMocks
    private PaymentController paymentController;

    private UUID testSessionId;
    private Long testAmount;

    @BeforeEach
    void setUp() {
        testSessionId = UUID.randomUUID();
        testAmount = 10000L;
    }

    @Test
    void testInitiatePayment_Success() {
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(testAmount);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                testSessionId,
                "PENDING",
                "http://morzio.com/pay/" + testSessionId.toString()
        );

        when(paymentService.url_for_qr(testAmount)).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertNotNull(response, "Response should not be null");
        assertEquals(testSessionId, response.getSessionId(), "Session ID should match");
        assertEquals("PENDING", response.getStatus(), "Status should be PENDING");
        assertTrue(response.getPaymentUrl().contains("http://morzio.com/pay/"), "Payment URL should contain base URL");
        verify(paymentService, times(1)).url_for_qr(testAmount);
    }

    @Test
    void testInitiatePayment_WithDifferentAmounts() {
        Long[] amounts = {1L, 100L, 50000L, 999999L};

        for (Long amount : amounts) {
            PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(amount);
            UUID sessionId = UUID.randomUUID();
            PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                    sessionId,
                    "PENDING",
                    "http://morzio.com/pay/" + sessionId.toString()
            );

            when(paymentService.url_for_qr(amount)).thenReturn(expectedResponse);

            PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

            assertNotNull(response, "Response should not be null for amount: " + amount);
            assertEquals("PENDING", response.getStatus(), "Status should be PENDING for amount: " + amount);
            verify(paymentService).url_for_qr(amount);
        }
    }

    @Test
    void testInitiatePayment_ResponseStructure() {
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(testAmount);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                testSessionId,
                "PENDING",
                "http://morzio.com/pay/" + testSessionId.toString()
        );

        when(paymentService.url_for_qr(testAmount)).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertNotNull(response, "Response should not be null");
        assertNotNull(response.getSessionId(), "SessionId should not be null");
        assertNotNull(response.getStatus(), "Status should not be null");
        assertNotNull(response.getPaymentUrl(), "PaymentUrl should not be null");
        assertFalse(response.getSessionId().toString().isEmpty(), "SessionId should not be empty");
        assertFalse(response.getStatus().isEmpty(), "Status should not be empty");
        assertFalse(response.getPaymentUrl().isEmpty(), "PaymentUrl should not be empty");
    }

    @Test
    void testInitiatePayment_NullAmount() {
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto();
        request.setAmount(null);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                UUID.randomUUID(),
                "PENDING",
                "http://morzio.com/pay/test"
        );

        when(paymentService.url_for_qr(null)).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertNotNull(response, "Response should not be null even with null amount");
        verify(paymentService, times(1)).url_for_qr(null);
    }

    @Test
    void testInitiatePayment_VerifyServiceCalled() {
        Long amount = 5000L;
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(amount);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                UUID.randomUUID(),
                "PENDING",
                "http://morzio.com/pay/test"
        );

        when(paymentService.url_for_qr(amount)).thenReturn(expectedResponse);

        paymentController.initiatePayment(request);

        verify(paymentService, times(1)).url_for_qr(amount);
        verifyNoMoreInteractions(paymentService);
    }

    @Test
    void testInitiatePayment_LargeAmount() {
        Long largeAmount = Long.MAX_VALUE;
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(largeAmount);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                testSessionId,
                "PENDING",
                "http://morzio.com/pay/" + testSessionId.toString()
        );

        when(paymentService.url_for_qr(largeAmount)).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertNotNull(response, "Response should not be null for large amount");
        assertEquals("PENDING", response.getStatus(), "Status should be PENDING");
        verify(paymentService).url_for_qr(largeAmount);
    }

    @Test
    void testInitiatePayment_ZeroAmount() {
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(0L);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                testSessionId,
                "PENDING",
                "http://morzio.com/pay/" + testSessionId.toString()
        );

        when(paymentService.url_for_qr(0L)).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertNotNull(response, "Response should not be null for zero amount");
        verify(paymentService).url_for_qr(0L);
    }

    @Test
    void testInitiatePayment_NegativeAmount() {
        Long negativeAmount = -1000L;
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(negativeAmount);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                testSessionId,
                "PENDING",
                "http://morzio.com/pay/" + testSessionId.toString()
        );

        when(paymentService.url_for_qr(negativeAmount)).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertNotNull(response, "Response should not be null for negative amount");
        verify(paymentService).url_for_qr(negativeAmount);
    }

    @Test
    void testInitiatePayment_VerifyPaymentUrlFormat() {
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(testAmount);
        String expectedUrl = "http://morzio.com/pay/" + testSessionId.toString();
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                testSessionId,
                "PENDING",
                expectedUrl
        );

        when(paymentService.url_for_qr(testAmount)).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertEquals(expectedUrl, response.getPaymentUrl(), "Payment URL should match the expected format");
    }

    @Test
    void testInitiatePayment_RequestBodyNotNull() {
        PaymentInitiateRequestDto request = new PaymentInitiateRequestDto(testAmount);
        PaymentInitiateResponseDto expectedResponse = new PaymentInitiateResponseDto(
                testSessionId,
                "PENDING",
                "http://morzio.com/pay/" + testSessionId.toString()
        );

        when(paymentService.url_for_qr(request.getAmount())).thenReturn(expectedResponse);

        PaymentInitiateResponseDto response = paymentController.initiatePayment(request);

        assertNotNull(response, "Response should not be null");
        verify(paymentService).url_for_qr(request.getAmount());
    }
}
