package com.morzio.server.services.PaymentService;

import com.morzio.server.dtos.PaymentService.PaymentInitiateResponseDto;

import java.util.UUID;
import com.morzio.server.dtos.PaymentService.PaymentServiceDto;

public interface PaymentService {
    PaymentInitiateResponseDto url_for_qr(Long amount);

    PaymentServiceDto getPaymentSession(UUID id);
}
