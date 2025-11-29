package com.morzio.server.services.PaymentService;

import com.morzio.server.dtos.PaymentService.PaymentInitiateResponseDto;

public interface PaymentService {
    PaymentInitiateResponseDto url_for_qr(Long amount);
}
