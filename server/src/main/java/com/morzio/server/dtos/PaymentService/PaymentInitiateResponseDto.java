package com.morzio.server.dtos.PaymentService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.UUID;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PaymentInitiateResponseDto {
    private UUID sessionId;
    private String status;
    private String paymentUrl;
    private java.util.List<InstallmentDto> installments;
}
