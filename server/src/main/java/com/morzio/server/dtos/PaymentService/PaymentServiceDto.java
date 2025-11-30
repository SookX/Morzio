package com.morzio.server.dtos.PaymentService;

import java.sql.Timestamp;
import java.util.UUID;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PaymentServiceDto {
    private UUID id;
    private String status;
    private Long amount;
    private Timestamp created;
    private java.util.List<InstallmentDto> installments;
}
