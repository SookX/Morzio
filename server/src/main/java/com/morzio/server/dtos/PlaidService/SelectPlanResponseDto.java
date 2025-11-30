package com.morzio.server.dtos.PlaidService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SelectPlanResponseDto {
    private boolean success;
    private String paymentId;
    private String message;
}
