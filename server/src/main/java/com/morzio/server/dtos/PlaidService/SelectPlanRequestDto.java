package com.morzio.server.dtos.PlaidService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SelectPlanRequestDto {
    private String sessionId;
    private Double downPaymentAmount;
    private Integer installments;
}
