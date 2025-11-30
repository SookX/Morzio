package com.morzio.server.dtos.PlaidService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PlaidOnSuccessResponseDto {
    private Boolean success;
    private String message;
    private List<InstallmentPlanDto> installmentPlans;
    private List<TransactionDto> transactions;
    private String error;
}
