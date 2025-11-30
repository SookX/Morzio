package com.morzio.server.dtos.PlaidService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class InstallmentPlanDto {
    private Integer installments;
    private Double amountPerInstallment;
    private String label;
}
