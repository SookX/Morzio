package com.morzio.server.dtos.PlaidService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PlaidOnSuccessRequestDto {
    private String publicToken;
    private Double transactionAmount;
}
