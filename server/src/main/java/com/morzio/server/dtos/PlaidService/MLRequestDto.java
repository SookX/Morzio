package com.morzio.server.dtos.PlaidService;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class MLRequestDto {
    private List<MLTransactionDto> transactions;
    private double transaction_amount;
    private long transaction_mcc = 5411;
}