package com.morzio.server.dtos.PlaidService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class TransactionDto {
    private String transactionId;
    private String name;
    private Double amount;
    private LocalDate date;
    private String category;
    private String merchantName;
    private Boolean pending;
}
