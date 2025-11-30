package com.morzio.server.dtos.PlaidService;

import java.time.LocalDate;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class MLTransactionDto {
    private Double amount;
    private LocalDate date;
    private String category;
}