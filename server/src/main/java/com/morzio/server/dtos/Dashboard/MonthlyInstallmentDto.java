package com.morzio.server.dtos.Dashboard;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MonthlyInstallmentDto {
    private String month;
    private Double amount;
}
