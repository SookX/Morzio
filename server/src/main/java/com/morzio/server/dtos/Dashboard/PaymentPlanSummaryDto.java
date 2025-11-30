package com.morzio.server.dtos.Dashboard;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PaymentPlanSummaryDto {
    private String id;
    private String customerName;
    private Double totalAmount;
    private Integer installmentCount;
    private Integer paidCount;
    private String nextDueDate;
    private String status;
}
