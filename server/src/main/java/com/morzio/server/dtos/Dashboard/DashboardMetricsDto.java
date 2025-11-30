package com.morzio.server.dtos.Dashboard;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class DashboardMetricsDto {
    private Double totalRevenue;
    private Double pendingAmount;
    private Integer activePlansCount;
    private Double completionRate;
    private String changePercentage;
}
