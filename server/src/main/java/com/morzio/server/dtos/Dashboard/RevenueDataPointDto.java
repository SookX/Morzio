package com.morzio.server.dtos.Dashboard;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class RevenueDataPointDto {
    private String month;
    private Double total;
}
