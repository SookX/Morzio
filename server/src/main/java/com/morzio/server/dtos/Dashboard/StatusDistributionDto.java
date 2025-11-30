package com.morzio.server.dtos.Dashboard;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class StatusDistributionDto {
    private String status;
    private Integer count;
    private Double percentage;
}
