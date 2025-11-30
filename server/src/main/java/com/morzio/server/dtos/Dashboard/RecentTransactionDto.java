package com.morzio.server.dtos.Dashboard;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import java.sql.Timestamp;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class RecentTransactionDto {
    private String id;
    private String customerName;
    private Double amount;
    private String status;
    private Timestamp timestamp;
}
