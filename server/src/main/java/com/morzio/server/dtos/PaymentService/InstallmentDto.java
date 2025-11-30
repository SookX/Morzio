package com.morzio.server.dtos.PaymentService;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class InstallmentDto {
    private Long amount;
    private String status;
    private String dueDate; // We might need to add due date to the entity later, but for now let's assume
                            // we can calculate or it exists
}
