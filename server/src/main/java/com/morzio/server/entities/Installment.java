package com.morzio.server.entities;


import jakarta.persistence.Entity;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;


@Entity
@Table(name = "installment")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Installment {
    private Long id;
    
    private Long amount;
    
    private String status;
    
    @ManyToOne
    @JoinColumn(name = "installment_plan_id")
    private InstallmentPlan installmentPlan;
}