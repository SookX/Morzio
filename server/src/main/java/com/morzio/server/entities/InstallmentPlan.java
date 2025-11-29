package com.morzio.server.entities;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;


@Entity
@Table(name = "installment_plan")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class InstallmentPlan {
    @Id
    private Long id;

    // weekly
    private Long installmentCount;

    private Long amountPerInstallment;

    private Long totalAmount;

    private Long interestRate;

    @OneToOne
    @JoinColumn(name = "session_id")
    private PaymentSession paymentSession;
}
