package com.morzio.server.dtos.PaymentService;

public class PaymentInitiateRequestDto {
    private Long amount;

    public PaymentInitiateRequestDto() {
    }

    public PaymentInitiateRequestDto(Long amount) {
        this.amount = amount;
    }

    public Long getAmount() {
        return amount;
    }

    public void setAmount(Long amount) {
        this.amount = amount;
    }
}
