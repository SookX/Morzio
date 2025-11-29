package com.morzio.pos.models

enum class PaymentStatus(val displayText: String) {
    WAITING("Waiting for customer..."),
    MANDATE_PENDING("Customer viewing options..."),
    AI_EVALUATION("Processing payment..."),
    INSTALLMENT_PENDING("Completing..."),
    COMPLETED("Payment successful"),
    ERROR("Payment failed"),
    TIMEOUT("Transaction timed out")
}
