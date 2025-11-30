package com.morzio.pos.data.api

import com.google.gson.annotations.SerializedName

data class PaymentInitiateRequest(
    @SerializedName("amount") val amount: Long
)

data class PaymentInitiateResponse(
    @SerializedName("sessionId") val sessionId: String,
    @SerializedName("status") val status: String,
    @SerializedName("paymentUrl") val paymentUrl: String,
    @SerializedName("installments") val installments: List<InstallmentDto>?
)

data class InstallmentDto(
    @SerializedName("amount") val amount: Long,
    @SerializedName("status") val status: String,
    @SerializedName("dueDate") val dueDate: String
)
