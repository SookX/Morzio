package com.morzio.pos.data.api

import com.google.gson.annotations.SerializedName

data class PaymentInitiateRequest(
    @SerializedName("amount") val amount: Long
)

data class PaymentInitiateResponse(
    @SerializedName("sessionId") val sessionId: String,
    @SerializedName("status") val status: String,
    @SerializedName("paymentUrl") val paymentUrl: String
)
