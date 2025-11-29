package com.morzio.pos.data.api

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface MorzioApi {
    @POST("api/payment/initiate")
    suspend fun initiatePayment(@Body request: PaymentInitiateRequest): Response<PaymentInitiateResponse>
}
