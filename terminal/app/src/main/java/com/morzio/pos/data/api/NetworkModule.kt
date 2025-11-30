package com.morzio.pos.data.api

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object NetworkModule {
    // Use 10.0.2.2 for Android emulator (maps to host localhost)
    // Use your machine's IP for physical device testing
    // Production: https://morzio.com/
    private const val BASE_URL = "http://10.0.2.2:8080/"

    val api: MorzioApi by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(MorzioApi::class.java)
    }
}
