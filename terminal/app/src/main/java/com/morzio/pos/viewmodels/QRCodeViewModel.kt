package com.morzio.pos.viewmodels

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.morzio.pos.models.PaymentStatus
import com.morzio.pos.utils.QRCodeGenerator
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.roundToLong

data class QRCodeState(
    val sessionId: String = "",
    val amount: Double = 0.0,
    val qrCodeBitmap: Bitmap? = null,
    val status: PaymentStatus = PaymentStatus.WAITING,
    val timeElapsed: Int = 0
)

class QRCodeViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(QRCodeState())
    val uiState: StateFlow<QRCodeState> = _uiState.asStateFlow()

    private var pollingJob: Job? = null
    private var timerJob: Job? = null

    fun initialize(amountString: String) {
        val amount = amountString.replace(",", ".").toDoubleOrNull() ?: 0.0
        val amountInCents = (amount * 100).roundToLong()

        viewModelScope.launch {
            try {
                val response = com.morzio.pos.data.api.NetworkModule.api.initiatePayment(
                    com.morzio.pos.data.api.PaymentInitiateRequest(amount = amountInCents)
                )

                if (response.isSuccessful && response.body() != null) {
                    val body = response.body()!!
                    _uiState.update {
                        it.copy(
                            amount = amount,
                            sessionId = body.sessionId,
                            qrCodeBitmap = QRCodeGenerator.generate(body.paymentUrl, 300, 300),
                            status = PaymentStatus.WAITING
                        )
                    }
                    startPolling()
                    startTimer()
                } else {
                    _uiState.update { it.copy(status = PaymentStatus.ERROR) }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                _uiState.update { it.copy(status = PaymentStatus.ERROR) }
            }
        }
    }

    private fun startPolling() {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            delay(5000) // 5s
            _uiState.update { it.copy(status = PaymentStatus.MANDATE_PENDING) }
            delay(5000) // 10s
            _uiState.update { it.copy(status = PaymentStatus.AI_EVALUATION) }
            delay(5000) // 15s
            _uiState.update { it.copy(status = PaymentStatus.INSTALLMENT_PENDING) }
            delay(6000) // 21s
            _uiState.update { it.copy(status = PaymentStatus.COMPLETED) }
        }
    }

    private fun startTimer() {
        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            for (i in 1..300) {
                delay(1000)
                _uiState.update { it.copy(timeElapsed = i) }
            }
            pollingJob?.cancel()
            _uiState.update { it.copy(status = PaymentStatus.TIMEOUT) }
        }
    }

    fun cancelTransaction() {
        pollingJob?.cancel()
        timerJob?.cancel()
        _uiState.update { it.copy(status = PaymentStatus.ERROR) }
    }

    override fun onCleared() {
        pollingJob?.cancel()
        timerJob?.cancel()
        super.onCleared()
    }
}